"""
Chance levels and trivial-predictor controls for the synthetic KB tasks.

WHY THIS EXISTS (dev_notes/experiment.log.md 2026-09-13, "contre-expertise"):
several months of conclusions in this project compared held-out accuracy
against the WRONG chance level -- the uniform-over-vocabulary rate (~1.5-3%
at vocab_size=64) -- and therefore read a plateau of 25% / 32-34% as "well
above chance, so the mechanism partially composes". It was not. The model
never chooses among the vocabulary: it copies a value that is present in the
episode's KB (measured: 97-98% of predictions). The correct reference is the
CONDITIONAL chance level 1/n_facts, and the plateaus matched it exactly:

    n_distractors=1 -> n_facts=3 -> 1/3 = 33.3%   observed 32-34%
    n_distractors=2 -> n_facts=4 -> 1/4 = 25.0%   observed 24.7-25.5%

Reporting `final_acc` alone cannot distinguish "learned something partial"
from "learned to copy an arbitrary KB value". Every run must therefore report,
alongside accuracy:

  - `conditional_chance`   = 1 / n_facts, the rate of a model that retrieves
                             SOME KB value without discriminating which;
  - `pred_in_kb_rate`      = fraction of predictions landing on any value
                             present in the episode's KB -- near 1.0 means
                             the comparison above is the relevant one;
  - trivial-predictor accuracies, measured empirically on the same data, for
    the structural shortcuts a model can exploit WITHOUT following any chain
    (see `trivial_baselines`). A model is only doing the task when it beats
    all of them by a clear margin.

The shortcuts are real and were mistaken for partial learning: at
`n_hops=3, n_distractors=1` the answer is one of the two values that never
appear as a key, so 50% is reachable with zero hops -- which is what the
baseline's 42-46% actually was.

Both synthetic tasks share the 4-token fact layout
`[KEY_MARK, key_id, VAL_MARK, value_id]` (data/kb_retrieval.py,
data/kb_chain_retrieval.py), so key/value slots are read positionally here.
"""

import torch

KEY_SLOT = 1
VAL_SLOT = 3
FACT_LEN = 4


def fact_keys_values(leaves: torch.Tensor, n_facts: int):
    """(B, N) leaves -> (keys, values), each (B, n_facts), real facts only."""
    keys = leaves[:, KEY_SLOT::FACT_LEN][:, :n_facts]
    values = leaves[:, VAL_SLOT::FACT_LEN][:, :n_facts]
    return keys, values


def prediction_stats(preds: torch.Tensor, labels: torch.Tensor,
                     leaves: torch.Tensor, n_facts: int) -> dict:
    """Per-batch counts for accuracy and `pred_in_kb_rate`."""
    _, values = fact_keys_values(leaves, n_facts)
    in_kb = (preds.unsqueeze(1) == values).any(dim=1)
    return {
        "correct": (preds == labels).sum().item(),
        "in_kb": in_kb.sum().item(),
        "total": labels.shape[0],
    }


def trivial_baselines(ds, batch_size: int = 256, n_batches: int = 8,
                      seed: int = 12345) -> dict:
    """
    Measures, on freshly sampled episodes of `ds`, the accuracy of predictors
    that use NO retrieval and NO chaining. Any of these matching the model's
    accuracy means the model has not demonstrated the capability under test.

      random_kb_value : uniform over the episode's fact values (= 1/n_facts
                        in expectation) -- the "copies some KB value" model.
      non_key_value   : uniform over values that never appear as a key in the
                        episode. On a chain task the final answer is always
                        such a value, so this shortcut skips every hop.
      first_hop_value : the value of the fact whose key equals the query, i.e.
                        one hop only. NOT a shortcut -- it is the exact
                        solver when n_hops == 1, and a progress marker
                        otherwise ("how much of the chain is reachable in a
                        single hop"). Reported separately, and deliberately
                        excluded from the margin computed in
                        `format_report`, which only counts predictors that
                        use no retrieval at all.

    `ds` must expose `sample_batch`, `n_facts` (and, for chain tasks,
    `n_hops`). Deterministic given `seed`.
    """
    gen = torch.Generator().manual_seed(seed)
    n_facts = ds.n_facts
    hits = {"random_kb_value": 0, "non_key_value": 0, "first_hop_value": 0}
    total = 0

    for _ in range(n_batches):
        leaves, _, _, query, label = ds.sample_batch(batch_size)
        keys, values = fact_keys_values(leaves, n_facts)
        B = label.shape[0]
        total += B

        # random_kb_value
        pick = torch.randint(0, n_facts, (B,), generator=gen)
        hits["random_kb_value"] += (values.gather(1, pick[:, None])[:, 0] == label).sum().item()

        # non_key_value: uniform over values not appearing as a key
        is_key = (values.unsqueeze(2) == keys.unsqueeze(1)).any(dim=2)   # (B, n_facts)
        weights = (~is_key).float()
        # an episode where every value is also a key falls back to uniform
        weights[weights.sum(dim=1) == 0] = 1.0
        pick = torch.multinomial(weights, 1, generator=gen)[:, 0]
        hits["non_key_value"] += (values.gather(1, pick[:, None])[:, 0] == label).sum().item()

        # first_hop_value: follow exactly one hop from the query
        match = (keys == query[:, :1])                                    # (B, n_facts)
        has_match = match.any(dim=1)
        idx = match.float().argmax(dim=1)
        one_hop = values.gather(1, idx[:, None])[:, 0]
        hits["first_hop_value"] += ((one_hop == label) & has_match).sum().item()

    out = {k: v / total for k, v in hits.items()}
    out["conditional_chance"] = 1.0 / n_facts
    out["vocab_chance"] = 1.0 / ds.total_vocab_size
    return out


def format_report(acc: float, pred_in_kb_rate: float, baselines: dict,
                  n_hops: int = None) -> str:
    """One block, printed by every training script, so no future reader can
    compare `final_acc` to the wrong reference by accident.

    The margin counts only the NO-RETRIEVAL shortcuts (`random_kb_value`,
    `non_key_value`). `first_hop_value` is reported next to them but never
    enters the margin: on a single-hop task it IS the task's exact solver, so
    including it would flag a perfect model as "not above trivial".
    """
    shortcut_best = max(baselines["random_kb_value"], baselines["non_key_value"])
    margin = acc - shortcut_best
    verdict = "ABOVE every no-retrieval shortcut" if margin > 0.05 else \
              "NOT above the no-retrieval shortcuts -- do not read this as partial learning"
    hop_note = "exact solver for this task" if n_hops == 1 else "first hop only"
    return (
        "--- chance-level report (learn/indexed_attention/eval_metrics.py) ---\n"
        f"final_acc:            {acc:.4f}\n"
        f"pred_in_kb_rate:      {pred_in_kb_rate:.4f}   (near 1.0 => compare against conditional_chance, not vocab_chance)\n"
        f"conditional_chance:   {baselines['conditional_chance']:.4f}   (1/n_facts: retrieves a KB value without discriminating)\n"
        f"vocab_chance:         {baselines['vocab_chance']:.4f}   (uniform over vocabulary -- almost never the right reference)\n"
        f"shortcut/random_kb:   {baselines['random_kb_value']:.4f}   (no retrieval)\n"
        f"shortcut/non_key:     {baselines['non_key_value']:.4f}   (no retrieval, skips every hop)\n"
        f"probe/first_hop:      {baselines['first_hop_value']:.4f}   ({hop_note}; excluded from the margin)\n"
        f"margin_over_shortcut: {margin:+.4f}   -> {verdict}\n"
        "---------------------------------------------------------------------"
    )
