"""
§8ter multi-hop protocol (2026-09-20, long-term-memory-builder), fallback step:
HotpotQA's distractor config has num_hops>=2 on EVERY example (verified directly on
data/distill/hotpotqa/val.jsonl: min=2, distribution {2:1394,3:438,4:143,5:19,6:5,8:1}) --
there is no natural "1 doc suffices" control group in that dataset, so stratifying by
num_hops there cannot show anything (confirmed empirically: answer_hops_le1_n=0 on every
val batch of the real run). This script builds a SYNTHETIC dataset with both groups
present by construction, output in the same jsonl schema RetrievalPromptDataset/
prepare_retrieval_data.py already consume (question/answer/context/context_docs/
is_supporting/text/num_tokens/num_hops/id).

`is_supporting` (2026-09-21, model-design): per-doc boolean aligned with context_docs,
True for the doc(s) structurally REQUIRED to answer (both A/B in composition, the single
direct doc in control), False for distractors -- added so eval_causal_control.py's
targeted/count-matched corruption (built for HotpotQA's is_supporting) works unmodified
on this dataset. Unlike HotpotQA, necessity here is exact by construction (not just
statistically likely), giving a higher-power causal control for the same question:
does the model's advantage over noctx come from genuinely targeted retrieval of the
documents it needs, or from generic sensitivity to processing coherent text?

Two groups, same surface structure, same distractor pool, only the number of documents
genuinely REQUIRED to answer differs:
- COMPOSITION (num_hops=2): Doc A states "<entity> <rel1> <bridge>.", Doc B states
  "<bridge> <rel2> <final>." -- the final value never appears with the entity directly,
  only via the bridge; both docs are necessary.
- CONTROL (num_hops=1): a single doc states "<entity> <rel1+rel2> <final>." directly.

Distractors: same template family, independently-sampled entities/bridges/finals from
the same pools, so a lexical shortcut ("one rare shared word") can't trivially solve the
question without actually resolving the bridge chain -- same spirit as HotpotQA's own
distractors. Entities/places are invented (never real-world names) so neither the
tokenizer nor a pretrained Teacher has prior lexical/factual purchase on them.
"""
import argparse
import json
import random

REL1 = ["est née a", "a grandi dans le village de", "a commence sa carriere a"]
REL2 = ["se situe dans le pays de", "fait partie de la region de", "appartient au territoire de"]
REL_DIRECT = [
    ("est née a", "se situe dans le pays de", "est née dans le pays de"),
    ("a grandi dans le village de", "fait partie de la region de", "a grandi dans la region de"),
    ("a commence sa carriere a", "appartient au territoire de", "a commence sa carriere dans le territoire de"),
]

PERSON_SYLL = ["Kav", "Sen", "Dor", "Mira", "Tol", "Vex", "Ona", "Ruk", "Fen", "Zia"]
# combinatorial (prefix x suffix), not a fixed short list -- with n_distractors up to ~8-10
# per example, a dozen-entry fixed pool exhausts itself fast; prefix*suffix gives thousands
# of distinct invented place names instead, so "used_places" exclusion never starves sampling.
PLACE_PREFIX = ["Sendo", "Velka", "Norha", "Quinta", "Ombre", "Astal", "Thorn", "Vask",
                "Glind", "Korven", "Elthi", "Draum", "Panora", "Kestrel", "Vorna", "Ilyth"]
PLACE_SUFFIX = ["rak", "stan", "via", "ra", "lune", "veil", "moor", "hara", "mark", "ris", "mond", "dan"]


def make_name(rng, syllables, n=2):
    return "".join(rng.choice(syllables) for _ in range(n)).capitalize()


def make_place(rng, used_places, max_tries=50):
    for _ in range(max_tries):
        name = rng.choice(PLACE_PREFIX) + rng.choice(PLACE_SUFFIX)
        if name not in used_places:
            return name
    raise RuntimeError("place name pool exhausted -- increase PLACE_PREFIX/PLACE_SUFFIX or lower n_distractors")


def sample_triple(rng, used_places):
    entity = make_name(rng, PERSON_SYLL, 2)
    bridge = make_place(rng, used_places)
    final = make_place(rng, used_places | {bridge})
    return entity, bridge, final


def build_composition_example(rng, idx, n_distractors):
    entity, bridge, final = sample_triple(rng, set())
    rel1, rel2 = rng.choice(list(zip(REL1, REL2)))
    doc_a = f"{entity} {rel1} {bridge}."
    doc_b = f"{bridge} {rel2} {final}."
    question = f"Dans quel lieu {entity} a-t-elle fini par se retrouver, en suivant son parcours ?"
    slots = [(doc_a, True), (doc_b, True)]
    used = {bridge, final}
    for _ in range(n_distractors):
        d_entity, d_bridge, d_final = sample_triple(rng, used)
        used |= {d_bridge, d_final}
        d_rel1, d_rel2 = rng.choice(list(zip(REL1, REL2)))
        # distractors are themselves split doc_a/doc_b style, same surface form,
        # so a length/format-based shortcut doesn't separate real docs from noise
        d_text = (f"{d_entity} {d_rel1} {d_bridge}." if rng.random() < 0.5
                  else f"{d_bridge} {d_rel2} {d_final}.")
        slots.append((d_text, False))
    rng.shuffle(slots)
    docs = [s[0] for s in slots]
    is_supporting = [s[1] for s in slots]
    context = "\n".join(docs)
    text = f"USER: {question}\nContext:\n{context}\nASSISTANT: {final}"
    return {
        "question": question, "answer": final, "context": context, "context_docs": docs,
        "is_supporting": is_supporting,
        "text": text, "num_tokens": len(text.split()), "num_hops": 2,
        "id": f"synth-composition-{idx}",
    }


def build_control_example(rng, idx, n_distractors):
    entity, _bridge, final = sample_triple(rng, set())
    _rel1, _rel2, rel_direct = rng.choice(REL_DIRECT)
    doc_direct = f"{entity} {rel_direct} {final}."
    question = f"Dans quel lieu {entity} a-t-elle fini par se retrouver, en suivant son parcours ?"
    slots = [(doc_direct, True)]
    used = {final}
    for _ in range(n_distractors):
        d_entity, d_bridge, d_final = sample_triple(rng, used)
        used |= {d_bridge, d_final}
        d_rel1, d_rel2 = rng.choice(list(zip(REL1, REL2)))
        d_text = (f"{d_entity} {d_rel1} {d_bridge}." if rng.random() < 0.5
                  else f"{d_bridge} {d_rel2} {d_final}.")
        slots.append((d_text, False))
    rng.shuffle(slots)
    docs = [s[0] for s in slots]
    is_supporting = [s[1] for s in slots]
    context = "\n".join(docs)
    text = f"USER: {question}\nContext:\n{context}\nASSISTANT: {final}"
    return {
        "question": question, "answer": final, "context": context, "context_docs": docs,
        "is_supporting": is_supporting,
        "text": text, "num_tokens": len(text.split()), "num_hops": 1,
        "id": f"synth-control-{idx}",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_per_group", type=int, default=400, help="examples per group (composition/control)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--n_distractors", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="data/distill/synthetic_composition")
    args = p.parse_args()

    rng = random.Random(args.seed)
    examples = []
    for i in range(args.n_per_group):
        examples.append(build_composition_example(rng, i, args.n_distractors))
        examples.append(build_control_example(rng, i, args.n_distractors))
    rng.shuffle(examples)

    n_val = int(len(examples) * args.val_frac)
    val, train = examples[:n_val], examples[n_val:]

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_comp = sum(1 for r in rows if r["num_hops"] == 2)
        n_ctrl = sum(1 for r in rows if r["num_hops"] == 1)
        print(f"{path}: {len(rows)} examples ({n_comp} composition, {n_ctrl} control)")


if __name__ == "__main__":
    main()
