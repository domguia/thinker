"""Permanent regression test for a severe bug found 2026-09-14 via
train_associative_recall.py's `--latent_reset_at_query` results (a
non-monotonic, near-collapsed `n_memory` sweep that made no sense under the
intended design).

`ToyThinker.forward`'s pre-loop `memory = x` used the FULL, unsliced `x` for
step 0's `attn_compute` call, regardless of `x_reveal_mask[0]` -- masking
only took effect starting from step 1 (applied at the END of each loop
iteration, for the NEXT step). So at step 0, every position of `x` -- every
fact AND the query, in `train_associative_recall.py`'s case -- was visible
simultaneously, defeating the entire point of a staged reveal: exactly the
single-shot-softmax trap `x_reveal_mask` exists to avoid
(dev_notes/indexed_attention_spec.md Sec 9.1). This affected every
`x_reveal_mask` run before the fix, including all of Exp.7's `add`/
`associative_recall` results predating this commit.

This test asserts the direct, falsifiable consequence of the fix: at step
0, with `x_reveal_mask[0]` revealing only certain positions, the model's
output must be EXACTLY unchanged when a position `x_reveal_mask[0]` does
NOT reveal is changed -- if this ever fails again, the bug (or an
equivalent one) has been reintroduced.
"""
import torch

from core.toy_model import ToyThinker


def test_x_reveal_mask_step0_hides_unrevealed_positions():
    torch.manual_seed(0)
    n_facts, vocab, T = 4, 32, 2 * 4 + 1
    model = ToyThinker(
        vocab_size=vocab, max_latent=16, max_input_len=T, max_output_len=1,
        d_model=32, nhead=2, d_hid=64, nlayers=1, n_probe=0, dropout=0.0,
    )
    model.eval()

    reveal_step0 = torch.zeros(1, T, dtype=torch.bool)
    reveal_step0[0, 0] = True
    reveal_step0[0, 1] = True  # step 0 reveals ONLY positions 0,1 (the first fact)

    x = torch.randint(0, vocab, (4, T))
    x_changed = x.clone()
    x_changed[:, -1] = (x_changed[:, -1] + 1) % vocab  # change an UNREVEALED position (the query slot)

    with torch.no_grad():
        out_orig = model(x, target=1, n_latent=4, n_step=1, n_memory=10000,
                         is_full_ar=False, is_output_ar=False, x_reveal_mask=reveal_step0)
        out_changed = model(x_changed, target=1, n_latent=4, n_step=1, n_memory=10000,
                            is_full_ar=False, is_output_ar=False, x_reveal_mask=reveal_step0)

    logits_orig = out_orig[1][:, -1, -1, :]
    logits_changed = out_changed[1][:, -1, -1, :]
    assert torch.equal(logits_orig, logits_changed), (
        "step 0's output changed when only an UNREVEALED position (x_reveal_mask[0]==False) "
        "was modified -- x_reveal_mask is not actually hiding that position at step 0, "
        "the 2026-09-14 bug (or an equivalent regression) is back."
    )


def test_x_reveal_mask_step0_still_sees_revealed_positions():
    """Sanity check for the test above: revealed positions SHOULD matter, so the previous
    test isn't trivially passing because step 0's output is constant regardless of input."""
    torch.manual_seed(0)
    n_facts, vocab, T = 4, 32, 2 * 4 + 1
    model = ToyThinker(
        vocab_size=vocab, max_latent=16, max_input_len=T, max_output_len=1,
        d_model=32, nhead=2, d_hid=64, nlayers=1, n_probe=0, dropout=0.0,
    )
    model.eval()

    reveal_step0 = torch.zeros(1, T, dtype=torch.bool)
    reveal_step0[0, 0] = True
    reveal_step0[0, 1] = True

    x = torch.randint(0, vocab, (4, T))
    x_changed = x.clone()
    x_changed[:, 1] = (x_changed[:, 1] + 1) % vocab  # change a REVEALED position

    with torch.no_grad():
        out_orig = model(x, target=1, n_latent=4, n_step=1, n_memory=10000,
                         is_full_ar=False, is_output_ar=False, x_reveal_mask=reveal_step0)
        out_changed = model(x_changed, target=1, n_latent=4, n_step=1, n_memory=10000,
                            is_full_ar=False, is_output_ar=False, x_reveal_mask=reveal_step0)

    logits_orig = out_orig[1][:, -1, -1, :]
    logits_changed = out_changed[1][:, -1, -1, :]
    assert not torch.equal(logits_orig, logits_changed), (
        "step 0's output did not change when a REVEALED position was modified -- "
        "x_reveal_mask may not be wired to attn_compute at all."
    )
