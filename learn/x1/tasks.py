"""X1 (thesis/research/X1_DISPATCH.md): synthetic algorithmic task generators
for testing length/iteration generalization (H2). Shared digit/symbol-level
vocab across tasks (tiny, no tokenizer dependency -- avoids the vocab-head
memory cost that dominated every prior experiment this session).

Each task exposes `generate(n_examples, size_range, seed) -> list[Example]`
where Example = (input_ids: list[int], position_ids: list[int], target_ids:
list[int], input_len: int) -- target_ids is what the model must produce
(teacher-forced during training, exact-match-scored during eval). position_ids
implement the "digit place-value" positional scheme (McLeish et al. 2024
"Abacus embeddings" simplified variant, per X1_DISPATCH table: "Abacus /
indices de position de chiffre") -- same place value -> same position id,
so the position embedding table itself doesn't need to grow with sequence
length, the mechanism the length-generalization literature identifies as
necessary for OOD extrapolation on these tasks.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# Shared vocab: digits 0-9, then task-specific symbols appended.
DIGITS = list("0123456789")
PAD, BOS, EOS, PLUS, EQUALS, TIMES, SEP = "<pad>", "<bos>", "<eos>", "+", "=", "*", ","
VOCAB = DIGITS + [PAD, BOS, EOS, PLUS, EQUALS, TIMES, SEP]
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
PAD_ID = TOK2ID[PAD]
VOCAB_SIZE = len(VOCAB)


@dataclass
class Example:
    input_ids: list
    position_ids: list
    target_ids: list  # same length as input_ids; -100 where not supervised (teacher-forced next-token targets, HF convention shifted by caller)
    prompt_len: int  # length of the "input" portion (question), rest is the answer to supervise/score


def _digits_of(n: int) -> list:
    return [int(c) for c in str(n)]


def _encode_number_reversed(n: int) -> list:
    """Least-significant digit first (standard trick for grade-school-algorithm
    learnability in the length-generalization literature, e.g. McLeish 2024)."""
    return list(reversed(_digits_of(n)))


def _place_value_positions(n_digits: int, start: int = 0) -> list:
    """position i (from the right) -> place value i, so the SAME position id
    is reused across examples regardless of total sequence length."""
    return list(range(start, start + n_digits))


def gen_addition(n_examples: int, digit_range: tuple, seed: int, position_offset_max: int = 0) -> list:
    """T1: A + B = C, A/B independently sampled with digit_range[0]..digit_range[1]
    digits each. Reversed digit order (LSD-first) for both operands and answer.

    position_offset_max (McLeish et al. 2024 "Abacus embeddings" -- the actual
    mechanism that makes place-value position ids generalize OOD): a random
    offset in [0, position_offset_max] is added to every position id in an
    example. Without this, absolute place-value ids beyond digit_range's max
    are NEVER SEEN during training (a 20-digit-max train set only ever uses
    position ids 0-19), so the position embedding table has nothing learned
    for OOD place values and cannot generalize -- the offset randomization is
    what exposes the FULL position range needed at test time, in expectation,
    across many short training examples, even though any single example is
    still short. Set to (max_test_digits - digit_range[1]) or more when
    calling for train data; 0 (no offset) for eval/test data (always start at
    the true place value there)."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n_examples):
        da = rng.randint(*digit_range)
        db = rng.randint(*digit_range)
        a = rng.randint(10 ** (da - 1) if da > 1 else 0, 10 ** da - 1)
        b = rng.randint(10 ** (db - 1) if db > 1 else 0, 10 ** db - 1)
        c = a + b
        offset = rng.randint(0, position_offset_max) if position_offset_max > 0 else 0

        a_digits = _encode_number_reversed(a)
        b_digits = _encode_number_reversed(b)
        c_digits = _encode_number_reversed(c)

        ids = [TOK2ID[str(d)] for d in a_digits] + [TOK2ID[PLUS]] + \
              [TOK2ID[str(d)] for d in b_digits] + [TOK2ID[EQUALS]]
        pos = _place_value_positions(len(a_digits), offset) + [offset + len(a_digits)] + \
              _place_value_positions(len(b_digits), offset) + [offset + max(len(a_digits), len(b_digits)) + 1]
        prompt_len = len(ids)

        target_ids = [TOK2ID[str(d)] for d in c_digits] + [TOK2ID[EOS]]
        target_pos = _place_value_positions(len(target_ids), offset)

        full_ids = ids + target_ids
        full_pos = pos + target_pos
        examples.append(Example(input_ids=full_ids, position_ids=full_pos,
                                 target_ids=[-100] * prompt_len + target_ids, prompt_len=prompt_len))
    return examples


def gen_prefix_sum_parity(n_examples: int, len_range: tuple, seed: int, position_offset_max: int = 0) -> list:
    """T3: cumulative binary parity (running XOR) of a random 0/1 sequence.
    input: b_0 b_1 ... b_{L-1} SEP ; target: p_0 p_1 ... p_{L-1} EOS, where
    p_i = XOR(b_0..b_i). This is the "prefix sums" task from Schwarzschild
    et al. 2021's length-generalization work (parity is the binary variant).
    position_offset_max: see gen_addition's docstring."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n_examples):
        L = rng.randint(*len_range)
        bits = [rng.randint(0, 1) for _ in range(L)]
        offset = rng.randint(0, position_offset_max) if position_offset_max > 0 else 0
        ids = [TOK2ID[str(b)] for b in bits] + [TOK2ID[SEP]]
        pos = _place_value_positions(L, offset) + [offset + L]
        prompt_len = len(ids)

        parity = []
        acc = 0
        for b in bits:
            acc ^= b
            parity.append(acc)
        target_ids = [TOK2ID[str(p)] for p in parity] + [TOK2ID[EOS]]
        target_pos = _place_value_positions(len(target_ids), offset)

        full_ids = ids + target_ids
        full_pos = pos + target_pos
        examples.append(Example(input_ids=full_ids, position_ids=full_pos,
                                 target_ids=[-100] * prompt_len + target_ids, prompt_len=prompt_len))
    return examples


TASKS = {"addition": gen_addition, "prefix_sum": gen_prefix_sum_parity}
