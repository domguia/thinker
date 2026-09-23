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
MOVE_R, MOVE_D, MOVE_U, MOVE_L = ">", "v", "^", "<"
VOCAB = DIGITS + [PAD, BOS, EOS, PLUS, EQUALS, TIMES, SEP, MOVE_R, MOVE_D, MOVE_U, MOVE_L]
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


def gen_p_hop_induction(n_examples: int, hop_range: tuple, seed: int, position_offset_max: int = 0,
                         n_nodes: int = 10) -> list:
    """T4 (X1_DISPATCH.md, Saunshi et al. 2025 arXiv:2502.17416 "p-hop induction"):
    each loop iteration = one reasoning hop, so this task directly probes whether
    Thinker's n_step recurrence composes p atomic lookups, train p<=8 test p 9-32.

    Input encodes a random permutation over n_nodes (single-digit node ids,
    n_nodes<=10 so DIGITS alone suffice) as n_nodes pointer pairs "i -> perm[i]",
    then a query: EQUALS, decimal digits of p, SEP, the start node. Target: the
    single node reached after following the permutation p times from start,
    then EOS. p can exceed n_nodes (the permutation's cycles just wrap), which
    is what makes p>n_nodes-ish OOD values well-defined for the test range."""
    assert n_nodes <= 10, "single-digit node ids only"
    rng = random.Random(seed)
    examples = []
    for _ in range(n_examples):
        perm = list(range(n_nodes))
        rng.shuffle(perm)
        p = rng.randint(*hop_range)
        start = rng.randint(0, n_nodes - 1)
        offset = rng.randint(0, position_offset_max) if position_offset_max > 0 else 0

        ids = []
        for i in range(n_nodes):
            ids += [TOK2ID[str(i)], TOK2ID[str(perm[i])], TOK2ID[SEP]]
        p_digits = _digits_of(p)
        ids += [TOK2ID[EQUALS]] + [TOK2ID[str(d)] for d in p_digits] + [TOK2ID[SEP], TOK2ID[str(start)]]
        prompt_len = len(ids)
        pos = _place_value_positions(prompt_len, offset)

        cur = start
        for _ in range(p):
            cur = perm[cur]
        target_ids = [TOK2ID[str(cur)], TOK2ID[EOS]]
        target_pos = _place_value_positions(len(target_ids), offset)

        full_ids = ids + target_ids
        full_pos = pos + target_pos
        examples.append(Example(input_ids=full_ids, position_ids=full_pos,
                                 target_ids=[-100] * prompt_len + target_ids, prompt_len=prompt_len))
    return examples


def gen_multiplication(n_examples: int, digit_range: tuple, seed: int, position_offset_max: int = 0) -> list:
    """T2 (X1_DISPATCH.md, McLeish et al. 2024): A * B = C, same reversed-digit /
    place-value-position scheme as gen_addition. digit_range applies to BOTH
    operands independently (dispatch table: train 1-5 digits, test OOD up to
    10x10 -- i.e. also digit_range but wider at eval time, position_offset_max=0
    there as with the other tasks)."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n_examples):
        da = rng.randint(*digit_range)
        db = rng.randint(*digit_range)
        a = rng.randint(10 ** (da - 1) if da > 1 else 0, 10 ** da - 1)
        b = rng.randint(10 ** (db - 1) if db > 1 else 0, 10 ** db - 1)
        c = a * b
        offset = rng.randint(0, position_offset_max) if position_offset_max > 0 else 0

        a_digits = _encode_number_reversed(a)
        b_digits = _encode_number_reversed(b)
        c_digits = _encode_number_reversed(c)

        ids = [TOK2ID[str(d)] for d in a_digits] + [TOK2ID[TIMES]] + \
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


def _carve_perfect_maze(n: int, rng: random.Random):
    """Randomized-DFS spanning tree over an n x n grid (Bansal et al. 2022's maze-solving
    setup): guarantees exactly one path between any two cells, so the shortest path from
    (0,0) to (n-1,n-1) is unique and well-defined (no path-choice ambiguity for the target).
    Returns right_open[r][c] (c in 0..n-2, True=passage open between (r,c)-(r,c+1)) and
    down_open[r][c] (r in 0..n-2, True=passage open between (r,c)-(r+1,c))."""
    visited = [[False] * n for _ in range(n)]
    right_open = [[False] * (n - 1) for _ in range(n)]
    down_open = [[False] * n for _ in range(n - 1)]
    stack = [(0, 0)]
    visited[0][0] = True
    while stack:
        r, c = stack[-1]
        neighbors = []
        if c + 1 < n and not visited[r][c + 1]:
            neighbors.append(("R", r, c + 1))
        if c - 1 >= 0 and not visited[r][c - 1]:
            neighbors.append(("L", r, c - 1))
        if r + 1 < n and not visited[r + 1][c]:
            neighbors.append(("D", r + 1, c))
        if r - 1 >= 0 and not visited[r - 1][c]:
            neighbors.append(("U", r - 1, c))
        if not neighbors:
            stack.pop()
            continue
        direction, nr, nc = rng.choice(neighbors)
        if direction == "R":
            right_open[r][c] = True
        elif direction == "L":
            right_open[r][c - 1] = True
        elif direction == "D":
            down_open[r][c] = True
        elif direction == "U":
            down_open[r - 1][c] = True
        visited[nr][nc] = True
        stack.append((nr, nc))
    return right_open, down_open


def _maze_path_moves(right_open, down_open, n: int) -> list:
    """BFS from (0,0) to (n-1,n-1) over the spanning tree's open edges; since it's a tree
    the path is unique, BFS just recovers it (no shortest-path ambiguity to worry about)."""
    from collections import deque

    parent = {}
    move_from_parent = {}
    visited = [[False] * n for _ in range(n)]
    q = deque([(0, 0)])
    visited[0][0] = True
    while q:
        r, c = q.popleft()
        if (r, c) == (n - 1, n - 1):
            break
        cand = []
        if c + 1 < n and right_open[r][c]:
            cand.append((r, c + 1, MOVE_R))
        if c - 1 >= 0 and right_open[r][c - 1]:
            cand.append((r, c - 1, MOVE_L))
        if r + 1 < n and down_open[r][c]:
            cand.append((r + 1, c, MOVE_D))
        if r - 1 >= 0 and down_open[r - 1][c]:
            cand.append((r - 1, c, MOVE_U))
        for nr, nc, mv in cand:
            if not visited[nr][nc]:
                visited[nr][nc] = True
                parent[(nr, nc)] = (r, c)
                move_from_parent[(nr, nc)] = mv
                q.append((nr, nc))
    moves = []
    cur = (n - 1, n - 1)
    while cur != (0, 0):
        moves.append(move_from_parent[cur])
        cur = parent[cur]
    moves.reverse()
    return moves


def gen_labyrinth(n_examples: int, size_range: tuple, seed: int, position_offset_max: int = 0) -> list:
    """T5 (X1_DISPATCH.md, Bansal et al. 2022 "End-to-End Algorithm Synthesis with
    Recurrent Networks": maze solving, train 9x9, test 13x13-33x33): each loop iteration
    can be read as one step of path-following/relaxation, probing the same H2 iteration
    question as T3/T4 but on a spatial (non-linear-chain) reasoning structure.

    size_range is the maze SIDE LENGTH n (n x n grid), NOT a token-sequence length like
    the other tasks' size_range -- callers must set --n_positions from the actual encoded
    sequence length (~2*n*(n-1)+1+path_len), not from n itself, or position ids will run
    out of range for n>~10.

    Input: the maze's spanning-tree connectivity, flattened row-major as one '1'/'0' token
    per potential right-neighbor and down-neighbor edge (1=open passage, 0=wall), then SEP.
    Target: the unique shortest path from (0,0) to (n-1,n-1) as a sequence of move tokens
    (MOVE_R/MOVE_D/MOVE_U/MOVE_L), then EOS. Maze is a perfect maze (spanning tree) so this
    path is unique -- no shortest-path-choice ambiguity in the supervised target."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n_examples):
        n = rng.randint(*size_range)
        offset = rng.randint(0, position_offset_max) if position_offset_max > 0 else 0
        right_open, down_open = _carve_perfect_maze(n, rng)

        ids = []
        for r in range(n):
            for c in range(n):
                if c < n - 1:
                    ids.append(TOK2ID[str(int(right_open[r][c]))])
                if r < n - 1:
                    ids.append(TOK2ID[str(int(down_open[r][c]))])
        ids.append(TOK2ID[SEP])
        prompt_len = len(ids)
        pos = _place_value_positions(prompt_len, offset)

        moves = _maze_path_moves(right_open, down_open, n)
        target_ids = [TOK2ID[m] for m in moves] + [TOK2ID[EOS]]
        target_pos = _place_value_positions(len(target_ids), offset)

        full_ids = ids + target_ids
        full_pos = pos + target_pos
        examples.append(Example(input_ids=full_ids, position_ids=full_pos,
                                 target_ids=[-100] * prompt_len + target_ids, prompt_len=prompt_len))
    return examples


TASKS = {"addition": gen_addition, "prefix_sum": gen_prefix_sum_parity, "p_hop": gen_p_hop_induction,
         "multiplication": gen_multiplication, "labyrinth": gen_labyrinth}
