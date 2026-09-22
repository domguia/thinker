"""
Character-level vocabulary for Thinker's optional "answer_chars" output
stream (2026-09-22, user-proposed diagnostic).

Motivation (raw/... discussion, 2026-09-22): the 'answer' stream's head
(nn.Linear(d_model, vocab_size)) is, at this project's tiny d_model, the
single biggest parameter block by far when vocab_size is large (already
noted in core/indexed_thinker_model.py's own docstring, §13). That makes it
hard to tell whether an improvement on the answer-stream CE comes from the
Student actually having learned the answer's *content*, or just from the
big softmax head fitting surface statistics of a large subword vocabulary.
Reading the SAME underlying signal out through a small, fixed, human-
readable character alphabet instead removes almost all of that head
capacity (a few hundred rows instead of tens/hundreds of thousands) and
lets the raw decoded string be read directly -- a much sharper probe of
whether the core recurrent loop + SM readout actually carries the answer,
independent of the projection head's bias.

This is deliberately CE-only, never KD (project default is KD, see this
project's CLAUDE.md "Training methodology default: KD, not pure CE" --
this is the explicit, stated exception under clause (b): the Teacher
produces logits over ITS OWN subword vocabulary, not over this closed
character alphabet, so there is no way to get Teacher targets for this
stream at all). Flag this explicitly at every call site that trains it,
don't let it default silently.

Not byte-level (would need multi-byte UTF-8 reassembly for accented
characters, extra bookkeeping for no benefit yet) and not full Unicode
(unbounded -- defeats the point of a small head). A small closed alphabet:
printable ASCII + the accented Latin letters likely to appear in this
project's data (French prompts/answers alongside the mostly-English
OpenR1-Math-220k/HotpotQA sources), a handful of common math/physics
symbols (OpenR1-Math answers can contain these, unlike HotpotQA's plain-
text answers -- confirmed by scanning real data, see below), plus 4
special symbols. Anything else maps to <unk> rather than crashing -- if
that turns out to lose too much information in practice, extend
`_EXTRA_LATIN`/`_MATH_SYMBOLS` (the model's own head width must move in
lockstep with `vocab_size`, so this is a deliberate, visible constant to
edit, not something to make dynamic/data-driven).

`encode()` NFKC-normalizes first (2026-09-22, after coverage-testing this
alphabet) -- folds compatibility variants that are visually/semantically
the same character into the form already in this alphabet (e.g. the
superscript "²" -> "2", full-width "Ａ" -> "A", the "fi" ligature ->
"f"+"i"), for free, before falling back to <unk>. It does NOT fold
genuinely distinct symbols ("π" stays "π", not "pi") -- those have to be
covered explicitly by `_MATH_SYMBOLS` instead.

**Coverage check performed 2026-09-22** (djm's concern: is 130-ish
characters actually enough, or will real answers silently degrade to
<unk>?): scanned every `answer` field in the only real data available
locally at the time -- a 60-example real HotpotQA sample (/tmp/hotpot_
real_sample, /tmp/hotpot_real_labeled.jsonl) and this project's own
data/distill/synthetic_composition/{train,val}.jsonl (French) -- 1120
answers, 9618 characters total, **zero** OOV. No real OpenR1-Math-220k
data was available locally to scan the same way (the actual training data
lives on Grid'5000, see dev_notes/grid5000_usage.log.md) -- re-run
`CharVocab.scan_coverage()` (below) against the real reasoning jsonl
before trusting this alphabet for a reasoning run, not just an assumption.
Simulating typical math-answer strings by hand DID surface a real gap
(closed by adding `_MATH_SYMBOLS` here): unicode math symbols like "√2",
"π", "≤"/"≥", "±", "×"/"÷", "∞", "∑", "°" were all OOV before this fix --
LaTeX-style ascii answers ("\\frac{1}{2}", "\\boxed{7}", "x^2") were
already fine (every character involved is plain ASCII, already covered).

No sliding-window / local+global attention split yet, even though answer
strings can run 4-8x longer in characters than in subword tokens (spec
discussion 2026-09-22): OutputStream in sequence_mode already does no
self-attention between output positions (each position only cross-attends
to the SM trajectory independently, see OutputStream's docstring) so a
longer T only means more of that same independent cross-attention, not a
new O(T^2) cost -- full attention is fine to start with. A real cost only
shows up in SM trajectory length (n_step) and in generation being
autoregressive over characters instead of subword tokens (more steps per
answer) -- both are Thinker's existing text-length/step trade-offs and can
be revisited later if/when this stream is used on much longer outputs.
Only the 'answer' stream is a char-stream candidate; per user decision
(2026-09-22) the 'thinking' stream stays token-level -- a reasoning trace
does not need this probe.
"""
import unicodedata

_PRINTABLE_ASCII = [chr(c) for c in range(32, 127)]  # space .. '~'
_EXTRA_LATIN = list("àâäéèêëïîôöùûüçñ ÀÂÄÉÈÊËÏÎÔÖÙÛÜÇÑ")
# 2026-09-22: closes the gap found by simulating OpenR1-Math-style answers
# (see this module's docstring) -- common math/physics symbols that plain
# ASCII + accented Latin doesn't cover and that NFKC normalization can't
# fold into something else (they're distinct symbols, not compatibility
# variants of an ASCII character).
_MATH_SYMBOLS = list("√±≈≠≤≥×÷∞∑∏∫°πθφΔΣμαβγλ")
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class CharVocab:
    """Fixed, non-learned mapping char <-> id. No `special_tokens_map`/
    `save_pretrained` machinery like a HF tokenizer -- this is intentionally
    a plain, tiny, dependency-free class; every method it needs to support
    the `_tokenize_padded`/`_teacher_forced_target` pattern in
    data/prompt_response_dataset.py is defined right here."""

    def __init__(self):
        # dict.fromkeys instead of a plain list+set to dedupe _EXTRA_LATIN
        # against _PRINTABLE_ASCII (the stray space above) while preserving
        # order and stable ids.
        self.itos = list(dict.fromkeys(SPECIAL_TOKENS + _PRINTABLE_ASCII + _EXTRA_LATIN + _MATH_SYMBOLS))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_length: int = None, add_eos: bool = True) -> list:
        """No BOS prepended here (mirrors this project's subword convention,
        see _teacher_forced_target's own leading-pad shift) -- <eos> marks
        the end of the answer so a trained model can learn to stop
        generating, the same role </s>/eos_token_id plays for the subword
        'answer' stream."""
        text = unicodedata.normalize("NFKC", text)
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_eos:
            ids = ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def scan_coverage(self, texts) -> dict:
        """Diagnostic helper (not used at training time): counts, over
        `texts` (an iterable of raw strings, e.g. every `answer` field of a
        real dataset jsonl), how many characters fall outside this alphabet
        after NFKC normalization -- run this against the REAL reasoning/
        retrieval data before trusting this alphabet on it (see this
        module's docstring for what was actually checked so far, and
        against what data). Returns {"n_chars": int, "n_oov": int,
        "oov_counts": Counter}."""
        import collections
        covered = set(self.itos)
        oov_counts = collections.Counter()
        n_chars = 0
        for text in texts:
            for ch in unicodedata.normalize("NFKC", text or ""):
                n_chars += 1
                if ch not in covered:
                    oov_counts[ch] += 1
        return {"n_chars": n_chars, "n_oov": sum(oov_counts.values()), "oov_counts": oov_counts}

    def decode(self, ids) -> str:
        """Stops at the first <eos>, like generation should -- used both for
        teacher-forced sanity-printing during training and for autoregressive
        decoding at inference."""
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
        return "".join(out)
