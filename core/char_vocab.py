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
<unk>?): first pass scanned every `answer` field in the only real data
available locally at the time -- a 60-example real HotpotQA sample and
this project's own data/distill/synthetic_composition/{train,val}.jsonl
(French) -- 1120 answers, 9618 characters, zero OOV. That undersold the
real risk: **run against the ACTUAL Grid'5000 training data**
(data/distill/openr1_math(_full)/train.jsonl, data/distill/hotpotqa(_full)/
train.jsonl, via `oarsh` on a running job, no dedicated reservation needed
-- see dev_notes/grid5000_usage.log.md 2026-09-22) found real, non-zero
OOV: 0.01% of characters on openr1_math_full (314835 chars, 65 OOV -- top
offenders: a literal "\n" inside some multi-line answers, which the
printable-ASCII-only range silently dropped -- a real bug, now fixed by
adding "\n" below -- plus scattered Czech/Chinese/currency characters from
a few non-English/symbolic answers), and 0.12% on hotpotqa_full (1110656
chars, 1328 OOV -- mostly Latin-diacritic proper nouns: en/em dashes,
curly quotes, and letters like š/č/á/í/ū/ř from foreign place/person
names, plus rare Cyrillic/CJK fragments). Simulating typical math-answer
strings by hand ALSO surfaced a gap (closed by adding `_MATH_SYMBOLS`):
unicode math symbols like "√2", "π", "≤"/"≥", "±", "×"/"÷", "∞", "∑", "°"
were all OOV before that fix -- LaTeX-style ascii answers ("\\frac{1}{2}",
"\\boxed{7}", "x^2") were already fine (every character involved is plain
ASCII).

Response to the real-data findings (2026-09-22), in order of what actually
moves the needle: (1) added "\n" and common typographic punctuation
(en/em dash, curly quotes) to the alphabet outright -- cheap, and "\n" in
particular is a genuine content character here, not decoration, so
dropping it silently would be a real information loss, not a rounding
error. (2) added a DIACRITIC-STRIPPING FALLBACK in `_normalize_char`
(NFD-decompose, drop combining marks, retry) for anything still unknown
after that -- turns š->s, č->c, á->a, ī->i, etc. into their plain-ASCII
base letter instead of <unk>, at zero extra vocab cost, since these are
overwhelmingly proper-noun transliterations where the base letter still
carries real information. Deliberately NOT extended to cover the residual
long tail (Cyrillic и/о/е, CJK 当/或/丙, Arabic ا/ل, atomic
non-decomposable Latin letters like ø/ł/ı/æ/ð) -- those are genuine
different-script or non-diacritic characters, not just "one more accent to
add", and the alphabet's whole point is staying small; they degrade to
<unk> by design. Re-ran the same scan after this fix to confirm it
actually helped rather than assuming: openr1_math_full's OOV dropped from
65 to 22 chars (314835 total -- 0.007%, the "\n" cases and most Latin
diacritics gone, only CJK/currency-symbol answers left), hotpotqa_full's
from 1328 to 363 (1110656 total -- 0.033%, only genuine
different-script/atomic-letter cases left, exactly the documented
boundary above). Re-run `CharVocab.scan_coverage()` again after any
further extension of either dataset to confirm the OOV rate stays this
low.

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
# 2026-09-22, found by scanning the REAL Grid'5000 training data (see this
# module's docstring): "\n" is a genuine content character (some OpenR1-Math
# answers span multiple lines) that _PRINTABLE_ASCII's range(32,127) excludes
# outright -- dropping it would silently corrupt those answers, unlike the
# diacritic long tail below which degrades gracefully via the NFD fallback.
# The dashes/quotes are common enough in HotpotQA proper-noun/prose answers
# (en-dash alone: 300 occurrences across the two hotpotqa files) to be worth
# their own slots rather than falling back to <unk> or a stripped ASCII "-"/"'"
# that would lose the distinction.
_PUNCTUATION = ["\n", "–", "—", "‘", "’", "“", "”"]  # \n, en/em dash, ‘’“”
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
        self.itos = list(dict.fromkeys(
            SPECIAL_TOKENS + _PRINTABLE_ASCII + _EXTRA_LATIN + _MATH_SYMBOLS + _PUNCTUATION
        ))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def _normalize_char(self, ch: str) -> str:
        """A single already-NFKC-normalized char, mapped to whatever this
        alphabet actually has a slot for: itself if covered, else its
        diacritic-stripped base letter (NFD-decompose, drop combining marks,
        e.g. "š"->"s", "ī"->"i") if THAT is covered, else left as-is for the
        caller to map to <unk>. 2026-09-22, added after scanning real
        hotpotqa_full/openr1_math_full data found this exact long tail (see
        module docstring) -- covers foreign-name transliterations at zero
        extra vocab cost; deliberately does not attempt anything for a
        different script entirely (Cyrillic, CJK) or an atomic
        non-decomposable Latin letter (ø, ł, ı, æ) -- those stay <unk>."""
        if ch in self.stoi:
            return ch
        stripped = "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
        return stripped if stripped in self.stoi else ch

    def encode(self, text: str, max_length: int = None, add_eos: bool = True) -> list:
        """No BOS prepended here (mirrors this project's subword convention,
        see _teacher_forced_target's own leading-pad shift) -- <eos> marks
        the end of the answer so a trained model can learn to stop
        generating, the same role </s>/eos_token_id plays for the subword
        'answer' stream."""
        text = unicodedata.normalize("NFKC", text)
        ids = [self.stoi.get(self._normalize_char(ch), self.unk_id) for ch in text]
        if add_eos:
            ids = ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def scan_coverage(self, texts) -> dict:
        """Diagnostic helper (not used at training time): counts, over
        `texts` (an iterable of raw strings, e.g. every `answer` field of a
        real dataset jsonl), how many characters fall outside this alphabet
        AFTER the same NFKC + diacritic-stripping pipeline `encode()` itself
        applies -- i.e. what actually reaches <unk> during real encoding,
        not a raw pre-fallback count. Run this against the REAL reasoning/
        retrieval data before trusting this alphabet on it (see this
        module's docstring for what was actually checked so far, and
        against what data). Returns {"n_chars": int, "n_oov": int,
        "oov_counts": Counter}."""
        import collections
        oov_counts = collections.Counter()
        n_chars = 0
        for text in texts:
            for ch in unicodedata.normalize("NFKC", text or ""):
                n_chars += 1
                if self._normalize_char(ch) not in self.stoi:
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
