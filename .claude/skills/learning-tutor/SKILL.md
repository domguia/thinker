---
name: learning-tutor
description: Track the user's own comprehension questions and learning moments across all tasks in this project (not specific to any one topic — ML, Grid'5000, tooling, anything), separately from technical/experiment journals. Use whenever the user asks a question aimed at understanding a concept rather than requesting an action (e.g. "why does X work this way", "I don't get why...", "can you explain..."), or explicitly asks to log/review their learning.
---

# Learning tutor (draft v0 — expect to iterate)

Context: the user is doing real production/research work on this project while simultaneously learning the underlying concepts (ML, distillation, infra, etc.) as they go. Comprehension questions asked mid-task are valuable signal about what they're still building intuition for — but they get lost in the flow of a long working session unless captured separately. This skill is that capture mechanism, plus a seed for later self-testing (quizzes, spaced-repetition-style recall) once there's enough logged material to draw from.

**Not yet implemented, deliberately left for a v1 once the log has real content**: automatic quiz/QCM generation, spaced-repetition scheduling, "did you retain this" check-ins. This draft only defines the trigger and the log format — the review/testing tooling comes later, once we see what real entries look like.

## When to trigger

A **comprehension question**: the user is trying to understand *why*/*how* something works, not asking you to *do* something. Signals: "pourquoi...", "comment ça marche...", "je ne comprends pas...", "est-ce que X implique Y ?", "c'est quoi la différence entre...", or any question where a good answer is an explanation rather than an action or a file change.

Not every question qualifies — a quick factual lookup ("quelle est la taille du vocab ?") isn't a comprehension moment by itself. Log it when the user is visibly building or checking a mental model, especially if they follow up, push back, or ask "donc si je comprends bien...".

## What to do when triggered

1. Answer the question normally, in context, as you would anyway.
2. Append an entry to `dev_notes/learning_journal.md` (create it from the template below if it doesn't exist yet) — **a separate file from `dev_notes/experiment.log.md` and `dev_notes/grid5000_usage.log.md`**, since this is about the user's understanding, not the project's technical state.
3. Keep the entry short: the question (paraphrased, not verbatim transcription noise), the core of the explanation (not the full answer — just the concept, so it's useful as a later recall prompt), and a topic tag.
4. Don't force a "did you understand?" confirmation every single time — it gets tedious. Do check in when the topic is dense/foundational, or when the user's own phrasing suggests uncertainty ("je ne sais pas si...", "j'ai l'impression que..."). Record the confirmation status honestly (`confirmed` / `unconfirmed` / `partial`) rather than assuming yes.

## Journal format (`dev_notes/learning_journal.md`)

```markdown
# Learning journal

One entry per comprehension question/moment. Not a technical log — see
experiment.log.md and grid5000_usage.log.md for that. This is about what the
user is building understanding of, for later review/self-testing.

## YYYY-MM-DD — <short topic tag, e.g. "muP / hyperparameter transfer">
- Question: <paraphrased>
- Core explanation: <the key idea, 2-4 sentences, written so it works as a
  standalone recall prompt later — not "see above"/"as discussed">
- Status: confirmed | unconfirmed | partial
- Related: <other topic tags, if this connects to an earlier entry>
```

## Example (drawn from this project's actual session)

```markdown
## 2026-09-04 — Top-K distillation & vocabulary coverage
- Question: With a ~152k-token vocabulary and a small dataset, does a
  reduced Top-K logit capture (K=32) actually cover the tokens that matter —
  or are there vocabulary tokens that never appear at all in a small
  dataset, giving the student zero learning signal for them regardless of K?
- Core explanation: Top-K controls how much of the TEACHER's per-token
  probability mass is captured at each position that DOES appear in the
  data — it says nothing about which vocabulary items appear across the
  dataset. Those are two separate coverage questions: (1) per-token, is K
  large enough to capture the teacher's real uncertainty (usually yes, teacher
  distributions are typically peaked); (2) per-dataset, are there vocabulary
  tokens that never occur in any example — for those, no K helps, because
  the student never sees a training signal touching that token id at all,
  independent of Top-K. (2) is checkable directly: tokenize the dataset,
  count distinct token ids seen, compare to vocab_size.
- Status: unconfirmed (explanation given, waiting on the user's reaction / the
  actual coverage measurement to close the loop)
- Related: distillation-topk-storage-formula
```

## Future ideas (not built yet — revisit once the log has enough entries)

- A small script that reads `learning_journal.md` and generates a short quiz (multiple-choice or short-answer) from `confirmed`/`partial` entries, for periodic self-testing.
- Spaced-repetition scheduling (e.g. resurface an entry after 1 day / 1 week / 1 month) rather than reviewing linearly.
- Cross-linking `Related:` tags into an actual concept graph once there are enough entries for that to be useful rather than premature structure.
