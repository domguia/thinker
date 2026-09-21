---
name: model-families
description: Registre des familles de modèles (Student/Teacher) du projet Thinker — quel alias `--tokenizer` va avec quel Teacher KD, tailles de vocabulaire, et pièges de compatibilité déjà rencontrés (mismatch vocab silencieux). À consulter avant tout choix de tokenizer/Teacher, tout precompute KD, ou toute décision touchant à une famille de modèle (LFM2/OLMo/Qwen).
---

# Familles de modèles — Student (`--tokenizer`) et Teacher (KD)

Source de vérité pour le code : `core/model_families.py` (`MODEL_FAMILIES`,
`resolve_model_name`). Source de vérité pour le raisonnement de sélection :
`dev_notes/model_selection_small_vocab_reasoning.md`. Ce fichier est un résumé
d'accès rapide — en cas de doute, relire ces deux-là, pas seulement ce résumé.

## Règle d'or

**Le Top-K KD (`topk_kd_loss`) indexe directement les logits du student aux
mêmes indices que le Top-K du Teacher.** Ça n'a de sens QUE si Student et
Teacher partagent EXACTEMENT le même tokenizer/vocab. Un `--tokenizer` et un
`--model_dir`/Teacher de vocabulaires différents ne crashent pas forcément —
ils désalignent silencieusement les indices (déjà rencontré 2x : mismatch
`qwen`/`Qwen3.8-27B-FP8` le 2026-09-20, mismatch tokenisation en-contexte vs
autonome `eval_causal_control.py` le 2026-09-21). Toujours vérifier
`vocab_size`/`len(tokenizer)` des deux côtés avant un nouveau couplage
Student/Teacher, pas après.

## Table de compatibilité (alias `--tokenizer` ↔ Teacher KD)

| Alias | ID HuggingFace (Student) | Vocab | Teacher KD compatible | Statut |
|---|---|---|---|---|
| `lfm2` (défaut) | `LiquidAI/LFM2-350M` | 64 400 | `LiquidAI/LFM2-1.2B` (v1, PAS "Thinking") | ✅ utilisé pour tous les résultats actuels (flagship, general, reasoning) |
| `olmo` | `allenai/OLMo-2-0425-1B` | 100 278 | N'importe quel OLMo-2/3 (1B→32B, vocab identique sur toute la famille, vérifié) | Pas encore de Teacher utilisé en pratique |
| `qwen` | `Qwen/Qwen3-0.6B` | 151 643 | **AUCUN Teacher actuellement téléchargé ne correspond** | ❌ ne pas coupler à `qwen_big` |
| `qwen_big` | `Qwen/Qwen3.8-27B-FP8` (en fait un checkpoint **Qwen3.5** VLM utilisé en texte) | 248 077/248 320 | C'est lui-même le Teacher, pas un alias Student autonome | Teacher déjà téléchargé, `qwen_big` sert à le nommer explicitement côté `--tokenizer` du precompute |
| `qwen35` | `Qwen/Qwen3.5-0.8B` | 248 320 (identique 0.8B→397B-A17B, vérifié) | `qwen_big` (même famille Qwen 3.5) | ✅ ajouté 2026-09-21 pour résoudre le mismatch `qwen`/`qwen_big` sans retélécharger de Teacher |

**Piège déjà rencontré** : `qwen` (Qwen3-0.6B, 151 643) et `qwen_big`
(Qwen3.8-27B-FP8, 248 077/248 320) sont TOTALEMENT incompatibles malgré le nom
proche — vérifié directement le 2026-09-20 (`dev_notes/experiments/
distillation.md:250`), IDs différents sur une phrase test identique. Pour un
Teacher Qwen, utiliser `qwen35`/`qwen_big` ensemble, jamais `qwen` avec
`qwen_big`.

## Precompute repr-KD : toujours capturer la dernière couche

`learn/distill/precompute_teacher_targets.py --hidden_layers` accepte le mot
`last` (seul ou mélangé dans une liste, ex. `last,8`) qui résout toujours vers
la dernière couche du Teacher CHARGÉ, quel que soit son nombre réel de
couches — ne jamais coder en dur un numéro de couche en supposant qu'il
correspond à "la dernière" pour un Teacher donné (LFM2-1.2B a 16 couches,
donc `--hidden_layers 16` = last SEULEMENT pour ce Teacher précis).

## Ordre d'expérimentation retenu (2026-09-16)

1. **LFM2** — petit vocab, petite taille, déjà opérationnel de bout en bout.
2. **OLMo 2/3** — vocab garanti stable 1B→32B, aucun run encore lancé.
3. **Qwen** — en dernier, comparaison malgré vocab large ; nécessite `qwen35`
   côté Student pour matcher `qwen_big` côté Teacher (voir table ci-dessus).

Étendre à OLMo/Qwen est un chantier de generalisation distinct du résultat
central (flagship LFM2) — vérifier le statut/priorité courant avant de
lancer du calcul dessus (peut être hors scope selon la deadline active).
