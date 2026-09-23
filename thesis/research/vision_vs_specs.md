# Vision vs. Specs — Thinker

Comparaison de la vision du projet (9 points) avec l'état actuel des specs/notes du dépôt.
Sources consultées : `README.md`, `program.md`, `dev_notes/indexed_attention_spec.md`,
`dev_notes/ideas/*.md`, `thesis/notes.md`, `agents/OBJECTIVES_LOG.md`.

## V1 — Pourquoi les boucles (AR/diffusion = pas répétés du même modèle, capacité mal condensée)

**PARTIAL.** L'idée de calcul itératif via un bloc réutilisé (poids partagés, boucle latente) est bien
au cœur de l'architecture (`README.md` — « layer weight sharing to allow reuseable compute block »,
`dev_notes/ideas/ideas-draft.md`). Mais l'argumentaire explicite « AR et diffusion partagent le même
principe de découpage en petits pas » et « les petits modèles modernes battent GPT-3 → la capacité est
mal condensée » n'apparaît nulle part comme justification écrite — c'est une motivation implicite,
jamais formulée dans les specs ou `thesis/notes.md`.

## V2 — Cœur récurrent latent, le modèle décide où/comment scaler son calcul

**COVERED** (`dev_notes/indexed_attention_spec.md`, `dev_notes/ideas/ideas-draft.md`,
`README.md`). Le raisonnement en espace latent (pas en espace token) via un bloc récurrent partagé est
la description centrale de l'architecture depuis le README (« hidden latent vector as information
passing », cross-attention). `ideas-draft.md` §43 évoque même varier le budget de calcul et la
séquence d'appels de blocs selon la tâche (avec RL comme piste). L'idée que le modèle *décide*
dynamiquement combien d'itérations faire reste cependant à l'état d'idée/piste, pas de mécanisme
spécifié et validé (curriculum `n_step` actuel = choix externe/aléatoire, pas appris par le modèle).

## V3 — Paramètres/connaissance hors du cœur, dans une mémoire indexée externe interrogée

**COVERED** (`dev_notes/indexed_attention_spec.md`, section sur `HierarchicalMemory` /
`core/indexed_memory.py`). C'est le chantier principal actif du dépôt : séparation explicite
core (calcul) / KB hiérarchique (faits), formalisée en détail (§5 et suivants du spec, statut
"Chantier principal actif" dans `README.md`).

## V4 — Mémoire organisée hiérarchiquement, auto-organisation par domaine, guidage sinon

**PARTIAL.** La structure hiérarchique elle-même est spécifiée et implémentée
(`dev_notes/indexed_attention_spec.md` §5, `HierarchicalMemory`/`LevelCompressor`). L'idée de
*guider* l'organisation via des embeddings d'un modèle pré-entraîné (clustering) apparaît bien
dans `dev_notes/ideas/retrievial_training_design.md` (assigner des tokens latents à des clusters
d'embeddings Wikipedia) et `thesis/notes.md:133` (« cluster targets embedding with knn »). Mais
il n'y a nulle part de vérification/attente écrite que la hiérarchie *s'auto-organise par domaine*
(coding, médecine, etc.) — ni de protocole pour tester cette hypothèse spécifique ; c'est une piste
de guidage évoquée en marge, pas un point de vision formulé et suivi tel quel.

## V5 — Angle inférence/infra : pas toute la mémoire résidente, contrairement au MoE, prefetch prédictible

**MISSING.** Aucune occurrence de "prefetch", "MoE"/mixture-of-experts, ni d'argumentaire
"chargement prédictible du domaine vs appels d'experts imprévisibles" dans le spec, les ideas
notes, `thesis/notes.md`, ou `README.md`/`program.md`. La distinction mémoire-résidente vs
mémoire-à-charger, et la comparaison avec le trafic mémoire imprévisible du MoE, n'est documentée
nulle part dans le dépôt — c'est un angle qui n'a pas encore été couché à l'écrit.

## V6 — Petits modèles fine-tunés battent les gros → bruit de connaissance ; ne récupérer que le savoir utile

**PARTIAL.** L'intuition apparaît en germe dans `ideas-draft.md:260` (utiliser le "teacher" petit
modèle seulement dans son domaine d'entraînement pour ne pas perturber) et dans la distinction
KB/streams du spec (les faits ne doivent pas se diffuser dans les poids du cœur — c'est justement
pour éviter le bruit). Mais l'argument "fine-tuned small models beat big ones → knowledge noise"
comme motivation explicite pour l'indexation n'est écrit nulle part tel quel.

## V7 — Signal d'entraînement faible → boosters (KD, output streams auxiliaires, supervision précoce, puis RL)

**COVERED** (`thesis/notes.md:202,267` — constat explicite du signal faible ; `dev_notes/ideas/ideas-draft.md:281`
— proposition de "stream parallèle" renforcé par le raisonnement, puis retrait progressif, puis RL ;
`dev_notes/indexed_attention_spec.md` mentionne les "output streams" comme composant à part entière ;
KD est la méthodologie par défaut du projet, cf. `CLAUDE.md`). Les quatre boosters (KD, streams
auxiliaires greffés/retirés, supervision précoce, RL ensuite) sont tous documentés quelque part,
bien que dispersés entre plusieurs fichiers plutôt que synthétisés en un seul endroit.

## V8 — Mémoire auto-générée : le modèle encode un document (token spécial) en mémoire KV, une autre instance l'utilise (+ distracteurs) comme KB, gradient de bout en bout

**COVERED** (`dev_notes/indexed_attention_spec.md`, section "ingestion"/`Thinker.ingest`,
`sm_write_proj`, `add_static_level`, `ingest_documents()` dans
`learn/indexed_attention/train_prompt_response.py`). Le mécanisme d'ingestion de document en KV via
un token dédié, la ré-interrogation en QA, et le gradient de bout en bout (y compris via gradient
checkpointing stateless pour rester correct) sont spécifiés et implémentés en détail. Le point
"distracteurs" spécifique (documents non pertinents mélangés à la KB pour forcer une vraie
récupération) est mentionné comme piste ouverte (densité/diversité "type contrastive learning",
§ "non implémentés, toujours [OUVERT]") mais pas encore un protocole d'expérience concret.

## V9 — Tâches arithmétiques jouets (addition/multiplication sur digits) doivent fonctionner

**COVERED** (`thesis/notes.md` — résultats concrets : addition base 16, ~97% accuracy ; `data/numbers.py`
curriculum génératif ; multiplication citée comme tâche cible dès `thesis/notes.md:17-18`). Ces
toy tasks sont le premier banc de test historique du projet et ont déjà des résultats chiffrés.

---

## Résumé rapide

| Point | Statut |
|---|---|
| V1 | PARTIAL |
| V2 | COVERED |
| V3 | COVERED |
| V4 | PARTIAL |
| V5 | MISSING |
| V6 | PARTIAL |
| V7 | COVERED |
| V8 | COVERED |
| V9 | COVERED |
