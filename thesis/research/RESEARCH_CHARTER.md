# RESEARCH CHARTER — Thinker

> Document d'ancrage. **Tout agent (expérience, analyse, rédaction) doit le lire avant d'agir.**
> Il fixe la question scientifique et les hypothèses. Il ne change que sur décision explicite de l'utilisateur (noter la date en bas).
> Détails d'exécution : `thesis/paper/WRITING_PLAN.md` (papier), `agents/OBJECTIVES_LOG.md` (journal).

## 1. Question scientifique

**Peut-on séparer, dans un modèle de langage, le *calcul* (un petit cœur récurrent latent, itéré autant que nécessaire) de la *connaissance* (une mémoire externe indexée et organisée), de sorte que :**
- (a) le compute s'échelonne **à l'intérieur** du modèle (plus d'itérations = plus de raisonnement), sans passer par l'espace des tokens ;
- (b) la connaissance soit chargée **à la demande et de façon prévisible**, ce qui rend l'inférence moins coûteuse en paramètres résidents et en trafic mémoire ?

## 2. Intuitions fondatrices (pourquoi on y croit)

- **I1 — La décomposition en petites étapes est ce qui marche.** Les LM autorégressifs découpent la génération en milliers de petits calculs ; la diffusion découpe une image en nombreuses étapes du même réseau. Dans les deux cas, on réutilise le même modèle sur des sous-problèmes. Thinker rend ce découpage **interne et latent** : le modèle choisit où et combien itérer.
- **I2 — Le compute est mal condensé.** Des modèles de quelques milliards de paramètres dépassent GPT-3 (175B), et un petit modèle fine-tuné bat souvent un gros modèle généraliste sur sa tâche. La connaissance diffuse est du bruit pour une tâche donnée.
- **I3 — La connaissance doit être externe et organisée.** Si un index hiérarchique groupe la connaissance (par domaine : code, médecine…), le modèle peut charger le bon sous-ensemble dans ses premières itérations, le garder, puis ne plus faire que calculer.

## 3. Hypothèses testables

| id | Hypothèse | Test minimal | Statut (2026-09-23) |
|---|---|---|---|
| H1 | Un cœur récurrent partagé, entraîné à n_step aléatoire, extrapole à plus d'itérations | CE/acc vs n_step_test | **Soutenue** (CE plate à 16) — nouveauté à vérifier vs littérature |
| H2 | Plus d'itérations **améliorent** la solution sur des problèmes plus durs (vraie extrapolation algorithmique : addition à plus de chiffres, labyrinthes plus grands, prefix sums) | acc vs taille du problème × n_step | **Non testée proprement** — priorité |
| H3 | La mémoire externe indexée permet le rappel associatif et la composition multi-sauts | associative recall, multi-hop synthétique | Partielle (bug pooling K/V corrigé) |
| H4 | Un index hiérarchique s'**auto-organise** par domaine ; sinon, on peut le guider (clusters d'embeddings de modèles pré-entraînés) | pureté des clusters de l'index vs labels de domaine | **Non testée** |
| H5 | Localité d'accès : les premières itérations fixent le sous-ensemble mémoire utilisé → préchargement prévisible (≠ MoE) | fraction de mémoire touchée, recouvrement des accès entre itérations, octets déplacés | **Non testée** — angle infra (partenaire de Rennes) |
| H6 | Le signal d'apprentissage d'un modèle récurrent est faible ; des « boosters » l'aident : KD, flux de sortie auxiliaires greffés pendant l'entraînement puis retirés, supervision précoce, puis RL | ablation avec/sans booster | Partielle (KD vs CE non concluant) |
| H7 | **Mémoire auto-générée** : le modèle encode un document (token spécial) dans une mémoire KV ; une autre instance l'utilise, parmi des distracteurs, comme base de connaissances long terme ; le gradient traverse jusqu'à l'encodage | QA sur KB auto-encodée vs distracteurs | Tentée, peut-être mal conduite — à refaire proprement |
| H8 | Coût de la récurrence partagée : effondrement de calibration (~0.5 % acc vs 21 % dense) | ablations | **Observé**, cause inconnue |

## 4. Ce que ce n'est PAS (garde-fous pour les agents)
- Pas un concours de qualité de génération libre ou de SOTA sur un benchmark LLM.
- Pas une optimisation d'un transformer dense classique.
- Un résultat négatif **bien caractérisé** a de la valeur ; il ne faut pas le « réparer » en changeant silencieusement de question.
- Toute expérience doit dire quelle hypothèse H1–H8 elle teste.

## 5. Échelle visée
- Court terme : tâches algorithmiques jouets (arithmétique, labyrinthes, recherche associative), comme dans la littérature sur les réseaux récurrents qui extrapolent.
- Moyen terme : un LM « utilisable », d'environ 200M à 1B paramètres (ordre des petits modèles d'appel d'outils), montrant un gain mesurable du mécanisme.

## 6. Contraintes du projet
KD par défaut ; Teacher de même famille (skill `model-families`) ; `--hidden_layers` inclut `last` ; budget faible → économie de tokens et de compute ; cluster Grid'5000 (abacus27 H100 pour le lourd).

## Historique des décisions
- 2026-09-23 : création à partir de la vision exprimée par l'utilisateur (session writing-lead-agent).
