# Plan d'expérimentation & de recherche — Indexed Attention (au-delà du MVP)

Objectif : passer du MVP testé uniquement en local sur CPU (`dev_notes/indexed_attention_spec.md` §11) à une validation progressive sur GPU (Grid'5000), alignée sur la vision complète du projet (séparer raisonnement et mémoire, cf. §-1 de la spec) plutôt que de s'arrêter à la mécanique de base. Chaque phase a un objectif de recherche précis, un coût croissant, et une décision go/no-go avant de passer à la suivante — on ne scale pas avant d'avoir confirmé que la phase précédente marche.

Référence pour tout ce qui est cluster/GPU : skill `grid5000` + `dev_notes/grid5000_usage.log.md`. Réutiliser les paliers matériels déjà validés dans le chantier de distillation existant (`dev_notes/experiment.log.md`) plutôt que de redécouvrir les contraintes de zéro :
- **Dry-run / petite échelle** : `abacus3`/`abacus10` (A5000 24G) ou `abacus22` (A40 45G) — peu demandés, démarrage rapide.
- **KD / échelle moyenne** : `abacus26` (2×L40S 45G) ou `chuc` (4×A100 40G).
- **À éviter** : `abacus1/2` (P100), `drac`, `chiclet` — pas de BF16/FlashAttention.

## Phase 0 — Sanity check GPU du MVP actuel (le plus rapide, à faire en premier)

**But** : confirmer que `HierarchicalMemory`/`IndexedThinker` (déjà validés sur CPU, tests unitaires verts) tournent correctement et efficacement sur GPU réel à une échelle un peu plus significative que le test de sur-apprentissage CPU (8 exemples, 4 facts) — et obtenir de premiers chiffres de débit/mémoire.

- Étendre `data/kb_retrieval.py` : plus de facts distracteurs (64-256), hiérarchie plus profonde (depth 3-4), `d_model` plus grand (256-512).
- Un seul GPU du palier dry-run, budget ~15-30 min (convention `program.md`).
- Comparer explicitement `depth>0` (hiérarchique) vs `depth=0` (Baseline C plate) à cette échelle plus grande — sur le test CPU minuscule les deux triviallement sur-apprennent, ici on doit commencer à voir la hiérarchie apporter un avantage réel (ou pas — signal important si elle n'en apporte pas).
- **Go/no-go** : la hiérarchie doit au moins égaler la baseline plate en accuracy avec moins de FLOPs/mémoire ; sinon, retour à la conception avant de continuer.
- Logguer en `EXP-XXX` (convention déjà utilisée dans `dev_notes/experiment.log.md`).

## Phase 1 — Baselines A/B/C (isoler ce qui contribue réellement)

**But** : implémenter concrètement les trois baselines de la spec §9 comme simples variantes de configuration d'`IndexedThinker`, pas des modèles séparés :
- **Baseline A** : `n_step=1` (un seul passage, pas de boucle).
- **Baseline B** : accès mémoire désactivé (`o_kb`/`o_sm` forcés à zéro), boucle intacte.
- **Modèle complet** : les deux actifs.

Réutiliser la tâche de la Phase 0. Vérifier le pattern attendu : B doit échouer sur la récupération, le modèle complet doit réussir. C'est la première vraie preuve (même synthétique) que la boucle + la mémoire externe apportent chacune quelque chose.

## Phase 1bis — Balayage des variantes architecturales (le point de départ de tout ce projet)

**But** : cette idée est née avec plusieurs variantes explicitement destinées à être testées (`dev_notes/ideas/branch_indexed_attention_synthesis.md` §4, `indexed_attention_comparison.md`) — les phases précédentes ne couvrent que des *ablations* (Baselines A/B/C : boucle/mémoire activées ou non), pas les variantes du *mécanisme* lui-même. À faire ici, sur les tâches synthétiques bon marché des Phases 0/2 (pas besoin d'attendre les données réelles) puisque c'est justement là que les itérer est le moins coûteux.

**Variantes déjà câblées, zéro code à ajouter — juste lancer les runs :**
- **$M=1$ vs $M>1$** : `n_slots` existe déjà de bout en bout (`LevelCompressor` → `HierarchicalMemory` → `IndexedThinker`). Comparer `n_slots=1` (idée initiale de l'utilisateur) vs `n_slots∈{2,4}` sur la tâche multi-sauts (Phase 2) — c'est justement là que plusieurs "aspects sémantiques" par bloc pourraient aider.
- **Confirmation, pas une variante à trancher** : $Q_{KB}$ et $Q_{SM}$ sont déjà des projections découplées (`HierarchicalMemory.q_proj` vs `IndexedThinker.sm_q_proj`) — la suggestion de Gemini jamais confirmée par l'utilisateur (comparaison §2, axe 7) est en fait déjà le comportement par défaut de l'implémentation actuelle, pas quelque chose à ajouter.

**Variantes nécessitant un petit ajout de code (flag, pas de nouvelle architecture) :**
- **Stop-gradient sur les clés SM** (spec §4.1, jamais tranché — "plus de stop gradient" reste ambigu) : ajouter un flag `detach_sm_keys` sur l'écriture SM d'`IndexedThinker` (`new_k.detach()` avant l'append) et comparer aux deux lectures possibles de la phrase source.
- **Dropout stochastique des niveaux hauts** (§11bis idée propre, non implémentée) : masquer aléatoirement les niveaux compressés pendant l'entraînement dans `HierarchicalMemory.attend()`, probabilité croissante avec le niveau — tester si ça améliore la robustesse multi-échelle sur la tâche multi-sauts.

**Variantes structurellement plus lourdes — à isoler dans une comparaison dédiée plus tard, pas dans ce balayage :**
- **Softmax unifié (actuel) vs fusion par branches à gating façon NSA** : nécessiterait une seconde implémentation quasi complète du mécanisme d'attention (branches séparées + gate appris) — trop coûteux pour un simple balayage, à traiter comme sa propre comparaison A/B architecture-complète si les résultats du reste du plan motivent la question.
- **Résolution de niveau implicite (actuelle) vs routeur explicite** : dépend du mécanisme de largeur adaptative (§7.1 de la spec), déjà hors scope avant la Phase 7.

**Go/no-go** : si `n_slots>1` ou le dropout de niveaux n'apportent aucun gain mesurable sur la tâche multi-sauts, les garder en configuration par défaut la plus simple (`n_slots=1`, pas de dropout) pour la suite du plan plutôt que d'ajouter de la complexité non justifiée empiriquement.

## Phase 2 — Tâche synthétique multi-sauts (le vrai test de la thèse du projet)

**But** : tester directement l'hypothèse centrale (séparer raisonnement et mémoire permet un raisonnement plus long à budget de calcul inférieur) sur une tâche qui *nécessite* plusieurs itérations, pas juste une récupération simple.

- Étendre `data/kb_retrieval.py` en variante "chaîne de faits" : la valeur du fait $i$ est la clé du fait $i+1$ (2 à 4 sauts), toujours synthétique donc peu coûteux.
- Faire varier $N_{\text{step}}$ (1 à 6) et comparer accuracy vs Baselines A/B sur ce split — le modèle complet doit profiter de plus d'itérations, Baseline A doit plafonner tôt (un seul passage ne peut pas chaîner), Baseline B doit échouer partout (pas de faits en mémoire).
- **Go/no-go** : si l'accuracy ne s'améliore pas avec $N_{\text{step}}$ croissant sur les tâches à 3-4 sauts, c'est un signal que la mécanique de fusion KB/SM (§6.1, actuellement une simple projection linéaire) est probablement le goulot — retour à la conception avant de passer à des données réelles.

## Phase 3 — Passage à des données textuelles réelles (toujours sans Teacher)

**But** : valider `HierarchicalMemory` sur un vocabulaire et une distribution de tokens réels, pas juste le petit vocabulaire synthétique (32-64 tokens) des phases précédentes.

- Réutiliser `scripts/fetch_wiki.py`/`data/wiki_samples.json` (déjà dans le repo) comme source de "faits" (paragraphes courts), avec des paires question/réponse extractives simples.
- Tokenizer réel (ex. GPT-2 ou un petit BPE existant), vocab_size réaliste (10k-50k).
- Reprend le protocole Baselines A/B/C de la Phase 1 sur ce nouveau jeu de données.

## Phase 4 — Introduction d'un vrai Teacher (distillation, Option A "black-box" d'abord)

**But** : brancher le stream de sortie `answer` sur une vraie distillation, en réutilisant l'infrastructure KD déjà construite et validée dans `learn/distill/` plutôt que de repartir de zéro :
- `learn/distill/precompute_teacher_targets.py` : précalcul des cibles Top-K du Teacher (déjà utilisé pour EXP-003 à EXP-006).
- `topk_kd_loss()` de `learn/distill/train_sft.py` : la perte KD Top-K est agnostique à l'architecture de l'étudiant (elle ne prend que des logits en entrée) — directement réutilisable pour `IndexedThinker.streams['answer']`.
- Option A (black-box, perte uniquement sur les tokens finaux) d'abord — c'est ce vers quoi l'utilisateur penchait déjà dans la conversation source, et c'est la baseline MVP la moins coûteuse en ingénierie de données (pas de pipeline d'annotation de trajectoires). Option B (supervision explicite Compute/Retrieve par étape, §11bis) reste une extension si la convergence en Option A est trop lente.
- **Attention au piège FP8 déjà rencontré sur ce projet** (`dev_notes/experiment.log.md`, bug de dequantization) : vérifier `model.is_quantized`/l'agrément top-1 sur un petit échantillon avant de faire confiance aux cibles précalculées si le Teacher est chargé en FP8.

## Phase 5 — Stratégie de construction de la KB : passer de l'espace unifié à la Sub-KB par batch

**But** : le MVP utilise la stratégie 1 (espace unifié input/KB, §8) — simple mais avec un risque de dilution documenté. Une fois sur données réelles, tester la stratégie 2 (Sub-KB générée à la volée par batch + distracteurs négatifs, plus stable par construction) en A/B contre la stratégie 1 sur le même split d'évaluation, pour objectiver si le risque de dilution est réel à cette échelle ou négligeable.

## Phase 6 — Montée en échelle du modèle avec muP

**But** : éviter de re-tuner LR/init à chaque palier de taille, en réutilisant la discipline déjà validée dans le chantier de distillation (`apply_mup_init`, `build_mup_param_groups`, `apply_depth_mup_scaling` de `learn/distill/train_sft.py`), adaptée aux noms de modules d'`IndexedThinker`.
- **Leçon déjà apprise à ne pas reproduire** (`dev_notes/experiment.log.md`, section muP) : le LR optimal trouvé au plus petit palier ne transfère pas correctement si profondeur *et* largeur augmentent simultanément sans correction — appliquer Depth-muP-lite dès le premier palier de scaling, pas seulement muP en largeur.
- Paliers suggérés : reprendre les tiers déjà dimensionnés pour la distillation (40M / 150M / 500M-core) plutôt que d'en inventer de nouveaux.

## Phase 7 — Benchmark multi-sauts réel (HotpotQA / MuSiQue)

**But** : le test décisif de la spec §9 sur données réelles, plus seulement synthétiques. Comparer Baselines A/B/C au modèle complet sur les splits "raisonnement pur" / "récupération pure" / "multi-sauts" (§9, tableau des datasets proposés : GSM8K/MATH pour le raisonnement pur, NIAH/Wikipedia pour la récupération pure, HotpotQA/MuSiQue pour le multi-sauts).

## Phase 8 — Multi-output streams pour de vrai

**But** : ajouter un stream `thinking` en embedding (§11bis), aligné sur une couche médiane (~40-65% de profondeur) d'un vrai Teacher via une perte de similarité/MSE, avec le curriculum d'extinction progressive mentionné dans les notes (non formalisé — à définir empiriquement à ce stade : probable décroissance linéaire du poids de cette perte, à valider).

## Phase 9 — Sanity check généraliste

**But** : vérifier que le modèle reste un assistant généraliste cohérent (pas seulement un spécialiste de la récupération) — sanity check léger sur MT-Bench/AlpacaEval, pas un objectif principal à ce stade du projet mais un garde-fou pour détecter une sur-spécialisation.

## Ce que ce plan laisse volontairement de côté (cf. spec §7)

No-Op/largeur adaptative, dropout stochastique de niveaux, curriculum de largeur "large→étroit" — toutes reportées après la Phase 7, une fois la mécanique de base (hiérarchie + boucle + streams + KD) validée sur données réelles. Les introduire plus tôt ajouterait des variables libres avant d'avoir une base fiable pour juger si elles aident (même risque que celui identifié par l'utilisateur lui-même dans la conversation source pour la baseline MVP complète).
