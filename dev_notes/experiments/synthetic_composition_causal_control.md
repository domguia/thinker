# Contrôle causal sur dataset synthétique composition (nécessité absolue par construction)

Contexte : le contrôle causal fin sur HotpotQA (voir `prompt_response_pipeline.md`, entrées 2026-09-20) n'a trouvé aucun signal net de récupération ciblée une fois le confond de comptage retiré (t=0.315 poolé) -- HotpotQA-distractor n'a aucun exemple à 1-hop natif, la nécessité des documents "supporting" n'y est que statistique. `learn/distill/prepare_synthetic_composition_data.py` (long-term-memory-builder, patché `is_supporting` par model-design, commit `550eb07`) construit un dataset où la nécessité est ABSOLUE par construction : groupe COMPOSITION (num_hops=2, Doc A+B tous deux strictement nécessaires, chaîne entité→pont→lieu) vs groupe CONTRÔLE (num_hops=1, un seul doc suffit), mêmes distracteurs indépendants.

Données : `data/distill/synthetic_composition/{train,val}.jsonl` (`--n_per_group 2000 --n_distractors 8 --seed 0`) -- 3600 train (1795 composition / 1805 contrôle), 400 val (205 composition / 195 contrôle).

## 2026-09-21 — Run CE pur (point secondaire, non comparable au flagship)

Note méthodologique : la première recommandation (CE pur) était une erreur -- corrigée par `model-design` (relais utilisateur) car ça introduit un facteur de confusion supplémentaire (régime d'entraînement) alors que le contrôle causal cherche justement à isoler un effet potentiellement régularisant. **Ce run CE pur est gardé comme point secondaire non comparable**, la comparaison qui compte est le run KD (config flagship) en cours.

Config : `d_model=256, n_head=4, n_step=4, use_ff, batch_size=32, bf16, compile, max_steps=6000, seed=0`, CE pur (pas de `--teacher_targets`), `abacus4-1` (A40).

Meilleur checkpoint (`checkpoints/synth_composition_best.pt`) : `val_answer=1.1966` (surapprentissage sévère ensuite, final=3.70 à step 6000 -- pattern habituel, checkpoint best utilisé pour le contrôle causal ci-dessous).

### Contrôle causal fin (`eval_causal_control.py --fine_grained`, comptage égalisé)

**Ensemble (val complet, n=400)** :
| condition | val_answer |
|---|---|
| documents réels | 1.1951 |
| tous mélangés | 1.4015 |
| supporting seul mélangé | 1.4037 |
| distracteurs seuls mélangés | 1.2065 |
| distracteurs mélangés (comptage égalisé au nb de supporting) | 1.1970 |

Diff appariée (supporting - distracteur_apparié), n=400 : **mean=0.2066, std=0.3942, se=0.0197, t=10.480**.

**Groupe COMPOSITION (num_hops=2, nécessité absolue des 2 docs, n=205)** :
diff appariée (comptage égalisé) : **mean=0.2248, std=0.4376, se=0.0306, t=7.355**
(degradation supporting-real=0.2273 vs distracteur_apparié-real=0.0041)

**Groupe CONTRÔLE (num_hops=1, nécessité absolue d'1 seul doc, n=195)** :
diff appariée (comptage égalisé) : **mean=0.2054, std=0.4047, se=0.0290, t=7.088**
(degradation supporting-real=0.2086 vs distracteur_apparié-real=-0.0051)

**Lecture** : contraste total avec HotpotQA (t=0.315 poolé, aucun signal) -- ici, **signal de récupération ciblée massif et net dans les DEUX groupes** (t=7.09 à t=7.36, bien au-delà du seuil de bruit), avec un comptage de documents corrompus rigoureusement égalisé. Corrompre le(s) document(s) réellement nécessaire(s) dégrade fortement la réponse (+0.21 à +0.23), corrompre un nombre égal de distracteurs n'a quasiment aucun effet (-0.005 à +0.004, indistinguable de zéro). **Confirme que le modèle EST capable de récupération ciblée réelle lorsque la nécessité documentaire est absolue et sans ambiguïté** -- l'absence de signal sur HotpotQA s'explique donc bien par la nécessité seulement statistique de ce dataset (les distracteurs y contiennent souvent des indices partiels ou une structure similaire, diluant le signal), pas par une incapacité structurelle de l'architecture. Cohérent avec le diagnostic synthétique positif (I4) déjà noté par `model-design`. **Résultat obtenu sur le run CE pur (secondaire) -- à confirmer/renforcer par le run KD (config flagship) en cours.**

Relayé à `model-design`.
