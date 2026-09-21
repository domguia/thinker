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

## 2026-09-21 — Run KD (seed=0, protocole flagship correct) : bug de harnais trouvé et corrigé, signal ENCORE plus fort

Run KD (`d_model=256, n_head=4, n_step=4, use_ff, kd_alpha=0.5, bf16, compile, max_steps=6000, seed=0`, `--teacher_targets`/`--val_teacher_targets` = precompute combiné top_k32+hidden_layers16 sur train/val synth-composition) -- meilleur checkpoint (`checkpoints/synth_composition_kd_best.pt`) : **val_answer=1.8472 à step 900** (surapprentissage ensuite, val=2.40 à step 6000).

**Bug de harnais trouvé et corrigé (`data/prompt_response_dataset.py:165`, `_resolve_span`)** : quand `--teacher_targets` est actif, la réponse est tokenisée EN CONTEXTE (`_locate_token_span`) ; sans teacher, tokenisation AUTONOME (`_tokenize_padded`). `eval_causal_control.py` ne passait jamais `--teacher_targets`, donc réévaluait tout checkpoint KD avec des labels legèrement décalés par rapport à l'entraînement -- premier essai (sans le flag) donnait `val_answer(réel)=9.7564`, très éloigné du 1.8472 rapporté par le training. Patché en parallèle par model-design (commit `5435a40`, ajout du flag `--teacher_targets` à `eval_causal_control.py`) -- avec le flag et le `val_topk32_hidden.npz` correspondant, `val_answer(réel)=1.8454`, cohérent avec le training (diff de 0.002, bruit numérique normal). **Vérification faite en parallèle par model-design : le contrôle causal historique HotpotQA (t=0.315 poolé, "aucun signal") portait sur des checkpoints CE purs des deux côtés (train ET eval) -- pas de désalignement possible, ce résultat reste valide tel quel, pas besoin de le refaire.**

Pour la décomposition par groupe, split `val_composition.jsonl`/`val_control.jsonl` nécessite son PROPRE precompute (doc_id = position dans le fichier source, même piège que le bug d'alignement repr-KD déjà rencontré sur retrieval#1 -- rappel appliqué directement plutôt que redécouvert) : `val_composition_topk32_hidden.npz` (205 ex) et `val_control_topk32_hidden.npz` (195 ex) précomputés séparément.

### Contrôle causal fin corrigé (comptage égalisé)

**Ensemble (n=400)** : diff appariée (supporting - distracteur_apparié) = **mean=0.2623, std=0.2689, se=0.0134, t=19.510**

**Groupe COMPOSITION (num_hops=2, n=205)** : **mean=0.2597, std=0.2836, se=0.0198, t=13.109**
(degradation supporting-real=0.2627 vs distracteur_apparié-real=0.0030)

**Groupe CONTRÔLE (num_hops=1, n=195)** : **mean=0.2626, std=0.2354, se=0.0169, t=15.580**
(degradation supporting-real=0.2486 vs distracteur_apparié-real=-0.0147)

**Lecture** : signal ENCORE plus fort et net que le run CE (t=13-19 contre t=7-10 précédemment) -- la version KD (protocole correct, comparable au flagship) confirme et renforce la conclusion : récupération ciblée réelle et massive dans les deux groupes dès que la nécessité documentaire est absolue. Reste à répliquer sur 2-3 seeds supplémentaires (demande model-design) avant traitement comme résultat définitif pour le papier -- seeds 1/2 en cours.
