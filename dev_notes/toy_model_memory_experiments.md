# Toy model — la mémoire moyen terme construite à la volée n'a jamais été testée

**[SUPERSEDED 2026-09-13]** Document de passation initial, gardé tel quel comme trace du constat de départ. Le plan actif, à jour (implémentation, corrections de `thinker-5b`, protocole d'exécution) est **`dev_notes/toy_memory_experiment_plan.md`** — le lire en premier. La « réserve à lever » du §2 ci-dessous a été vérifiée par `model-design` : voir §5 de ce fichier ou §1bis de `toy_memory_experiment_plan.md` — le constat tient, elle ne l'affaiblit pas.

**Statut : constat + propositions d'expérience. Rien n'est implémenté.** Document de passation écrit le 2026-09-13 par la session `thinker-5b` (contre-expertise), transmis à `model-design` pour implémentation et à `experiment-manager` pour exécution. Voir « État de la passation » en fin de document.

## 0. Pourquoi ce document

Rôle assigné par l'utilisateur aux premières expériences toy : **vérifier si le « loop reasoning with medium term memory built on the fly » fonctionne**. Ce document établit que cette question n'a jamais été mise à l'épreuve, et propose le protocole pour le faire.

C'est le même angle mort que le bug du compresseur trouvé le même jour (`experiment.log.md`, entrée « Contre-expertise ») : **le mécanisme existe dans le code, rien ne force jamais son usage, et aucune métrique en place ne l'aurait révélé.**

## 1. Où les travaux se sont arrêtés

- **Dernier résultat (Sept 18-19, « Ça marche ! »)** — tâche `cumsum`, loss pondérée sur les steps. Après une pause à l'epoch 30k, la capacité de calcul est augmentée (`n_latent` 16→24, `n_step` 12→16) et le modèle **généralise immédiatement à toutes les capacités**. C'est un résultat d'extrapolation en compute, le plus intéressant de la série toy, et il reste debout (il ne dépend pas du point §2).
- **8 mars — point de décision explicite, pas un échec** : « Should I implement language model distillation task? or keep going with toy task and model? » → distillation choisie. La branche toy est **garée, pas invalidée**.
- **Jamais implémenté** : le « planned flow runner » conçu le 6 janvier (tableau `read_input / mem_lookup / mem_write / static_mem_lookup / output` par step). C'est précisément l'instrument dont la question a besoin — voir Exp. 4.

## 2. Le constat

`core/toy_model.py:205-207` a bien la mécanique :

```python
memory = latents if i >= read_step else [x] + latents
if len(latents) > n_memory: latents.pop(0)   # FIFO
```

Mais :

- `read_step = n_step - 1` est **codé en dur** partout — `scripts/train.py:77`, `scripts/th1nker_runner.py:1044` (dont le commentaire dit « remove on output step », confirmant que l'intention était de ne forcer que la sortie à lire la mémoire) ;
- `n_memory` n'a **jamais** été fixé (défaut `1e4`, FIFO illimitée) ;
- aucune occurrence de `read_step` / `n_memory` dans `experiment.log.md` — aucun balayage n'a jamais été fait.

**Trace de la boucle** (`core/toy_model.py:188-212`) : la mémoire est mise à jour *après* le calcul du latent et *avant* le calcul de la sortie. À l'itération `i`, le calcul du latent consomme la mémoire fixée à la fin de `i-1` ; avec `read_step = n_step - 1`, cette mémoire vaut `[x] + latents` pour tout `i-1 < n_step - 1`. Seule la requête de sortie, au dernier pas, lit `latents` seuls.

**Conclusion : à chaque step de calcul, l'entrée est restée visible. Seul le dernier saut vers la sortie passe par la mémoire.** Le modèle n'a jamais eu à transporter de l'information à travers plusieurs itérations sans pouvoir relire l'entrée.

**Conséquence sur les résultats connus** : 97 % sur l'addition base 16 (Dec 2023) et l'extrapolation `cumsum` (Sept) sont **entièrement compatibles** avec « le modèle relit l'entrée à chaque pas et le latent n'est qu'un espace de travail ». Ils ne démontrent pas que la mémoire construite à la volée porte l'information. Ils ne la réfutent pas non plus — la question est simplement ouverte.

**Réserve à lever (demandée explicitement à `model-design`)** : ce constat vient de la lecture du code et de la trace de la boucle, pas d'une exécution. Le run « Ça marche » de septembre a pu partir du notebook (`notebooks/Th1nker_runner.ipynb`) plutôt que des deux scripts vérifiés. **Vérifier le chemin réellement utilisé par ce run avant d'investir** : si une expérience passée coupait réellement l'entrée, le constat s'affaiblit.

## 3. Propositions d'expérience

### Exp. 0 — Niveaux de référence pour les tâches de calcul **[prérequis, bon marché]**

`scripts/train.py:90` mesure `(targets == preds).float().mean()` : une accuracy **moyennée par token**. Sur l'addition base 16, beaucoup de positions sont prédictibles sans calculer (tokens de format, préfixe partagé, zéros de tête). Le « 97 % » doit être recalibré avant de servir de référence à quoi que ce soit.

À produire : exact-match de séquence, découpage par position, et deux prédicteurs triviaux — (a) recopier les chiffres d'entrée, (b) token le plus fréquent par position. Même esprit que `learn/indexed_attention/eval_metrics.py` (niveaux de hasard obligatoires) ; mutualiser plutôt que dupliquer si la forme s'y prête.

### Exp. 1 — Balayage de `read_step` **[l'expérience décisive]**

`read_step ∈ {0, 1, …, n_step}` à `n_step` fixé.

- `read_step = n_step` : entrée toujours visible — borne haute de contrôle.
- `read_step = 0` : entrée vue au premier calcul seulement, ensuite le modèle ne travaille que sur ses propres latents.

Prédictions discriminantes :

| Si… | Alors quand `read_step → 0` |
|---|---|
| la mémoire à la volée porte l'information | dégradation **douce et progressive** |
| le latent n'est qu'un espace de travail | **falaise** dès que `read_step < n_step - 1` |

- **Entraîner à la valeur testée**, pas seulement évaluer — sinon on mesure un décalage de distribution, pas une capacité. Faire les deux : apparié train/test, **et** train à `read_step` élevé / éval à bas (extrapolation).
- Ordre des tâches : `copy` d'abord (transport pur d'information, aucun calcul — si ça casse là, ça cassera partout), puis `cumsum`, puis addition base 16.
- Coût : une ligne à déparamétrer, les tâches existent déjà (`data/numbers.py`).

### Exp. 2 — Capacité mémoire `n_memory` : « mémoire » ou simple récurrence ?

À `read_step = 0`, balayer `n_memory ∈ {1, 2, 4, 8, ∞}`.

- `n_memory = 1` suffit ⇒ il n'y a pas de mémoire, juste un état récurrent, et la revendication « medium term memory » tombe.
- l'accuracy croît avec `n_memory` ⇒ le modèle accumule réellement une trace exploitable.

C'est cette distinction qui constitue le **vrai contenu scientifique** de la question, et elle n'a jamais été mesurée.

### Exp. 3 — Attribution causale : la mémoire est-elle seulement lue ?

À l'évaluation, corrompre les latents en mémoire à un step donné (zéro, bruit, permutation de l'ordre) et mesurer l'effet sur la sortie. Si la sortie ne bouge pas, la mémoire n'est pas utilisée, **quelle que soit l'accuracy**. C'est le diagnostic d'attribution que le plan Indexed Attention impose déjà pour la KB ; la même logique s'applique ici pour un coût quasi nul.

### Exp. 4 — Implémenter le flow runner du 6 janvier

Le tableau de stratégie par step généralise `read_step`/`n_memory` en un vrai instrument, et permet de tester la stratégie que l'architecture Thinker **suppose** : « lire une fois au début, calculer sur la mémoire, sortir à la fin ». C'est aussi le bon endroit pour le curriculum « large → étroit » de la spec §10.

### Exp. 5 — Signature du raisonnement itératif

Une tâche dont le `n_step` minimal requis **croît avec la longueur d'entrée** (`cumsum` le fait partiellement). Si un `n_step` fixe suffit quelle que soit la longueur, ce n'est pas du raisonnement itératif.

## 4. Pourquoi ça vaut le détour maintenant

`Thinker` (`core/indexed_thinker_model.py`) a **exactement la même structure** : la SM est construite par `APPEND` à chaque itération pendant que la KB reste interrogeable en permanence — soit `read_step = n_step` en permanence. La question « est-ce que la mémoire construite à la volée porte vraiment l'information ? » conditionne donc la conception de la SM dans l'architecture principale, et le toy model est un banc d'essai CPU de quelques minutes là où le Thinker demande du GPU.

Rattachements : spec §3 (options de couplage KB/SM, toujours `[OUVERT]`), spec §14.2 (SM remise à zéro entre fenêtres, registre reporté — un choix qui suppose que la SM porte réellement quelque chose à l'intérieur d'une fenêtre).

## 4bis. Réserve du §2 levée (`model-design`, 2026-09-13)

Vérifié : `notebooks/Th1nker_runner.ipynb`, cellule 44 (celle qui utilise `NumbersCopyDataset` et un balayage `hp_n_latent`/`hp_n_step` — le chemin plausible pour un run copy/cumsum comme celui du 18-19 sept.) contient exactement le même hardcode : `read_step = n_step - 1  # remove on output step`, ligne identique à `scripts/train.py:77` et `scripts/th1nker_runner.py:1044`. La config de cette cellule (`n_latent=[range(4,16+1,2)]`, `n_step=[range(4,12+1)]`) correspond au bloc de config cité dans `experiment.log.md` pour l'entrée "Sept 18 — Ça marche!". **Les trois chemins d'exécution du dépôt (les deux scripts et le notebook) hardcodent tous `read_step = n_step - 1`, sans exception trouvée.** La réserve est donc levée : le constat du §2 ne s'affaiblit pas, il se confirme par triangulation — rien dans ce dépôt n'a jamais entraîné avec un `read_step` autre que `n_step-1`.

## 5. État de la passation — **clôture de la session `thinker-5b`, 2026-09-13**

Ce document a rempli son rôle (établir le constat et proposer le protocole). **Il est désormais historique : la référence vivante est `dev_notes/toy_memory_experiment_plan.md`, tenu par `model-design`, plus à jour et plus complet.** Ne pas le faire diverger ; le fusionner dans l'autre si l'occasion se présente, en n'en gardant que la trace du constat initial (§1-§2).

### Ce qui a été fait, et par qui

- **`thinker-5b` (cette session, close)** : le constat du §2 et les propositions du §3. Aucune implémentation, aucun run.
- **`model-design`** : a vérifié le constat indépendamment (y compris `all_losses_compute` et le flux d'accuracy de `scripts/train.py`, non couverts ici) et **confirmé** — rien à contredire. Puis a implémenté :
  - `learn/toy_memory/eval_metrics.py` — exact-match séquence, accuracy par position, baselines triviales (`copy_input` traité comme **solveur exact** sur `copy` et non comme raccourci ; `most_common_token` mesuré empiriquement), `capacity_budget`, bloc de rapport ;
  - `learn/toy_memory/train_toy_memory.py` — `--read_step` **obligatoire** (le défaut `n_step-1` qui était la cause du problème ne peut plus être réintroduit par inadvertance), `--eval_read_step` pour l'extrapolation, `--read_step_curriculum`, tâches `copy`/`cumsum` ;
  - `dev_notes/toy_memory_experiment_plan.md` — plan complet avec table de décision §4.5.
  - A aussi relâché `N == block_size ** depth` → multiple, côté `HierarchicalMemory` : la Phase 0bis à `depth=1` proposée dans le plan Indexed Attention n'était pas exprimable sans ça.
- **`experiment-manager`** : détient le protocole d'exécution. Aucun résultat au moment de cette clôture.

### Deux corrections apportées à la conception avant lancement (à ne pas re-défaire)

1. **`read_step` ne varie pas une seule chose.** `memory = x` est fixé avant la boucle, donc `x` est visible pendant `read_step + 1` computes, et chaque compute écrit `n_latent` vecteurs. Le budget d'absorption vaut `(read_step + 1) × n_latent`. Sous le contenu de l'entrée, l'échec est une **impossibilité**, pas un verdict sur le mécanisme. D'où le drapeau `capacity_constraining`, dont le critère primaire doit rester un **compte de vecteurs** (`write_budget_vectors < seq_len`) : une comparaison dims-flottantes/bits ne se déclenche quasiment jamais et donne une fausse assurance.
2. **Une falaise à `read_step` bas ne se conclut pas.** Chaque point est entraîné from scratch ; le modèle doit découvrir seul la stratégie « tout écrire au step 0 ». Précédents de ce projet : « having a plateau doesn't mean that the model is at capacity » (22 déc. 2023) et le plateau `n_facts=64` que le curriculum a entièrement débloqué (Phase -1/0). D'où le curriculum sur `read_step` **obligatoire avant tout verdict**, et l'attribution causale (Exp. 3) avant de conclure à une limite réelle si le curriculum ne débloque pas non plus.

### En attente

- **Résultats d'Exp. 1** : contrôle préalable (`copy` à `read_step = n_step` doit atteindre ~100 % exact-match — si ce point échoue, rien d'autre n'est interprétable), puis grille `read_step ∈ {0..6}` × 3 seeds sur `copy` aux deux longueurs (8 et 32), puis `cumsum`, avec curriculum sur toute cellule qui montre une falaise.
- **Exp. 2-5** : au plan, non prioritaires. Exp. 2 (`n_memory`) est celle qui distingue « mémoire » de « simple récurrence » — c'est le vrai contenu scientifique de la question, à ne pas oublier une fois Exp. 1 tranchée.
- **Addition base 16** : troisième tâche, pas encore câblée.

### ⚠️ Travail non commité au moment de cette clôture

Tout ce qu'a produit `thinker-5b` est commité (`49aa28b`, `c1b0016`, `4336e6f`, `351e3b8`, `0d11533`). **Le travail de `model-design` ne l'est pas** : `learn/toy_memory/`, `dev_notes/toy_memory_experiment_plan.md`, `dev_notes/future_experiments.md`, et ses modifications de `core/indexed_memory.py`, `dev_notes/indexed_attention_experiment_plan.md`, `learn/indexed_attention/train_kb_retrieval.py`. À committer par son auteur.

Sont également non commités, antérieurs à tout ceci et appartenant à l'utilisateur : `learn/distill/train_sft.py`, `thesis/paper/main.typ`, `dev_notes/grid5000_usage.log.md`, `thesis/encadreur_profile.md`, `dev_notes/ideas/pretraining_for_dynmaic_inference.md`, `dev_notes/ideas/retrievial_training_design.md`.

### Réserve levée

Le §2 portait une réserve (« constat issu de la lecture du code, pas d'une exécution ; le run de septembre venait peut-être du notebook »). `model-design` a retracé `core/toy_model.py:183-208` indépendamment et confirmé. **La réserve est levée sur la mécanique** ; le chemin exact du run de septembre n'a pas été retracé, mais il ne change plus rien au constat.
