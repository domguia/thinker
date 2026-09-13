# Toy model — la mémoire moyen terme construite à la volée n'a jamais été testée

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

## 5. État de la passation (2026-09-13)

- **Fait** : ce constat + ces propositions. Rien d'implémenté, aucun run lancé.
- **Délégué** : briefing envoyé à la session `model-design [f957bf]` — implémenter Exp. 0 et Exp. 1 (le socle, les autres en dépendent), déléguer l'exécution à `experiment-manager [9053f3]`, ne pas lancer les runs elle-même, documenter dans `dev_notes/`, rapporter à l'utilisateur ou poser ses questions. Exp. 2-5 transmises pour le plan, hors priorité immédiate.
- **En attente** : la réponse de `model-design`, y compris sur la réserve du §2 (vérifier le chemin du run de septembre).
- **Si personne n'a repris ce chantier** : Exp. 0 puis Exp. 1 sont chacune à portée d'une soirée de CPU, dans cet ordre. Exp. 1 est celle qui répond à la question.
