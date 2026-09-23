# WRITING_PLAN — Papier ICLR 2027 (deadline 26/09/2026)

> Document de **direction**, pas de rédaction. La rédaction est faite par un système IA séparé (LaTeX, gabarit officiel ICLR 2027).
> Ce fichier dit : quoi affirmer, avec quelle preuve, quoi anticiper, quelles expériences lancer.
> Statut : v1 (2026-09-23), cadrage §1 PROVISOIRE — en attente de validation utilisateur.
> Revue : après chaque étape, un Opus vérifie la conformité texte ↔ ce plan (§8).

---

## 0. Checklist générique ICLR

- [ ] Gabarit officiel ICLR 2027 (LaTeX, `iclr2027_conference.sty`) — vérifier sur le site/OpenReview la limite de pages (historiquement 9 p. main text hors refs/annexes ; 10 p. en camera-ready).
- [ ] Soumission **anonyme** : pas de nom, pas d'affiliation, pas de lien GitHub non anonymisé, pas de « Grid'5000 jdomguia », pas de « nos travaux précédents [X] » auto-identifiants. Acknowledgements (Grid'5000, Rennes) → retirés pour la soumission, remis au camera-ready.
- [ ] Code : lien anonyme (anonymous.4open.science) ou zip en supplementary.
- [ ] **Reproducibility statement** (section recommandée ICLR) : hyperparamètres, seeds, budget GPU, pointer vers annexes.
- [ ] **Ethics statement** (court).
- [ ] **Déclaration d'usage de LLM** (politique ICLR depuis 2026 : obligatoire si LLM utilisé pour la rédaction/recherche) — à vérifier dans le CFP 2027.
- [ ] Profil OpenReview de tous les auteurs créé à l'avance (délai de modération possible !) — **à faire aujourd'hui**.
- [ ] Deadline abstract (souvent ~1 semaine avant le full paper) : vérifier qu'elle n'est pas déjà passée.
- [ ] Obligation de reviewing réciproque ICLR pour les auteurs — vérifier.
- [ ] Bibliographie : fusionner `references.typ` (48) et `references.bib` (32) → un seul `.bib` propre, vérifier chaque entrée (titre/année/venue ; pas de refs hallucinées).
- [ ] Figures vectorielles (PDF), lisibles en N&B, légendes autoportantes.
- [ ] Tableaux avec moyenne ± écart-type (seeds) ou mention explicite « single run ».
- [ ] Nommage cohérent partout (Thinker, n_step, K_mem…) — glossaire notation en annexe.

---

## 1. Cadrage (PROVISOIRE — à valider)

**Thinker = vision** (architecture latente récurrente + mémoire indexée externe). **Ce papier = un sous-ensemble démontré** :

> *Titre de travail :* « Depth on Demand: Test-Time Adjustable Recurrent Computation — Benefits, Efficiency, and a Calibration Failure Mode »

Contributions (ordre d'importance) :
1. **C1 — Profondeur ajustable à l'inférence** : l'entraînement à n_step aléatoire (U(1,8)) rend le modèle robuste à n_step_test hors distribution (16 : +0.029 CE vs +4.02 CE en n_step fixe). Preuve : `agents/OBJECTIVES_LOG.md` (22/09), phase13/14 scripts `tmp_scripts_local/phase13_extrapolate.sh`, `phase14_nsteprand.sh`.
2. **C2 — Efficience d'inférence** : params et débit vs LLM dense (5.8× moins de params, 8.5× tok/s vs Qwen3.5-0.8B, L40S). Preuve : commits `2b0e269`, `46286ec`. Angle « infrastructure » (intérêt du partenaire de Rennes). **Doit être réécrit en comparaison équitable (voir E2).**
3. **C3 — Analyse d'un mode d'échec** : effondrement de calibration (~0.5 % acc argmax teacher-forced vs 21.2 % Baseline C dense même échelle), 6 hypothèses isolées écartées (n_step, KB off, answer head, n_register=1, combiné, largeur 2×). Preuve : commits `63949b3`…`2005a0d`. Présenté comme contribution scientifique (« ce que la récurrence partagée coûte »), pas caché.

**Hors-périmètre / limitations assumées** : retrieval EM=0 % (`ecb7cfb`, `2ca910d`) → section Limitations + futur ; KD vs CE → annexe (non concluant) ; mémoire indexée → décrite dans l'archi, résultats mémoire synthétiques (bug pooling K/V corrigé, `dev_notes/experiments/b1_associative_recall.md`, `toy_memory.md`) en annexe **seulement si** un résultat positif propre existe.

**Alternative si l'utilisateur refuse ce cadrage** : papier « mémoire indexée + boucle » → exige de résoudre EM=0 % avant le 25 → risque très élevé, déconseillé.

---

## 2. Structure cible (≈9 pages)

| § | Contenu | Pages | Preuves / figures |
|---|---|---|---|
| Abstract | problème (profondeur fixe = compute fixe), méthode, 3 résultats chiffrés, honnêteté sur le coût | — | — |
| 1 Intro | motivation : adapter le compute à l'inférence sans retrain ; déploiement/infra ; contributions C1–C3 en puces | 1 | Fig. 1 : courbe CE vs n_step_test (fixe vs aléatoire) — le « money plot » |
| 2 Related work | Universal Transformer, PonderNet/ACT, recurrent-depth (Geiping et al. 2025 « Scaling test-time compute with latent reasoning »), looped transformers (Giannou, Yang, Saunshi 2025), Perceiver/Perceiver-AR, early-exit/LayerSkip, memory-augmented (NTM, Memorizing Transformers), KD. **Positionnement explicite vs Geiping** (le plus proche) | 1 | — |
| 3 Architecture | Thinker : latent + cross-attn input + bloc partagé itéré + (mémoire indexée, brièvement) | 1.25 | Fig. 2 : schéma |
| 4 Training | n_step aléatoire, KD (Teacher LFM2-1.2B, top-K), données, budget | 0.75 | Tab. hyperparams → annexe |
| 5 Exp. C1 | extrapolation n_step, seeds, plusieurs datasets, baseline UT/dense | 1.5 | Fig. 1, Tab. 1 |
| 6 Exp. C2 | efficience : params, FLOPs/tok, latence, tok/s, **à qualité égale ou en Pareto** | 1 | Fig. 3 : Pareto qualité vs coût |
| 7 Analyse C3 | effondrement de calibration : constat, 6 ablations, ce qui est exclu, hypothèse restante | 1.5 | Tab. 2 ablations, Fig. 4 (calibration / distribution des logits) |
| 8 Limitations | retrieval EM=0, échelle ≤ ~260M, génération libre dégénérée (commune à la baseline), single family (LFM2) | 0.5 | — |
| 9 Conclusion | | 0.25 | — |
| Annexes | KD vs CE, hyperparams, mémoire indexée/toy, détails bench, compute utilisé | ∞ | |

---

## 3. Anticipation des reviewers (à compléter par la revue adverse §7)

| Attaque probable | Réponse prévue | Expérience |
|---|---|---|
| Single run, pas de variance | 3 seeds sur C1 et Baseline C | E1 |
| Bench d'efficience inéquitable (Qwen 6× plus gros, pas de KV-cache Thinker, qualité non égale) | Baseline C même taille dans le bench ; FLOPs/tok ; courbe Pareto | E2 |
| « Pas nouveau vs UT / Geiping 2025 » | Différence : cross-attn latente (coût indépendant de la longueur de contexte ?), randomisation + analyse de l'échec ; comparer directement à un UT/looped dense de même taille | E3 |
| « Le modèle ne marche pas (0.5 % acc) » | C3 est justement l'analyse ; montrer que C1 tient **sur la métrique qui compte** ; + tentative de remède | E4, E5 |
| Tâches/échelle jouet | Assumer « étude de mécanisme » ; au moins un point de mise à l'échelle | E6 |
| La CE plate à n_step=16 = le modèle ignore les itérations ? | Montrer que plus d'itérations **aide** sur une tâche où la profondeur compte (algorithmique) | E7 |
| Pourquoi KD ? confond | annexe KD vs CE | existant |
| Data leakage / contamination benchmark | décrire splits | — |

---

## 4. Expériences à lancer (liste pour experiment-manager)

Priorité **P0** = le papier est attaquable sans ; **P1** = renforce fortement ; **P2** = bonus si capacité libre. Chaque ligne : hypothèse → décision.

### P0
- **E1 — Seeds C1** : 3 seeds × {n_step fixe, n_step U(1,8)}, éval n_step_test ∈ {1,2,4,8,12,16,24,32}. *H* : l'écart 0.03 vs 4.0 est robuste. *Si* variance > effet → C1 rétrogradé. Aussi 3 seeds Baseline C.
- **E2 — Bench d'efficience équitable** : Thinker vs Baseline C (même params) vs Qwen3.5-0.8B vs LFM2-350M ; mesurer params, FLOPs/tok (analytique + profiler), latence prefill/decode, tok/s, mémoire pic, **en fonction de la longueur de contexte** (512→8k) et de n_step ; activer le KV-cache côté dense, documenter l'absence côté Thinker. Reporter sur les mêmes axes la qualité (CE/ppl). *H* : l'avantage tient à params égaux grâce à la cross-attn latente sur long contexte. *Si* l'avantage disparaît à params égaux → C2 reformulé en « scaling avec la longueur de contexte » seulement.
- **E3 — Baseline récurrente dense** : Universal-Transformer/looped dense (self-attn, poids partagés, même params, même n_step aléatoire). *H* : C1 n'est pas spécifique à Thinker (probable) → alors C1 = « la randomisation suffit » et la contribution de Thinker = C2. *Si* le looped dense collapse aussi → C3 devient « propriété de la récurrence partagée » (résultat plus fort et général !). **Expérience la plus informative du lot.**
- **E7 — Profondeur utile** : tâche où la profondeur doit aider (addition multi-chiffres, parité, multi-hop synthétique, composition causale `dev_notes/experiments/synthetic_composition_causal_control.md`) : accuracy vs n_step_test. *H* : acc croît avec n_step jusqu'à saturation, et au-delà de n_train. *Si* plat → C1 = « robustesse », pas « compute ajustable » : reformuler le titre.

### P1
- **E4 — Remède à la calibration** : (a) temperature scaling / logit calibration post-hoc ; (b) perte auxiliaire à chaque itération (deep supervision) ; (c) supprimer le partage de poids sur la dernière itération (tête non partagée) ; (d) LR/normalisation (pre-norm vs post-norm, norme du latent par itération). *H* : dérive de norme du latent à travers les itérations. *Si* un remède marche → C3 passe de « constat » à « diagnostic + fix » (gros gain de score).
- **E5 — Diagnostic mécanistique du collapse** : norme/rang effectif du latent par itération, entropie des logits, logit lens par itération, similarité cosinus entre itérations (point fixe ?). *H* : effondrement de rang / convergence vers point fixe. Fournit Fig. 4.
- **E6 — Mise à l'échelle** : 3 tailles (≈30M, 130M, 400M) pour C1 et le collapse (déjà 2 points d_model 256/512). *H* : tendances stables avec l'échelle. Utiliser abacus27 (H100) pour 400M.
- **E8 — Distribution d'entraînement de n_step** : U(1,4), U(1,8), U(1,16), géométrique/Poisson (façon Geiping). Ablation standard attendue.

### P2
- **E9 — Early exit / profondeur adaptative** : critère d'arrêt (convergence du latent) → compute moyen vs qualité. Transforme C1 en gain d'efficience concret (lien C1↔C2, argument infra fort).
- **E10 — Seconde famille** (OLMo ou Qwen avec Teacher de même famille, cf. skill `model-families`) sur C1 : généralisation.
- **E11 — Retrieval** : Baseline C évaluée en EM/F1 sur le même retrieval. Si Baseline C aussi ~0 → le 0 % est un problème de données/échelle, pas de Thinker (le dire en limitation).
- **E12 — Throughput serving** : batching, débit multi-requêtes, mémoire par requête (pas de KV-cache qui croît ?) — argument infra pour Rennes.

Règles : KD par défaut (CLAUDE.md), hidden_layers inclut `last`, Teacher de même famille.

---

## 5. Figures/tableaux à produire (par les agents d'analyse)
1. Fig. 1 CE/acc vs n_step_test, fixe vs aléatoire, ±std (E1, E7).
2. Fig. 2 schéma archi.
3. Fig. 3 Pareto qualité–coût et coût vs longueur de contexte (E2).
4. Fig. 4 diagnostic collapse par itération (E5).
5. Tab. 1 extrapolation multi-datasets ; Tab. 2 ablations collapse (existant + E4) ; Tab. 3 efficience.
Données brutes + scripts de figures versionnés (reproductibilité).

## 6. Recherche bibliographique (déléguée, sous-agents peu coûteux)
- Vérifier/compléter : Geiping 2025 recurrent depth, Saunshi 2025 looped, Dehghani UT, PonderNet, ACT, Mixture-of-Depths, LayerSkip, Perceiver IO/AR, travaux 2025–26 sur la calibration des modèles récurrents/looped.
- Pour chaque : 1 phrase « ce qu'ils font / en quoi on diffère ».

## 7. Revue adverse
Un sous-agent joue un reviewer ICLR hostile sur ce plan → ses critiques sont intégrées en §3/§4 (voir §9).

## 8. Protocole de vérification (Opus, plus tard)
Pour chaque section rédigée : (a) chaque affirmation chiffrée pointe vers une preuve §1/§4 ; (b) aucune affirmation au-delà des preuves ; (c) les attaques §3 ont une réponse dans le texte ; (d) checklist §0 cochée.

## 9. Journal de revue adverse

### Revue 1 (2026-09-23, Sonnet + WebSearch) — score estimé 3/10 actuel, 5-6/10 si P0+P1 + recadrage
**⚠ Références à vérifier une par une (existence, auteurs, contenu) AVANT toute citation — risque d'hallucination :**
- « Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers », arXiv 2604.07822 (2026) — récurrence dynamique vs fixe → extrapolation. **Recouvre C1.**
- « Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping », arXiv 2606.29983 (2026) — profondeur d'entraînement randomisée. **Recouvre C1/E8.**
- « Stability and Generalization in Looped Transformers », Labovich, arXiv 2604.15259 (2026) — outer normalization nécessaire à la stabilité. **Cause candidate de C3.**
- « Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence? », Wang & Reid, arXiv 2609.01924 (2026) — **[vérifié 2026-09-23, résumé corrigé]** N'est PAS un effondrement représentationnel généralisé : étudie si un "global workspace" (représentations causalement actives) survit quand la profondeur est implémentée par récurrence plutôt que par couches distinctes (Ouro-2.6B, Huginn-0125 vs Qwen3.6-27B). Conclusion nuancée : un workspace se forme dans la partie itérée, mais la récurrence change son accessibilité — pas un collapse pur. **Recouvre C3/E5 partiellement, à citer avec cette reformulation précise, pas comme "collapse".**

**Conséquences (si les refs sont confirmées) :**
1. **C1 n'est plus une contribution nouvelle** → à présenter comme réplication contrôlée dans une architecture différente (cross-attn latente), à citer comme prior art ; E8 doit reprendre leurs schémas de distribution.
2. **Cadrage alternatif recommandé (à trancher par l'utilisateur)** — C3 devient la contribution principale :
   « When Shared-Weight Recurrence Fails Silently: Calibration Collapse in Depth-Adjustable Transformers, and What Fixes It ». Ordre : C3 (diagnostic + remède) > C2 (infra) > C1 (robustesse, non revendiquée comme nouvelle).
   Ce cadrage n'est fort **que si** E3 (le looped dense collapse aussi → propriété générale) et/ou E4/E13 (un remède marche) donnent un résultat positif. Sinon, repli : papier d'efficience C2 + analyse honnête.
3. **Nouvelle expérience E13 (P0, coût faible, rendement fort)** : ablation « outer normalization » (norme appliquée au latent entre les itérations / en sortie de boucle) sur Thinker ET sur le looped dense E3. *H* : son absence cause le collapse. *Si* acc remonte vers ~21 % (Baseline C) → C3 = diagnostic + fix → titre alternatif adopté. *Si* aucun effet → une hypothèse de plus écartée (ajouter au Tab. 2).
4. Ajouts §3 : attaque « C1 déjà publié » (réponse : réplication + citation, pivot vers C3) ; attaque « pas de comparaison à Huginn / stochastic stopping » (réponse : Related Work + E8).
5. Risque calendrier : si E2 n'est pas fini le 25, C2 est atténué dans l'abstract.

**Ordre de lancement révisé (P0) :** E13 → E3 → E7 → E1 → E2 (E13/E3 conditionnent le cadrage final).
