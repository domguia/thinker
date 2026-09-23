# Idée backlog : checkpoint/reprise robuste sur signal (pause/stop externe)

Demande utilisateur 2026-09-22, pas implémentée (fenêtre de tokens serrée
au moment de la demande, priorité au monitoring des runs actifs).

**Besoin** : pouvoir arrêter/mettre en pause un run d'entraînement déjà
lancé depuis l'extérieur (signal, ou mécanisme OAR) sans perdre la
progression, et reprendre exactement où ça s'est arrêté -- pas juste les
poids (`--init_from_checkpoint` existant), mais l'état complet
(optimiseur, position dans le LR schedule, compteur de pas, RNG).

**Pattern standard à suivre** (PyTorch Lightning / HF Trainer) :
1. Handler `SIGTERM`/`SIGUSR2` dans la boucle d'entraînement -> sauvegarde
   un checkpoint complet (`model.state_dict()`, `optimizer.state_dict()`,
   état du scheduler LR, `step`, éventuellement RNG state) puis quitte
   proprement.
2. Flag `--resume_from <path>` au lancement qui recharge cet état complet
   plutôt que de repartir d'un optimiseur/LR schedule neufs.
3. Pertinent pour Grid'5000 : OAR a un mécanisme natif de checkpoint
   (`oarsub -c`/`--checkpoint`, envoie un signal configurable N secondes
   avant la fin du walltime) -- à exploiter directement pour les jobs
   besteffort préemptibles (cas fréquent sur ce projet).

**Cible probable** : `learn/indexed_attention/train_prompt_response.py`
(et généralisable à `train_sft.py`/`train_real_text.py` si utile).

Pas de recherche de faisabilité nécessaire -- pattern connu et standard,
juste un chantier d'implémentation à prioriser plus tard.
