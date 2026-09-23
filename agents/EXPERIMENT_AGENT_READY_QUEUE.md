# File de tâches prêtes à lancer -- pour infra-agent

(Maintenue par experiment-agent. Piocher dans l'ordre dès qu'une ressource
GPU CC>=7.5 se libère, sans attendre validation. Toujours `rsync -avz
learn/x1/ <site>:~/thinker/learn/x1/` depuis le dépôt local avant de lancer
si le site n'a pas déjà `gen_multiplication`/`gen_labyrinth` à jour
(vérifier `grep -c gen_multiplication ~/thinker/learn/x1/tasks.py`).)

## 1. T2 (multiplication) -- PRIORITÉ HAUTE, grille en cours
- G1/T2 seed0 (graffiti-6, Nancy, job 6938285) et seed1 (abacus18-1, Rennes,
  job 4142742) tournent déjà, 100000 steps chacun, ~10-13min pour finir.
- **Dès qu'un GPU se libère** : lancer seed2 (même commande, `--seed 2
  --save_dir runs/x1_g1_m4_multiplication_seed2 --device cuda > logs/
  x1_g1_m4_multiplication_seed2.log`).
- Commande complète (adapter save_dir/log/seed) :
```
cd ~/thinker && mkdir -p logs && nohup ~/bin/micromamba run -n teacher311 python -m learn.x1.train_dense --task multiplication --train_size_range 1,5 --test_size_range 6,10 --n_embd 192 --n_head 4 --n_positions 64 --batch_size 64 --lr 3e-4 --max_steps 100000 --max_time_minutes 40 --eval_every 2000 --n_eval 200 --seed <N> --save_dir runs/x1_g1_m4_multiplication_seed<N> --device cuda > logs/x1_g1_m4_multiplication_seed<N>.log 2>&1 &
```

## 2. T2 -- G2 (M3 looped-dense), dès G1 validé (EM>=95% sur au moins 1 seed)
- Attention : train_size_range/test_size_range = digit_range pour cette
  tâche (comme T1 addition), PAS une longueur de séquence.
```
cd ~/thinker && mkdir -p logs && nohup ~/bin/micromamba run -n teacher311 python -m learn.x1.train_looped_dense --task multiplication --train_size_range 1,5 --test_size_range 6,10 --n_embd 320 --n_head 5 --n_positions 64 --n_step_train_max 16 --n_step_test 16 --batch_size 64 --lr 3e-4 --max_steps 60000 --eval_every 2000 --n_eval 200 --seed <N> --save_dir runs/x1_g2_m3_multiplication_seed<N> --device cuda > logs/x1_g2_m3_multiplication_seed<N>.log 2>&1 &
```

## 3. T2 -- G3 (M1 Thinker) + X2a (M2 outer_norm), dès G2 validé
```
cd ~/thinker && mkdir -p logs && nohup ~/bin/micromamba run -n teacher311 python -m learn.x1.train_thinker --task multiplication --train_size_range 1,5 --test_size_range 6,10 --disable_kb True --d_model 128 --n_register 8 --n_step_train_max 16 --n_step_test_sweep 1,2,4,8,12,16,24,32 --batch_size 64 --lr 3e-4 --max_steps 30000 --eval_every 2000 --n_eval 200 --seed <N> --save_dir runs/x1_g3_m1_multiplication_seed<N> --device cuda > logs/x1_g3_m1_multiplication_seed<N>.log 2>&1 &
```
(ajouter `--outer_norm` pour la variante X2a/M2, save_dir/log suffixé `x2a_m2_multiplication_seed<N>`)

## 4. T5 (labyrinthes) -- géré par data-agent
- Ne pas dupliquer. Vérifier avec data-agent avant de lancer quoi que ce
  soit sur T5.

## Notes générales
- Toujours vérifier `gpu_compute_capability >= '7.5'` avant tout lancement
  (Pascal/Volta = crash cudaErrorNoKernelImageForDevice avec notre torch).
- Nancy/Rennes ne sont PAS des dépôts git -- rsync `learn/x1/` seulement,
  pas de `git pull` (échoue, "not a git repository").
- Après chaque run terminé (FINAL présent dans le log), reporter le
  résultat à experiment-agent pour consolidation dans results.csv --
  ne pas juste enchaîner sans reporting, sinon les résultats se perdent.
