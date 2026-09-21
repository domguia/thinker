---
name: grid5000
description: Manage usage of the Grid'5000 cluster (account jdomguia, group wide) for the Thinker project — SSH access, GPU/CPU reservation via OAR, storage, pitfalls, and usage journaling. Use whenever Grid'5000, OAR, oarsub, kadeploy, or launching training/preprocessing on the cluster comes up.
---

# Grid'5000 — Cluster management for the Thinker project

Account: `jdomguia`, group `wide`. Usage patterns are not yet stabilized (data-prep storage, one-off vs. long compute reservations) — this file is expected to evolve with real experience.

**`wide` group managers** (contact for Group Storage requests, quota extensions, group management):
- Davide Frey (`dfrey`) — davide.frey@inria.fr
- Mvondo Djob Barbe Thystere (`mbarbethystere`) — barbe-thystere.mvondo-djob@inria.fr

## Access

- Bastion (`access.grid5000.fr`) → a site's frontend (e.g. `rennes.grid5000.fr`) → a compute node reserved via OAR. **Never compute on the bastion or a frontend** — only on a reserved node.
- Recommended `~/.ssh/config` (tested and working — see `dev_notes/grid5000_usage.log.md` for history):
  ```
  Host g5k
    User jdomguia
    Hostname access.grid5000.fr
    ForwardAgent yes
    BatchMode yes
    ConnectTimeout 10
    StrictHostKeyChecking accept-new
    HostkeyAlgorithms +ssh-rsa
    PubkeyAcceptedAlgorithms +ssh-rsa
    IdentityFile ~/.ssh/id_rsa

  Host *.g5k
    User jdomguia
    ProxyCommand ssh -W $(basename %h .g5k):%p g5k
    BatchMode yes
    ConnectTimeout 10
    StrictHostKeyChecking accept-new
    HostkeyAlgorithms +ssh-rsa
    PubkeyAcceptedAlgorithms +ssh-rsa
    IdentityFile ~/.ssh/id_rsa
  ```
  → enables `ssh rennes.grid5000.fr.g5k` (frontend) or `ssh <node>.<site>.grid5000.fr.g5k` (any node, including future ones, no new entry needed). **Pitfall**: never set both `ProxyJump` and `ProxyCommand` on `*.g5k` — `ProxyJump` wins and breaks the `.g5k` suffix stripping, causing resolution to fail ("Name or service not known"). Keep only `ProxyCommand`.
- The account exists on every site but data (home NFS) is not replicated across sites — manual transfer (`rsync`/`scp`) needed to move sites.
- Hardware/GPU details per site: https://www.grid5000.fr/w/Hardware (check on demand rather than duplicating here — it evolves, and a memorized cluster→GPU mapping can go stale; see the availability query in `references/doc-map.md` for a live, hardware-joined check).
- To navigate the Grid'5000 wiki/API (find the right page fast): see `references/doc-map.md`.

## Reserving with OAR

**Always check existing jobs first** (`oarstat -u jdomguia`, or the multi-site API equivalent in `references/doc-map.md`) before submitting a new reservation — avoids double-booking or losing track of a job still running from earlier.

**Always name jobs** with `-n "<short-purpose-tag>"` so `oarstat -u jdomguia` / job history is self-descriptive on its own, independent of the journal.

```bash
oarsub -I -n "debug" -l gpu=1,walltime=2:00:00 -p "gpu_model = 'A100'"   # interactive, debug
oarsub -n "run-name" -l gpu=1,walltime=8:00:00 ./run.sh                  # batch, long run
oarstat -u jdomguia                                                       # list your jobs
oardel <job_id>                                                           # release as soon as done
```

Check availability before reserving: `https://<site>.grid5000.fr/drawgantt-svg/` (visual planning) or `oarsub ... --dry-run` (estimated start delay). See `references/doc-map.md` for scripted availability checks (including the GPU-aware one).

**Exotic clusters need `-t exotic`**: Lyon's `hydra`, `gemini`, `sirius`, `neowise`, `pyxis` (and likely similar non-standard hardware elsewhere) are filtered out of a plain `oarsub -l gpu=1 ...` request — even when shown free in `status.json` — unless `-t exotic` is added. Without it, `oarsub` fails outright with a "Filtering out exotic resources" hint rather than reserving.

**The `wide` group has NO priority GPU allocation at Rennes — besteffort only**: confirmed by direct testing (2026-09-03) — a normal `oarsub -l gpu=1 -p "cluster='abacus27'"` (or `abacus21`) fails with "not enough resources" even when the node shows genuinely `free` in `status.json` (not just `busy_free_besteffort`), because all Rennes GPU resources are `production='YES'` and the default queue implicitly filters to `production='NO'`. Explicitly using `-q abaca` (the dedicated Rennes GPU queue mentioned in onboarding docs) gives a clearer error: *"You can only access the required resources in besteffort. Reserve with -q besteffort if it is what you want."* — this is a **project/group access-control fact, not a scheduling race**. Conclusion: any GPU job at Rennes for this project must be submitted with `-t besteffort` (`oarsub -t besteffort -l gpu=1 -p "cluster='...'" ...`), and can be killed at any time by a user with real allocation — never assume a "normal" GPU reservation will succeed at Rennes for this group. (Earlier session notes blaming a "race condition" for a similar failure were an incomplete diagnosis — this access-control restriction is the real and complete explanation.)

**Chemins qui varient d'un site à l'autre — utiliser `~/bin/micromamba`, pas `~/micromamba/micromamba`.** Le binaire `micromamba` n'est PAS au même endroit sur tous les sites (`~/micromamba/micromamba` sur Rennes, `~/bin/micromamba` sur Nantes, historiquement) — une commande de lancement copiée d'un site à l'autre sans vérifier échoue silencieusement (`nohup: failed to run command 'micromamba': No such file or directory`, piège vécu 2026-09-21 sur ecotaxe/Nantes). Un lien symbolique `~/bin/micromamba -> <chemin réel>` a été créé sur Rennes pour uniformiser (Nantes l'avait déjà nativement) — **toujours utiliser `~/bin/micromamba` dans les commandes de lancement désormais**, et si un nouveau site est utilisé pour la première fois, vérifier/créer ce lien avant de copier une commande d'un autre site (`ls ~/bin/micromamba || ln -sf $(find ~ -maxdepth 3 -iname micromamba -type f | head -1) ~/bin/micromamba`). Plus généralement : ne jamais supposer qu'un chemin qui marche sur un site marche sur un autre — `home` n'est pas répliqué (cf. section Storage), donc l'environnement construit sur un site n'existe pas ailleurs tel quel.

**Check compute capability before picking a GPU cluster**: the project's persistent `~/micromamba/envs/teacher311` (torch 2.14.0+cu130) only ships kernels for compute capability ≥7.5 (Turing and newer) — Pascal (`abacus5`, P100, CC 6.0) and Volta (`abacus30`, V100, CC 7.0) both fail with `cudaErrorNoKernelImageForDevice`, confirmed directly 2026-09-04. Stick to Ampere-or-newer: `abacus21` (A100), `abacus26` (L40S), `abacus27` (H100) all confirmed working. Before reserving an unfamiliar cluster, check `.../clusters/<cluster>/nodes.json` for `gpu_devices.*.model`/microarchitecture, not just free/busy status.

**Grace-Hopper (`hydra`, ARM64) needs a non-default environment for working CUDA** — confirmed working end-to-end 2026-09-21: the default deployed OS on `hydra` nodes (Debian 11) has no functional GPU support at all, regardless of how torch is installed. Fix: `kadeploy3 -e ubuntugh2404-big -f $OAR_NODEFILE -k` (note: the real environment name is **`ubuntugh2404-big`**, not `ubuntugh2404-arm64-big` as earlier notes guessed — check with `kaenv3 -l | grep -i hydra` before assuming a name). Confirmed timing: full kadeploy cycle (SetDeploymentMiniOSUntrusted + BroadcastEnvKascade + BootNewEnvClassical) takes **~9 minutes (539s)** on `hydra-2`, not hours — don't over-provision walltime for the deploy step itself. After deploy, `-k` copies your public key to root's `authorized_keys`; `ssh root@<node>` works within seconds of the deployment finishing (add a short retry loop — sshd needs a few seconds after the final reboot). `nvidia-smi` on the freshly deployed node correctly shows the GH200 480GB GPU (driver 570.124.06, CUDA 12.8).

**Plain `pip install torch` on this image can still give you `cuda_available()==False` — but for a different reason than the old aarch64-packaging bug.** The earlier "unresolved PyTorch packaging bug" framing (pytorch/pytorch#123835, filed April 2024) is **outdated**: as of PyTorch 2.11.0 (April 2026), `pip install torch` on aarch64 Linux does pull a real CUDA-enabled wheel by default from PyPI (see the [PyTorch blog on aarch64 packaging](https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/)). Confirmed 2026-09-21 on `hydra-2`: a bare `pip install torch` pulled **torch 2.14.0** with a `cuda-toolkit==13.0.3` dependency — but `torch.cuda.is_available()` was still `False`, because this wheel targets CUDA 13.x while `ubuntugh2404-big`'s driver (570.124.06) only supports up to CUDA 12.8 — a driver/runtime version mismatch, not a missing-wheel problem. **Fix: install explicitly with `pip install torch --index-url https://download.pytorch.org/whl/cu128`** (matches this driver) rather than trusting the bare `pip install torch` default on this specific image. Confirmed working: `torch 2.11.0+cu128`, `cuda available: True`, device `"NVIDIA GH200 480GB"`.

**Gotcha**: `ubuntugh2404-big`'s Python 3.12 does NOT ship `ensurepip`/venv support out of the box — `python3 -m venv ...` fails with "ensurepip is not available... apt install python3.12-venv". Run `apt-get install -y python3.12-venv` first — but note `apt-get update` on this image commonly **fails on an unrelated third-party repo** (`linux.mellanox.com/.../mlnx_ofed`, expired/missing GPG key, "no longer signed") — don't let that abort your script (`apt-get update || true`); the rest of the package lists still refresh fine. If venv still can't be made to work, `pip install --break-system-packages ...` against the system Python is an acceptable fallback. Or use `micromamba` instead to sidestep all of this.

**Confirmed compute throughput (2026-09-21, `torch 2.11.0+cu128`, 8192×8192 matmul, 10 iters)**: fp32 = **103.3 TFLOP/s**, bf16 = **292.4 TFLOP/s**. GPU memory: 100.2 GB free / 102.0 GB total (matches the 96GiB spec). `transformers` (5.17.0) imports and works fine. **`flash_attn` has no prebuilt wheel for aarch64** (`pip install flash-attn` / plain import fails) — this matters for this project specifically since `precompute_teacher_targets.py` depends on FlashAttention (see the 2026-09-20 OLMo-2-7B fp32/bf16 incident in the usage journal); running Thinker's actual precompute/training scripts on `hydra` would need either a from-source flash-attn build for aarch64 (not attempted, likely slow) or switching that code path to PyTorch's native SDPA attention.

**OAR scheduling on `hydra` is not simply "wait for a node to free up"**: this 4-node exotic cluster gets contended by other teams' long-running `deploy`-type reservations and at least one recurring cluster-wide `monitor=prom_.*` job. Observed 2026-09-21: predicted `scheduled_start` for an unpinned `deploy` job can sit hours in the future even while `oarstat`/the API show a node currently `free` — the OAR gantt prediction is not always refreshed in real time relative to another job naturally ending, and pinning to a specific node (`-p "host='hydra-N...'"`) does **not** reliably get you an earlier slot (OAR tends to place the highest-priority already-waiting unconstrained job on whichever node you pin to, pushing yours behind it instead of onto a genuinely-idle node). A `besteffort` job of type `deploy` on this cluster was **not preempted** by a newly submitted normal-queue `deploy` job requesting the same cluster — treat besteffort `deploy` jobs as non-preemptible in practice here (Grid'5000 likely avoids killing mid-deployment jobs to prevent leaving a node in a broken kadeploy state). **Practical takeaway: don't try to out-think the scheduler by pinning nodes or relying on point-in-time free/busy snapshots — just submit an unpinned request for the cluster and let FIFO run its course**, and prefer embedding the entire procedure (kadeploy + install + test) as the OAR job's own command (see below) so you don't need to be present exactly when it starts.

**Reusable launcher**: `tools/grid5000/hydra_gpu_task.sh` in this repo is a filled-in-the-CONFIG-block template encoding everything above (kadeploy, venv/torch/cu128, streaming transfers in and out via the frontend, PASSIVE-job release-on-exit) — copy it, edit the `CONFIG` block (source paths, the command to run, result destination), upload, and `oarsub -t exotic -t deploy -p "cluster='hydra'" -l walltime=<T> -n "<name>" "bash <copy>.sh"`. Don't re-derive this plumbing from scratch for the next hydra task.

**Embed the full setup as the OAR job command, don't rely on catching an interactive window**: a `sleep N` placeholder command + manually SSHing in once "Running" is fragile — a delayed or missed wake-up (scheduled reminder, monitoring script, etc.) burns the entire walltime for nothing (confirmed 2026-09-21: a 3h reservation ran a no-op `sleep 10800` end-to-end because nobody caught the window). Instead, pass a self-contained script as the `oarsub` command itself (e.g. `oarsub ... "bash ~/gh200_auto_setup.sh"`) that does `kadeploy3` → wait for SSH → install → test → write clear `MARKER: ...` lines to a log, all unattended. This also makes the job a `PASSIVE` job that naturally ends (and releases the node) as soon as the script finishes, rather than holding walltime hostage.

**Anti-pattern to avoid**: never conclude a site/cluster is saturated from aggregate free/busy counts alone. A site-wide `busy` reading can be a maintenance placeholder job, not real usage (hit this exact trap at Rennes once) — always sample a few nodes' `.reservations[].types` for `maintenance` before reporting saturation as real.

**Prefer a capability filter over pinning a single cluster name for besteffort jobs**: `oarsub -p "cluster='abacus27'"` can schedule far in the future (observed: 3.5h delay for `abacus27`/H100, despite a static GPU inventory snapshot showing it "free") — a hardware inventory query is a point-in-time snapshot, not real-time availability, and pinning to one cluster forces OAR to wait for that exact cluster even when equally-good alternatives are idle. Filter by capability instead and let OAR pick whatever compatible node is actually free right now:
```bash
oarsub -l gpu=1,walltime=<T> -p "gpu_compute_capability >= '7.5'" -t besteffort -t idempotent ...
```
This consistently got `Running` immediately in practice (2026-09-21), vs. hours of `Waiting` for a pinned cluster. Only pin a specific cluster when the experiment genuinely needs that exact hardware (e.g. one very large-VRAM run) — not as a default habit.

### Estimate duration from measured throughput before dispatching

**Never assume a newly-assigned node has good throughput — measure the first 2-3 minutes and compare to a known-fast baseline before committing a task to it, especially if the task blocks downstream experiments.** A capability filter (CC≥7.5) only guarantees *compatibility*, not *speed* — concrete contrast measured 2026-09-21 on the identical precompute workload: Nancy's `graffiti` cluster (RTX 2080 Ti, 12GB, CC7.5) ran at ~1.85-1.86 ex/s, vs. ~27-42 ex/s on various Rennes GPUs (A100/A40/L40S/RTX A5000) — a ~15x slowdown, costing ~84 real minutes of extra wall-clock on a task that was blocking a downstream matched-reference run.

How to apply: after launching any new task on a node not already benchmarked this session, check `[progress] ... rate=X ex/s` (or equivalent) after ~2-3 minutes; if it's far below the throughput already seen on comparable hardware, and the task is on the critical path for other experiments, kill it and relaunch on a faster free node (`oarsub` with the capability filter above) rather than letting it run to completion in place. Report the estimated time cost (slow-node ETA vs fast-node ETA) so the tradeoff is explicit rather than silently absorbed.

**A faster/bigger GPU is not always a faster run — check `nvidia-smi` utilization before assuming a node change will help.** Observed 2026-09-21 on the `precompute_teacher_targets.py --hidden_layers` (combined top-K + hidden-states) workload with the FP8 Teacher checkpoint: ~13.6 ex/s on an RTX A5000 (24GB, dequantizes FP8→bf16) vs. only ~1.8 ex/s on an H100 NVL (100GB, native FP8 support) for the IDENTICAL command — confirmed genuinely compute-bound on the H100 (`nvidia-smi` showed 99% util, ~390W draw, not idle/starved), not a node-availability or contention issue. Likely cause: the native-FP8 code path combined with `output_hidden_states=True` hits a less-optimized kernel path than the dequantized-bf16 path. Lesson: when a "better" GPU is unexpectedly slower on the SAME command, check `nvidia-smi` util/power first — if it's already maxed out, the workload/precision-path combination itself is the bottleneck, not the node, and switching nodes again won't help.

**Older/bigger-VRAM GPU is not automatically faster for the combined precompute — measured 3x SLOWER on A100 than H100**: tempting workaround for the OOM above is "use an older GPU with more VRAM" (e.g. A100 80GB `ecotaxe`/Nantes) to dodge the FP8-native-only requirement. Measured 2026-09-21 on `precompute_teacher_targets.py --hidden_layers` for the reasoning dataset: A100 80GB (bf16 dequantized) ran at **0.71 ex/s**, vs. H100 NVL (FP8 native) at 2.4-2.56 ex/s for a comparable combined workload — the A100 was **~3x slower**, not faster, despite having 80GB free (no OOM) vs H100's 100GB. Root cause: A100 has roughly 3x less raw FP16/BF16 tensor throughput than H100, and dequantized-bf16 forward passes are inherently heavier than the FP8-native path — the two effects compound. **Lesson: for this project's FP8 Teacher + combined-precompute workload, always prefer a CC≥8.9 (native FP8) GPU over a bigger/older GPU, even when the older GPU avoids OOM** — VRAM headroom does not compensate for the throughput gap on this workload. Only fall back to a big dequantizing GPU if literally no CC≥8.9 capacity exists anywhere and the deadline can't absorb a multi-hour queue wait — and measure the first 2-3 minutes before committing, exactly as this incident did (caught the 13.7h ETA before wasting more than ~10min).

**Native FP8 model + `--hidden_layers` extraction genuinely needs ≥48GB and ideally ≥64GB+ VRAM**: `precompute_teacher_targets.py --hidden_layers 16 --top_k 32` on this project's Qwen3.8-27B-FP8 checkpoint OOM'd on both a 24GB RTX A5000 and a 48GB A40 (confirmed 2026-09-21, dequantizes FP8→bf16 on any GPU with compute capability <8.9, needing ~55.6GB for weights alone per the script's own dtype="auto" docstring warning) — only succeeded on an H100 (100GB, native FP8, no dequantization). For this specific combined-precompute workload, target `abacus27` (H100 NVL) or another CC≥8.9 card directly rather than trying smaller nodes first and discovering the OOM by trial and error.

**Nancy's `graffiti` cluster (RTX 2080 Ti) specifically: avoid for any non-trivial GPU compute, last resort only.** If a precompute or training run ends up there (e.g. because it was the only free node at dispatch time), migrate the output/checkpoint to a fast site (typically Rennes) as soon as it's usable rather than continuing to compute there — see "Direct frontend-to-frontend transfers" below.

### Direct frontend-to-frontend transfers — never relay through the local machine

**Never route a Grid5000 inter-site data transfer (e.g. Nancy → Rennes) through the user's local PC.** Confirmed working pattern: SSH directly from one site's frontend into another site's frontend by bare hostname and run `rsync` there:
```bash
ssh rennes.grid5000.fr.g5k 'rsync -avz nancy.grid5000.fr:~/thinker/path/to/file ~/thinker/path/to/file'
```
This was tested and works (2026-09-21, 1.3GB in ~10s, direct inter-site network, no local hop). Transient "Connection timed out during banner exchange... port 65535" errors on the bastion or a frontend-to-frontend hop happen occasionally and self-resolve on a bare retry — this is ordinary network flakiness, not a structural block; retry once with a verbose (`-vvv`) SSH diagnostic before concluding a real connectivity problem exists, and investigate the actual cause rather than falling back to a local relay as a workaround.

### Staged workflow: CPU data-prep → GPU compute

For distillation work specifically, don't download/prepare datasets on a GPU reservation — it wastes fair-use priority on a scarce resource while the GPU sits idle.

```bash
# 1. Data prep (download, cleaning, tokenization) — CPU only, any site with availability
oarsub -I -n "distill-dataprep" -l host=1,walltime=2:00:00 -p "cluster='<cpu_cluster>'" -q default

# 2. Generation / training — GPU, same site as step 1 to avoid a cross-site transfer
oarsub -I -n "distill-gpu" -l gpu=1,walltime=8:00:00 -p "gpu_model = 'A100'"
```

## Storage

- `/home/jdomguia`: NFS shared per site. Good for code/results/final checkpoints. **No heavy dataloader I/O here.** Quotas, throughput benchmarks, and current known capacity notes: see `references/benchmarks.md`.
- `/tmp` (or `/scratch`) on the reserved node: fast local disk, **wiped at job end**. Copy datasets here at job start, train against it, only copy checkpoints back to home before walltime ends (otherwise `SIGKILL` + data loss).
- **Group Storage**: persistent multi-TB NFS space, shareable within the `wide` group, requestable (see `references/doc-map.md`). **Tied to one site's storage server** — not replicated or natively mounted elsewhere; cross-site access goes over the inter-site network, which can be a real bottleneck for large volumes. **Grid'5000 has no backup service** — copy anything important off the platform (or duplicate across sites).
  - **`killerdroid@storage3.rennes.grid5000.fr`** (confirmed access, `jdomguia` added to group `sto-killerdroid` until 2026-12-31): mounted at `/srv/storage/killerdroid@storage3.rennes.grid5000.fr/`. **This is another project's storage** (Android malware research — `androzoo`, `MalDroid-2020`, etc., owned by other users), not dedicated Thinker space — our files live under a project-specific subfolder, `thinker-distill/`, created for this purpose (holds the downloaded Teacher checkpoint, e.g. `.../thinker-distill/Qwen3.8-27B-FP8`). Already 90% full site-wide (3.5 TB free of 35 TB) — plenty for our needs, but be mindful it's shared with an unrelated team.
  - **Home quota discipline**: the Rennes home quickly exceeded the 25 GB soft limit once large downloads (Teacher checkpoint) and multiple Python environments (venvs, `micromamba`) accumulated — got an actual quota-exceeded email at 59.38 GB. Move large one-off artifacts (model checkpoints) to Group Storage rather than home, and delete superseded environments (e.g. an old venv after switching to a `micromamba` env) instead of leaving them around "just in case".

## Monitoring long-running jobs without wasting tokens

**Don't use a `Monitor` that emits on every progress-log line** — a `[progress]` line every ~20-40s over a multi-hour run generates one chat notification per line, burning tokens on pure noise with nothing actionable in almost every one. Prefer, in order of preference:
1. **Notify only on terminal state** (completion marker like `Wrote...`/`Budget reached`, or an anomaly signature like `Traceback|Error|OutOfMemory`) — the monitor script polls internally (`sleep 60-90`) but only echoes (and thus only notifies) on those lines. This is the default for any run expected to take longer than a few minutes.
2. If intermediate visibility is genuinely useful (e.g. deciding whether to migrate a slow node), throttle to sparse milestones (every 1000 examples / every 10% / every N minutes) rather than every log line — still far fewer notifications than raw tail -f.
3. Wrap the SSH monitor command in an outer retry loop (`while true; do ssh ... ; rc=$?; [ $rc -eq 0 ] && break; sleep 10; done`) — a bare `ssh` inside a `Monitor` can itself die from the transient bastion/network flakiness documented above, which kills the whole monitor (`exit 255`) with no further updates. Don't rely on a single unretried SSH staying up for a multi-hour watch.

When actively deciding whether to migrate a node (comparing throughput against a baseline, early in a run), it's fine to check `tail`/`nvidia-smi` directly a few times in the first 2-3 minutes rather than setting up a Monitor at all — reserve `Monitor` for the "let it run, tell me when it's done or breaks" phase once the node choice is validated.

## Toujours passer par `tools/exp/` — ne jamais gérer les jobs OAR à la main pour une tâche qui peut durer

Le projet a déjà un système d'ordonnancement conçu pour la préemption/reprise (`tools/exp/reserve.py` + `worker.py`, doctrine complète dans `dev_notes/compute_scheduling.md`) : réservation dimensionnée par `reserve.py`, file de cellules (`grid.jsonl`) consommée par des workers idempotents qui reprennent automatiquement après coupure. **Piège vécu (2026-09-21)** : un precompute prioritaire a été relancé à la main sur un job OAR *existant* (`4123387`, initialement soumis pour une autre tâche) sans vérifier son walltime restant — le job a expiré en pleine exécution, coupant le run par surprise (pas une éviction besteffort, juste un job réutilisé dont le temps était compté ailleurs). Règle : pour tout run qui peut dépasser quelques minutes, soit passer par `tools/exp/reserve.py --walltime <dimensionné pour la tâche>` (jamais réutiliser un job existant sans lire `oarstat -f -j <id> | grep -E "walltime|scheduled_start"` d'abord), soit au minimum toujours créer un job dédié dont le walltime est calculé pour couvrir l'ETA mesuré + marge (≥50%).

## Résilience — ne plus perdre de travail ni de temps de veille (règle générale, 2026-09-21)

Rappel du soir qui a motivé cette section : 3 puis 2 precomputes morts silencieusement (~90% de progrès perdu à chaque fois, aucune reprise possible), un job Hydra démarré tôt mais resté sur un `sleep` de test faute de surveillance au bon moment, plusieurs échecs DNS transitoires pris à tort pour des pannes de job. Ces pertes cumulées ce soir se chiffrent en heures de compute — appliquer systématiquement ce qui suit, pour tout script/job du projet, pas seulement pour le precompute Teacher.

**1. Tout script qui tourne plus de quelques minutes doit checkpointer, pas seulement écrire à la fin.** Référence : `learn/distill/precompute_teacher_targets.py` (2026-09-21) — shards périodiques vers un répertoire persistant, écriture en thread daemon (asynchrone, ne bloque pas la boucle de calcul), renommage atomique (fichier temporaire → `os.replace`), reprise automatique depuis le dernier shard complet et validé au relancement. **Piège vécu et corrigé** : `np.savez_compressed` ajoute silencieusement `.npz` au nom si absent — toujours faire finir le nom du fichier temporaire par `.npz` explicitement, sinon `os.replace` échoue (`FileNotFoundError`) après coup. À appliquer aussi aux autres scripts `prepare_*_data.py`/`precompute_*_clusters.py` s'ils deviennent longs (pas fait rétroactivement partout ce soir faute de temps — dette à surveiller).

**2. Entraînements** : `train_prompt_response.py` sauvegarde déjà le meilleur checkpoint (poids seuls, `--save_best_checkpoint_path`) à chaque amélioration en éval — toujours passer ce flag, ne jamais lancer un training sans lui. Pas de reprise d'état d'optimizer par défaut : acceptable pour un run court/moyen (repartir à froid coûte peu). Pour un entraînement très long (plusieurs heures), envisager en plus une sauvegarde inconditionnelle périodique (pas seulement sur amélioration de val) et, seulement si le coût d'un restart complet dépasserait clairement le coût de stockage, sauvegarder aussi l'état de l'optimizer — jugement au cas par cas, pas une règle systématique (l'optimizer est lourd).

**3. Connexions SSH/oarsh instables : ne jamais traiter un échec transitoire comme une panne réelle sans retenter.** Vécu ce soir plusieurs fois : `Could not resolve hostname X` ou `Connection timed out during banner exchange` sur une commande isolée, alors que le frontend répond `OK` immédiatement après un simple retry. Avant de conclure qu'un job/nœud a un problème réel, retenter 1-2 fois (`ssh <site>.grid5000.fr.g5k 'echo OK'` pour confirmer que c'est bien transitoire et pas une vraie panne) — ne diagnostiquer un job comme mort qu'après avoir écarté la cause réseau.

**4. Surveiller activement les premières minutes de toute tâche nouvellement lancée.** Ne jamais considérer un job "lancé avec succès" sur la seule base du message de confirmation du shell (`LAUNCHED`/`disown`) — vérifier dans les 1-2 minutes qui suivent qu'une vraie ligne de progression apparaît dans le log. Piège vécu (Hydra, cf. `dev_notes/grid5000_usage.log.md` entrée 2026-09-03) : une réservation démarrée plus tôt que prévu a tourné à vide (juste un `sleep` de test) faute de surveillance synchronisée au bon moment — créneau perdu. Corollaire : toute action différée/planifiée doit être embarquée directement dans la commande du job (script complet en argument `oarsub`, mode PASSIVE) plutôt que de compter sur une intervention manuelle au bon horaire.

**5. `oarstat` à "Running" ne prouve pas que le calcul est vivant.** Vécu ce soir à répétition (5 precomputes silencieusement morts au total, toujours coïncidant avec une coupure réseau généralisée) : le conteneur OAR peut rester "Running" alors que le process de calcul a été tué à l'intérieur. Vérifier la fraîcheur du log (âge du dernier octet écrit vs horloge courante), pas seulement sa présence ou son contenu — et si un job dont le rythme de progression est connu (constaté au lancement) n'a pas avancé depuis nettement plus que cet intervalle, traiter ça comme une mort probable avant de faire confiance à `oarstat`.

## Pitfalls to remember

- Walltime expires → `SIGTERM` then a quick `SIGKILL`, `/tmp` wiped: save regularly, not just at the very end.
- Launching `python -m ...` or a project script from `~/thinker` via `oarsh`/SSH needs `PYTHONPATH=.` explicitly (no implicit package install) — `ModuleNotFoundError: No module named 'core'` otherwise. Also, a plain `micromamba` on `$PATH` can be missing in a non-interactive `oarsh`/`bash -lc` shell — use the full path (`~/micromamba/micromamba run -p ~/micromamba/envs/<env> ...`) if `which micromamba` fails.
- `setsid nohup ... & disown -a` launched over `oarsh`/`ssh` routinely makes the launching command itself hit its own `timeout`/backgrounding — this is EXPECTED, not a failure: the detached process still starts correctly. Wait a few seconds and check the log file directly rather than treating the timeout as an error.
- Reserving a scarce GPU (A100/H100) without actively using it hurts fair-use priority — release with `oardel` as soon as done.
- Strictly personal access (never share the key/account); forbidden computations: mining, network scans, attacks, undeclared public web services.
- Account tied to the `wide` group with annual revalidation — watch for emails.
- Any publication resulting from Grid'5000 computations must cite: *"Experiments presented in this paper were carried out using the Grid'5000 testbed, supported by a scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations (see https://www.grid5000.fr)."*

## Usage journal

At the end of **every task** involving the cluster (a reservation, a run, a data transfer — not each individual command), add an entry to `dev_notes/grid5000_usage.log.md`: goal, site/cluster/resources used, result, follow-up if any.

A hook (`.claude/hooks/log-grid5000.sh`) additionally auto-logs every Grid'5000 command (`oarsub`/`oardel`/`ssh`/...) to `logs/grid5000_raw.log` — a mechanical safety net, not a substitute for the narrative entry above.
