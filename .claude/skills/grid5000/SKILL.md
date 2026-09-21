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

**Check compute capability before picking a GPU cluster**: the project's persistent `~/micromamba/envs/teacher311` (torch 2.14.0+cu130) only ships kernels for compute capability ≥7.5 (Turing and newer) — Pascal (`abacus5`, P100, CC 6.0) and Volta (`abacus30`, V100, CC 7.0) both fail with `cudaErrorNoKernelImageForDevice`, confirmed directly 2026-09-04. Stick to Ampere-or-newer: `abacus21` (A100), `abacus26` (L40S), `abacus27` (H100) all confirmed working. Before reserving an unfamiliar cluster, check `.../clusters/<cluster>/nodes.json` for `gpu_devices.*.model`/microarchitecture, not just free/busy status.

**Grace-Hopper (`hydra`, ARM64) needs a non-default environment for working CUDA**: a plain `pip install torch` on the node's default OS silently gives `torch.cuda.is_available() == False` despite `nvidia-smi` showing the GPU correctly (matching driver/CUDA build) — this is a known unresolved PyTorch packaging bug on aarch64+CUDA wheels ([pytorch/pytorch#123835](https://github.com/pytorch/pytorch/issues/123835)), compounded by Grid'5000's `hydra` nodes not having GPU support in their default deployed OS. Fix: deploy the `ubuntugh2404-arm64-big` environment via `kadeploy3`, or run an official Nvidia NGC PyTorch container for ARM64 via `apptainer run --nv <image>.sif` (lighter than a full kadeploy redeploy). Don't expect a bare `pip install torch` to work here.

**Anti-pattern to avoid**: never conclude a site/cluster is saturated from aggregate free/busy counts alone. A site-wide `busy` reading can be a maintenance placeholder job, not real usage (hit this exact trap at Rennes once) — always sample a few nodes' `.reservations[].types` for `maintenance` before reporting saturation as real.

**Prefer a capability filter over pinning a single cluster name for besteffort jobs**: `oarsub -p "cluster='abacus27'"` can schedule far in the future (observed: 3.5h delay for `abacus27`/H100, despite a static GPU inventory snapshot showing it "free") — a hardware inventory query is a point-in-time snapshot, not real-time availability, and pinning to one cluster forces OAR to wait for that exact cluster even when equally-good alternatives are idle. Filter by capability instead and let OAR pick whatever compatible node is actually free right now:
```bash
oarsub -l gpu=1,walltime=<T> -p "gpu_compute_capability >= '7.5'" -t besteffort -t idempotent ...
```
This consistently got `Running` immediately in practice (2026-09-21), vs. hours of `Waiting` for a pinned cluster. Only pin a specific cluster when the experiment genuinely needs that exact hardware (e.g. one very large-VRAM run) — not as a default habit.

### Estimate duration from measured throughput before dispatching

**Never assume a newly-assigned node has good throughput — measure the first 2-3 minutes and compare to a known-fast baseline before committing a task to it, especially if the task blocks downstream experiments.** A capability filter (CC≥7.5) only guarantees *compatibility*, not *speed* — concrete contrast measured 2026-09-21 on the identical precompute workload: Nancy's `graffiti` cluster (RTX 2080 Ti, 12GB, CC7.5) ran at ~1.85-1.86 ex/s, vs. ~27-42 ex/s on various Rennes GPUs (A100/A40/L40S/RTX A5000) — a ~15x slowdown, costing ~84 real minutes of extra wall-clock on a task that was blocking a downstream matched-reference run.

How to apply: after launching any new task on a node not already benchmarked this session, check `[progress] ... rate=X ex/s` (or equivalent) after ~2-3 minutes; if it's far below the throughput already seen on comparable hardware, and the task is on the critical path for other experiments, kill it and relaunch on a faster free node (`oarsub` with the capability filter above) rather than letting it run to completion in place. Report the estimated time cost (slow-node ETA vs fast-node ETA) so the tradeoff is explicit rather than silently absorbed.

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
