# Grid'5000 usage journal

Cluster/infra facts only (jobs, nodes, reservations, storage, access, environment setup). Experiment results (data, model, training) belong in `dev_notes/experiment.log.md`, not here. One entry per task/session (not per individual command). Format:

```
## YYYY-MM-DD — <short task objective>
- Site/cluster: ...
- Resources: ... (GPU/CPU, walltime, job id(s) if relevant)
- Result: ...
- Notes / follow-up: ...
```

## 2026-09-03 — Cluster access verification (bastion + Rennes frontend)
- Site/cluster: bastion (`access.grid5000.fr`) then frontend `rennes.grid5000.fr` (via `-J`)
- Resources: no reservation, plain SSH connectivity test with the existing `id_rsa` key
- Result: connection succeeded at both levels (`access-south`, `frennes`); `oarstat -u jdomguia` showed no active job
- Notes / follow-up: access fully functional, ready for a first test OAR reservation whenever a compute need comes up

## 2026-09-03 — CPU reservation at Rennes (job 4091061) + throughput benchmarks
- Site/cluster: Rennes, cluster `paradoxe`, job **4091061**, node `paradoxe-7.rennes.grid5000.fr`, 2h walltime, started 16:22:49
- Context: Rennes came back available right after the morning's NFS (home + group storage) maintenance ended
- Network/storage benchmarks measured from `paradoxe-7` (see `references/benchmarks.md` for the consolidated reference table): ~130 MB/s to a generic CDN, ~60 MB/s to Hugging Face; NFS home write ~992 MB/s; local `/tmp` (`/dev/sdb5`) write ~448 MB/s / read ~496 MB/s
- Point of attention: `df` showed the site's shared NFS volume at 92% global fill (18 TB/20 TB, all users) — no impact on personal quota at the time, but a sign the underlying volume can be capacity-constrained regardless of per-user quota
- Notes / follow-up: job later hit its 2h walltime and died (`state=Error`) before being reused — resubmitted as job 4091111 (see below)

## 2026-09-03 — HF token throughput A/B test (job 4091061)
- Site/cluster: Rennes, `paradoxe-7`, same job as above
- Result: same-file A/B/A test (anonymous → authenticated → anonymous) gave 58.7 / 52.4 / 61.1 MB/s — no measurable benefit from providing an HF token, within normal run-to-run variance
- Notes / follow-up: token file only exists in the Rennes home (per-site homes, not shared) — confirmed again later when Lyon/other-site jobs couldn't see it

## 2026-09-03 — CPU job 4091111 + first GPU access-control findings
- Site/cluster: Rennes (`paradoxe-55`, job **4091111**, CPU, 3h walltime); Lyon (`hydra-3`, job **2064691**, GPU H200, 2h walltime, requires `-t exotic` — Lyon's `hydra`/`gemini`/`sirius`/`neowise`/`pyxis` are typed "exotic" and `oarsub` refuses them without this flag even when free)
- Rennes GPU (`abacus27`, H100) reservation attempt **rejected at submission** ("not enough resources") despite showing `busy_free_besteffort` — investigated via a research subagent (Grid'5000/OAR docs): best-supported explanation at the time was a race condition (a normal job, `nbires`, claimed the resource between check and submission). This was later superseded by a more complete explanation (see 2026-09-04 entry): the `wide` group simply has no priority GPU allocation at Rennes at all.
- Discovered `killerdroid@storage3.rennes.grid5000.fr` (a Group Storage, added to the account later that day): 35 TB, 90% full (32 TB used / 3.5 TB free) — access confirmed via `df` from the Rennes frontend
- Teacher checkpoint (30.89 GB) downloaded on `hydra-3`: ~880 MB/s, 35s — see `references/benchmarks.md`
- Blocker: `hydra` (Grace-Hopper/ARM64) never got CUDA working via a plain `pip install torch` despite matching driver/CUDA/arch support — root cause (a known PyTorch packaging bug on aarch64+CUDA wheels, and Grid'5000's `hydra` needing a non-default `kadeploy3` environment for GPU support) confirmed by a research subagent; not pursued further given `abacus27`/`abacus21`/`abacus26` (x86_64) worked once the access-control issue below was understood
- Notes / follow-up: job 4091111 finished its data-prep task and was released; Lyon `hydra-3` job abandoned once the Grace-Hopper CUDA issue was diagnosed as not worth fixing (would need a full node redeploy)

## 2026-09-04 — Root cause of the Rennes GPU rejections + home quota incident
- Site/cluster: Rennes (`abacus27`, `abacus21`, `abacus26` — all besteffort; `paradoxe-3`, CPU normal queue)
- **Definitive root cause** of every earlier Rennes GPU rejection: the `wide` group has **no priority GPU allocation at Rennes at all** — confirmed via `-q abaca` giving an explicit "You can only access the required resources in besteffort" error. Every Rennes GPU job for this project must use `-t besteffort`, and can be preempted at any time by a user with real allocation (observed repeatedly: jobs 4091163, 4091180, 4091205, 4091219 were all killed mid-task by preemption, sometimes within ~25 minutes). Documented in `SKILL.md`.
- Environment setup discovered/needed for GPU work here, persisted in the home (`~/micromamba`) across job deaths: Python 3.9 (node default) is too old for `transformers`' git branch (needs the brand-new `qwen3_5` architecture support, requiring Python ≥3.10) — installed Python 3.11 via `micromamba` (no root needed). The node's default image also has no CUDA toolkit (`nvcc`) — installed via `micromamba install -c nvidia -c conda-forge cuda-nvcc gcc_linux-64 gxx_linux-64`, needed to compile any CUDA extension from source.
- **Home quota incident**: automated warning — 59.38 GB used on Rennes home vs. 25 GB soft / 100 GB hard limit (7-day grace period). Root cause: a large model checkpoint downloaded/copied multiple times across preemption cycles, plus an obsolete Python venv left behind after switching to the `micromamba` env. Same day, received confirmation of membership in group `sto-killerdroid` (until 2026-12-31), requested earlier specifically to solve this kind of storage problem. Moved the large checkpoint to `/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/` (a project-specific subfolder — `killerdroid` itself belongs to an unrelated Android-malware research group, sharing it courteously) and deleted the obsolete venv, bringing home usage back to 23 GB. A home quota extension to 200 GB is separately in progress via the UMS web form (submitted by the user, response pending).
- Notes / follow-up: any future GPU job on Rennes should point large-artifact paths at the `killerdroid` subfolder, not home or `/tmp`; expect besteffort preemption at any time and design jobs to save incremental progress accordingly

## 2026-09-04 — `reasoning` data prep re-run (job 4091210)
- Site/cluster: Rennes, `paradoxe-3`, job **4091210**, CPU, normal queue, 2h walltime
- Notes / follow-up: job released (`oardel`) immediately after its output was copied to persistent storage, since its purpose was fulfilled — avoided holding a CPU reservation idle. (See `experiment.log.md` for what the re-run was for.)

## 2026-09-04 — GPU jobs 4091219 (A100) and 4091255 (L40S), both besteffort
- Site/cluster: Rennes, `abacus21` (A100, job **4091219**) then `abacus26` (L40S, job **4091255**) — both besteffort, both reused the persistent `~/micromamba/envs/teacher311` environment set up earlier
- `abacus26` (L40S, Ada, compute capability 8.9) was deliberately chosen over Ampere for a Teacher-loading task since it's the lowest tier with native FP8 tensor core support — see `SKILL.md`/`qwen3.8-27b-notes.md` for why that matters for this specific checkpoint
- Both jobs eventually died via besteffort preemption (job 4091219 after supporting several hours of work across environment fixes and two training runs; 4091255 survived long enough for its task). No mid-task data loss on either, since intermediate artifacts were written to the home or `killerdroid`, not `/tmp`.
- Notes / follow-up: (see `experiment.log.md` for what actually ran on these jobs — kernel compilation, benchmark results, SFT/KD training)

## 2026-09-04 — P100/V100 nodes are unusable with the current torch build
- Site/cluster: Rennes, `abacus5` (P100 16GB) tried and released — attempted a cheap LR-sweep job there since it was the only immediately-free besteffort GPU (job **4091639**, `abacus26`/`abacus21` fully booked or scheduled ~2 days out at the time)
- **Finding**: `torch==2.14.0+cu130` (installed in the persistent `~/micromamba/envs/teacher311` env) only ships kernels for compute capability ≥7.5 (Turing and newer: sm_75/80/86/90/100/120) — Pascal (P100, CC 6.0) and **Volta (V100, CC 7.0) are both unsupported**, raising `cudaErrorNoKernelImageForDevice`. Confirmed directly (`torch.cuda.get_arch_list()`), not assumed. This rules out `abacus5` (P100) and `abacus30` (V100×8) for any future job using this env — restrict Rennes GPU picks to Ampere-or-newer (`abacus21`=A100, `abacus26`=L40S, `abacus27`=H100 confirmed working; A40 untested but should work at CC 8.6).
- Notes / follow-up: worth adding an explicit compute-capability check to `SKILL.md`'s GPU-picking guidance so this isn't rediscovered by trial-and-error again; re-checked `abacus27` (H100, busy_besteffort) instead and it worked (job 4091652).
