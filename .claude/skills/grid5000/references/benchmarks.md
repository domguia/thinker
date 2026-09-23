# Known benchmarks & capacity notes

Living cheat-sheet of durable, reusable facts confirmed on Grid'5000 — distinct from `dev_notes/grid5000_usage.log.md`, which is a chronological task journal. Update this file when a fact changes or a new measurement is confirmed; don't let it silently go stale (numbers below are dated).

## Home directory quota

Confirmed via the account's "Homedir quota management" web interface (2026-09-03):
- **25 GB soft limit / 100 GB hard limit, per site**, by default — already active, no request needed up to 100 GB.
- Extension possible up to **500 GB per site** via the "Request quota extension" web form (guidance shown in the form: don't ask for more than **200 GB** for long-term/multi-month needs).
- Beyond 500 GB, or for sharing across users, use **Group Storage** instead (see `doc-map.md`).
- Quota is independent per site (Lyon, Lille, Nancy, Nantes, Rennes, Sophia, Louvain, Grenoble, Toulouse, Luxembourg, Strasbourg each have their own).

## Throughput reference (measured on `paradoxe-7`, Rennes site, 2026-09-03)

Node/path-specific — re-verify if a decision depends on a precise number elsewhere or on a different site/node.

| Path | Operation | Result |
| --- | --- | --- |
| Internet → node (generic CDN, OVH 100 MB test file) | download | ~130 MB/s (~1 Gbps+) |
| Internet → node (Hugging Face, real 244 MB parquet shard from `ultrachat_200k`) | download, anonymous | ~50-60 MB/s across repeated runs — more representative of real dataset downloads than the generic CDN test |
| Internet → node (Hugging Face, same file, authenticated with an HF token) | download | ~52 MB/s — **no measurable improvement over anonymous** in a same-file A/B test (58.7 → 52.4 → 61.1 MB/s across anon/auth/anon runs, all within normal variance). Don't assume a token speeds up single-file throughput; its likely value is avoiding rate-limiting on many/small requests or accessing gated datasets, not raw bandwidth |
| Internet → node (Hugging Face, `hydra-3` at Lyon, real 30.89 GB model snapshot, 81 files, anonymous — no token, different site than the above) | download | **~880 MB/s** — much faster than the Rennes measurements above; likely a better network path at Lyon and/or large-file parallel fetch being more efficient than small-record streaming, not confirmed which |
| Node → `/home/jdomguia` (NFS) | write (1 GB, `dd conv=fdatasync`) | ~992 MB/s (reliable — sync forced) |
| Node → `/home/jdomguia` (NFS) | read (1 GB, `dd`) | ~9.7 GB/s — **not reliable**, likely inflated by page cache (no root to drop caches) |
| Node → `/tmp` local disk (`/dev/sdb5`) | write (1 GB, `dd conv=fdatasync`) | ~448 MB/s |
| Node → `/tmp` local disk (`/dev/sdb5`) | read (1 GB, `dd iflag=direct`) | ~496 MB/s (reliable — `iflag=direct` bypasses cache) |

**Practical takeaway**: neither NFS home nor local disk is the bottleneck for distillation data prep — the Hugging Face download throughput (~60 MB/s) is what actually dimensions prep time. At that rate: ~8 s for 500 MB, ~85 s for 5 GB, ~14 min for 50 GB, ~55 min for 200 GB.

## Site capacity notes

- **Rennes**, 2026-09-03: site-wide home NFS volume (`nfs:/export/home/...`) was at **92% global fill** (18 TB / 20 TB used, across all users of the site) per `df -h`. Doesn't affect an individual's personal quota, but worth knowing before requesting a quota extension there — the underlying volume may be capacity-constrained regardless of the per-user cap.

## GPU × tâche — carte de performance et d'accès

Recense les GPU du projet par cluster, avec les mesures réelles connues (peu nombreuses — compléter au fil des runs plutôt que deviner) et l'estimation théorique sinon signalée comme telle. Distinguer explicitement **precompute Teacher** (inférence FP8/dequant, gros modèle) vs **training student** (repr-KD, petit modèle, bf16) — un GPU rapide sur l'un n'est pas forcément rapide sur l'autre (cf. H100 natif-FP8 contre-intuitivement plus lent qu'A5000 dequant sur le precompute, section usage log 2026-09-21).

| Cluster (site) | GPU | CC | VRAM | Accès groupe `wide` (vérifié) | Mesure réelle connue | Estimation training student (théorique, non mesuré sauf L40S) |
| --- | --- | --- | --- | --- | --- | --- |
| `abacus27` (Rennes) | H100 NVL | 9.0 | 93GB ×4/noeud | besteffort seulement | precompute FP8 natif: 1.8-2.56 ex/s (lent, cf. piège FP8+hidden_states) | ~2.5-3x plus rapide que L40S (spec bf16 tensor ~990 TFLOPS) |
| `abacus26` (Rennes) | L40S | 8.9 | 44GB | besteffort seulement | **training student mesuré : 0.552 s/step @ batch=6/block=512** (2026-09-03, KD) | baseline de référence |
| `abacus21` (Rennes) | A100-PCIE-40GB | 8.0 | 40GB | besteffort seulement | precompute bf16 dequant: 0.71 ex/s (mesuré ~3x plus lent que H100 sur precompute) | ~comparable ou légèrement en dessous de L40S |
| `ecotaxe` (Nantes) | A100 80GB PCIe | 8.0 | 80GB ×3/noeud | **priorité normale confirmée** (`-t exotic` requis, pas besteffort) (2026-09-21) | precompute: voir A100 ci-dessus | ~comparable à L40S, VRAM généreuse (bon pour gros batch) |
| `abacus4` (Rennes) | A40 | 8.6 | 44GB ×2/noeud | besteffort, mais noeud souvent 100% libre en pratique | aucune | ~2x plus lent que L40S (spec bf16 tensor ~150 TFLOPS) |
| `abacus3` (Rennes) | RTX A5000 | 8.6 | 23GB | besteffort seulement | precompute FP8 dequant: 13.6 ex/s (rapide sur precompute spécifiquement) | ~2.2x plus lent que L40S |
| `abacus17`/`abacus18` (Rennes) | Quadro RTX 6000 | 7.5 | 22GB | besteffort seulement | aucune | ~2.5x+ plus lent, **pas de tensor core bf16 natif (Turing)** — risque de fallback/erreur si le script force bf16 |
| `graffiti` (Nancy) | RTX 2080 Ti | 7.5 | 11GB ×4/noeud | besteffort seulement (testé 2026-09-21, `oarsub` sans `-t besteffort` refusé) | precompute FP8 (workload lourd): ~1.85-1.86 ex/s, ~15x plus lent que Rennes sur ce workload précis ; training reasoning (léger): ~1.05-1.4 ex/s | même caveat Turing bf16 que RTX6000 ; VRAM limitante pour un gros batch |
| `chuc` (Lille) | (à vérifier — GPU non caractérisé ici) | ? | ? | **priorité normale confirmée** (`oarsub` sans besteffort accepté, 2026-09-21) | aucune | à mesurer avant de compter dessus |

**Limites à garder en tête** :
- Une seule mesure de training réelle existe (L40S, KD sur student) — tout le reste du côté "training" est extrapolé depuis les specs publiques (TFLOPS bf16 tensor-core théoriques), pas mesuré sur ce projet.
- Pour repr-KD spécifiquement, le vrai goulot peut être la lecture du memmap hidden-states (60-90GB) plutôt que le calcul GPU — un GPU plus rapide n'aide pas si le job est I/O-bound sur NFS/Group Storage réseau. À vérifier via `nvidia-smi` (util%) dès les premières minutes, cf. section "Estimate duration from measured throughput" du SKILL.md.
- L'accès normal vs besteffort est une propriété du **groupe `wide` sur ce cluster**, pas du noeud individuel — testé via une soumission réelle courte (`walltime=0:01:00`) plutôt que deviné, car pas d'option `--dry-run` disponible sur cette version d'`oarsub`.

## Wiki vs. platform credentials

The platform account password (used for SSH, OAR, and API `--netrc` auth on `api.grid5000.fr`) is **not necessarily** the same as the wiki (`www.grid5000.fr`, MediaWiki) login — they can be separate accounts. In practice, the wiki doesn't need to be logged into at all for read-only consultation.
