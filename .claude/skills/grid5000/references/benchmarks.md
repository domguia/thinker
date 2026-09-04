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

## Wiki vs. platform credentials

The platform account password (used for SSH, OAR, and API `--netrc` auth on `api.grid5000.fr`) is **not necessarily** the same as the wiki (`www.grid5000.fr`, MediaWiki) login — they can be separate accounts. In practice, the wiki doesn't need to be logged into at all for read-only consultation.
