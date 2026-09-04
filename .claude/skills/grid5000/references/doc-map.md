# Navigation map — Grid'5000 wiki & docs

Grid'5000 documents almost everything on its wiki (`https://www.grid5000.fr/w/`) and its API (`https://api.grid5000.fr/`). Rather than duplicate content that evolves, here's where to look for what.

**Credentials note**: the platform account password (API/SSH/OAR, e.g. `--netrc` auth on `api.grid5000.fr`) is not necessarily valid on the main web portal `www.grid5000.fr` — it's a MediaWiki that can have a separate account. Don't assume the two are synced without checking. In practice, no login is needed at all for read-only wiki consultation.

## General entry point
- `Getting_Started` — first steps, workflow overview (account → SSH → OAR → job).
- `Category:Portal:User` — index of all user docs, starting point if the exact page isn't known.

## Hardware & availability
- `Hardware` — GPU/CPU catalog per site and cluster (current, don't trust a frozen copy — see the GPU-aware query below for a live check instead of a memorized mapping).
- `Status` — site/platform status.
- `https://<site>.grid5000.fr/drawgantt-svg/` — visual planning of past/upcoming reservations per cluster (useful to choose one-off vs. long reservation based on real contention).
- `https://<site>.grid5000.fr/monika/` — instant node state (up/down/reserved).

## Resource reservation (OAR)
- `OAR_Docs` — general job manager doc.
- `Advanced_OAR` — property syntax (`-p "gpu_model = '...'"`), batch scripts, job arrays.
- `Reserving_specific_resources` — how to target a GPU model, core count, etc.

## Access & network
- `External_access` — SSH, VPN, web access.
- `SSH` — client configuration, keys, ProxyJump.

## Storage
- `Storage` — overview of options (home NFS, group storage, local `/tmp`).
- `Frontend_storage` — home quotas and management.
- `Group_Storage` — persistent multi-TB NFS storage, shareable within a group (`wide`). **Tied to one site's specific server** (`/srv/storage/<name>@<server>.<site>.grid5000.fr`), not replicated across sites — cross-site access goes over the inter-site network (throughput can be a real bottleneck for large volumes in practice, confirmed by field feedback). **No Grid'5000 backup service** — duplicate critical data elsewhere. Lighter alternative if <400 GB and time-limited: request a home quota extension instead of a dedicated group storage.
- `Ceph` — distributed object storage (not a POSIX filesystem), based at Rennes (~9 TB) and Nantes (~7 TB), ~15 TB aggregated total.

## Image deployment / bare-metal
- `Getting_Started_with_Kadeploy` — root access, custom images.
- `Advanced_Kadeploy` — custom images, environments.

## Software & ML environments
- `Software` / `Modules` — preinstalled software per site.
- No CUDA preinstalled in user space by default → Miniconda/Mamba in `$HOME`, or Singularity/Apptainer for Nvidia PyTorch containers.

## Programmable API
- `https://api.grid5000.fr/stable/` — REST API (status, jobs, sites), usable via `curl` without SSH.
- Python client: `python-grid5000` (PyPI) — REST API wrapper.
- **From inside the platform** (a site's frontend), the API responds without authentication.
- **From outside**, HTTP Basic auth is required via `~/.netrc`:
  ```
  machine api.grid5000.fr
  login jdomguia
  password <platform_account_password>
  ```
  (platform account password, potentially different from the wiki's — see the credentials note above)

### Useful queries (curl + jq)

Free/busy summary per node, on one site:
```bash
curl -s --netrc https://api.grid5000.fr/stable/sites/rennes/status.json \
  | jq '[.nodes[].soft] | group_by(.) | map({(.[0]): length}) | add'
```

Same, across several sites at once:
```bash
for site in rennes lyon lille nancy grenoble; do
  echo "=== $site ==="
  curl -s --netrc "https://api.grid5000.fr/stable/sites/$site/status.json" \
    | jq '[.nodes[].soft] | group_by(.) | map({(.[0]): length}) | add'
done
```

Per-cluster free/busy breakdown on a site (group by node-name prefix — useful to compare clusters at a glance, e.g. Nancy's `gros`/`grvingt`/`grouille`):
```bash
curl -s --netrc "https://api.grid5000.fr/stable/sites/nancy/status.json" \
  | jq -r '.nodes | to_entries[] | [(.key | split(".")[0] | split("-")[0]), .value.soft] | @tsv' \
  | sort | uniq -c | sort -rn
```

Find out why a specific node is busy (maintenance, another user's job, etc.):
```bash
curl -s --netrc https://api.grid5000.fr/stable/sites/rennes/status.json \
  | jq '.nodes | to_entries[] | select(.key | startswith("abacus27")) | .value.reservations'
```

Future reservations scoped to one whole cluster (not just one node) — useful for reservation planning, e.g. checking `grouille` at Nancy before booking:
```bash
curl -s --netrc "https://api.grid5000.fr/stable/sites/nancy/status.json" \
  | jq '.nodes | to_entries[] | select(.key | startswith("grouille")) | {node: .key, soft: .value.soft, reservations: [.value.reservations[]? | {user, state, queue, walltime, started_at, message}]}'
```

**Trap hit (2026-09-03)**: on `rennes`, every node showed up `busy` — not real usage, but a site-wide maintenance job (`type: maintenance`) reserving the whole site for its window. Always check a node's `.reservations[].types` before concluding a site is full.

**GPU-aware availability (not yet a ready one-liner)**: cluster hardware specs (including GPU model) are available via `/sites/{site}/clusters.json` and can be joined with the live `soft` state from `status.json` above by cluster name — this avoids relying on a memorized/possibly-stale cluster→GPU mapping. Worth turning into a documented recipe once used a second time.

## Rules & administration
- `Grid5000:UsagePolicy` — usage charter (mandatory reading).
- `Support` — to report a technical issue.
- Users mailing list: `users@lists.grid5000.fr`.

## How to find a specific page

The wiki runs on MediaWiki: search directly at `https://www.grid5000.fr/w/Special:Search?search=<keyword>` if the exact page isn't in this list, rather than guessing a URL.
