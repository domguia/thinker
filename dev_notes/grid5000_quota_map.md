# Grid'5000 — per-site home NFS quota map

Living reference, updated whenever a quota is checked or changed. Check
with `quota -s` from a site's frontend (`ssh <site>.grid5000.fr 'quota -s'`)
— soft/hard limits and current usage are **per site**, not shared
account-wide (each site has its own NFS home, not replicated).

Columns: soft quota (grace-period trigger), hard limit (writes rejected
outright once reached), usage at last check, grace status if over soft.

| Site       | Soft   | Hard   | Last checked usage | Grace          | Checked on   |
|------------|--------|--------|---------------------|----------------|--------------|
| Rennes     | 191G   | 263G   | 233G                | 5 days left    | 2026-09-22   |
| Nantes     | 24.4G  | 97.6G  | 30.4G               | 5 days left    | 2026-09-22   |
| Nancy      | 24.4G  | 97.6G  | 44.8G               | 4 days left    | 2026-09-22   |
| Lyon       | 24.4G  | 97.6G  | 18.6G               | under soft     | 2026-09-22   |
| Sophia     | 24.4G  | 97.6G  | ~0                  | under soft     | 2026-09-22   |
| Lille      | 24.4G  | 97.6G  | ~0                  | under soft     | 2026-09-22   |
| Grenoble   | 24.4G  | 97.6G  | ~0                  | under soft     | 2026-09-22   |
| Strasbourg | 24.4G  | 97.6G  | ~0                  | under soft     | 2026-09-22   |
| Toulouse   | 24.4G  | 97.6G  | ~0                  | under soft     | 2026-09-22   |

## Notes

- **Rennes is the most critical site right now** (2026-09-22): 233G used
  against a 263G hard limit — only ~30GB of headroom, already past the
  191G soft quota with 5 days of grace remaining. This is a bigger risk
  than the Nantes quota incident from the previous day.
- **Rennes quota increase requested by the user, outcome unclear**: the
  user recalls asking for 200GB in a Grid'5000 quota-increase email but
  believes they actually wrote 500GB. Current confirmed Rennes quota
  (checked 2026-09-22) is **191G soft / 263G hard** — neither matches
  500GB, and 263G hard is only slightly above the 200GB figure the user
  remembered asking for. Likely explanation: either the 500GB request
  wasn't (fully) granted, or a different/partial increase was applied
  from an earlier, smaller request. **Not resolved — flag to the user
  directly if they want to follow up with the `wide` group managers**
  (Davide Frey, Mvondo Djob Barbe Thystere — see `grid5000` skill for
  contact info) to check the actual state of that request.
- All non-Rennes/Nantes/Nancy sites are essentially untouched (near-zero
  usage) — safe to stage data there if Rennes/Nantes/Nancy get tight,
  though cross-site transfer cost still applies (see `grid5000` skill on
  never relaying transfers through the local machine).
- Policy reminder (per `agents/INFRA_AGENT.md`): don't delete reusable
  data just because it's redundant — keep it as long as usage stays under
  the *soft* quota (check grace period specifically, not just raw usage
  vs. hard limit). Only trim what's needed to get back under soft.
