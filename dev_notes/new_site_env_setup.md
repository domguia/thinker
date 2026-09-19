# Bootstrapping this project's env on a NEW Grid'5000 site — reusable notes

Written 2026-09-19/20 while setting up Nancy for the first time; the pitfalls below turned out to be generic to "any fresh site", not Nancy-specific (per the user's own correction) — kept as a standalone doc so the next new site goes faster, not just Nancy again.

Grid'5000 sites do **not** share a home NFS (confirmed: `ls ~/micromamba/envs/` on a fresh site's home returned nothing even though it's populated on Rennes) — every new site needs its own environment built from scratch.

## Steps that worked, in order

1. **Export the real env from a site that already has it — don't reinstall from memory/guesswork.** A first attempt guessing package versions failed non-obviously; this project also had no `requirements.txt`/`environment.yml` at all before tonight (a real gap, now fixed):
   ```bash
   # on the source site's frontend (e.g. Rennes)
   MAMBA_ROOT_PREFIX=~/micromamba ~/micromamba/micromamba env export -n teacher311 > env_teacher311.yml
   scp env_teacher311.yml <new-site>.grid5000.fr.g5k:~/env_teacher311.yml
   ```
   Committed at the repo root as `env_teacher311.yml`.

2. **Install micromamba fresh on the new site**:
   ```bash
   mkdir -p ~/micromamba
   curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/micromamba bin/micromamba --strip-components=1
   ```

3. **Recreate the conda-managed layer from the exported YAML** (this part reliably works, even across sites/hardware):
   ```bash
   export MAMBA_ROOT_PREFIX=~/micromamba
   ~/micromamba/micromamba create -y -n teacher311 -f ~/env_teacher311.yml
   ```

4. **Install the pip-managed layer in TWO separate commands, not one.** The exported YAML's pip section pins `torch`/`torchvision` with a `+cu121` local version tag that only exists on PyTorch's own wheel index, not default PyPI -- and `transformers`/`scikit-learn` aren't on that index either. A single combined `pip install` against `--index-url` fails **atomically** (pip resolves the whole command as one unit; a single missing package fails the entire install, leaving nothing installed, not a partial success):
   ```bash
   ~/micromamba/micromamba run -n teacher311 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ~/micromamba/micromamba run -n teacher311 pip install transformers scikit-learn numpy
   ```

5. **If `import torch` fails with `RuntimeError: NumPy was built with baseline optimizations: (X86_V2) but your machine doesn't support: (X86_V2)`**: pin an older numpy release with a lower CPU-feature baseline:
   ```bash
   ~/micromamba/micromamba run -n teacher311 pip install "numpy==1.26.4"
   ```
   This fixed it cleanly on `graffiti` (Nancy) tonight. Root cause not fully pinned down (plausibly the specific node's exposed CPU flags, possibly generic to a freshly-provisioned/less-common node rather than anything Nancy-specific) -- but the fix is a one-line, cheap thing to try first before deeper investigation.

6. **Verify `torch.cuda.is_available()` ON THE COMPUTE NODE, not the frontend.** A first check run directly on the site's frontend (`nancy.grid5000.fr.g5k`, no `oarsh`) returned `False` and looked like a driver/CUDA mismatch -- it wasn't; frontends have no GPU at all, so this is **always** `False` there regardless of anything else. The correct check goes through `oarsh` to the actual reserved node:
   ```bash
   OAR_JOB_ID=<job_id> oarsh <node>.<site>.grid5000.fr "export MAMBA_ROOT_PREFIX=~/micromamba; ~/micromamba/micromamba run -n teacher311 python -c 'import torch; print(torch.cuda.is_available())'"
   ```
   Confirmed `True` once checked correctly. **Easy, wasted-time mistake to repeat -- always route the CUDA check through the reservation, never the bare frontend.**

## Summary: full working bootstrap, condensed

```bash
# On source site (has the env already):
MAMBA_ROOT_PREFIX=~/micromamba ~/micromamba/micromamba env export -n teacher311 > env_teacher311.yml
scp env_teacher311.yml <new-site>.grid5000.fr.g5k:~/env_teacher311.yml

# On new site's frontend:
mkdir -p ~/micromamba && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/micromamba bin/micromamba --strip-components=1
export MAMBA_ROOT_PREFIX=~/micromamba
~/micromamba/micromamba create -y -n teacher311 -f ~/env_teacher311.yml
~/micromamba/micromamba run -n teacher311 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
~/micromamba/micromamba run -n teacher311 pip install transformers scikit-learn numpy
~/micromamba/micromamba run -n teacher311 pip install "numpy==1.26.4"   # only if the X86_V2 RuntimeError above shows up

# Verify ON THE RESERVED NODE, not the frontend:
OAR_JOB_ID=<job_id> oarsh <node>.<site>.grid5000.fr.g5k "export MAMBA_ROOT_PREFIX=~/micromamba; ~/micromamba/micromamba run -n teacher311 python -c 'import torch; print(torch.cuda.is_available())'"
```

Total time tonight from "empty home" to a working, GPU-verified env: roughly 30 minutes once the right sequence was known -- most of that was spent on steps 4-6 before they were understood; a repeat run following this doc directly should take a few minutes plus download time.
