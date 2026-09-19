# Experiment Log

## Résumé de nuit 2026-09-19 -> 2026-09-20 (autonomie confirmée par l'utilisateur ~15h35)

**En cours de rédaction, mis à jour au fil de la nuit -- lire la version la plus récente en tête de ce fichier au réveil.**

Réservations posées avant expiration :
- GPU : job 4121144 (7 GPU, abacus11/17/18, jusqu'à ~20:54) -- relève posée job **4121241** (7 GPU, cluster élargi abacus3/10/11/17/18/19/20/21/22/25/29, besteffort, walltime 12h, soumis ~15:38, Waiting).
- CPU : job 4121204 (paradoxe-27, jusqu'à ~21:21) -- relève posée job **4121245** (paradoxe, host=2, walltime 14h, queue normale, soumis ~15:39, Waiting).

Fils lancés ce soir (voir entrées datées ci-dessous pour le détail) : I1 (11/12, 1 cellule manquante en file), I4 (fait), Piste A step-matched (fait, **partiellement rétracté** -- LR non revalidé à d_model=1024), A1/A2 (fait, v1 rétracté pour artefact de mesure, v2 corrigé), I3 (en cours), I2 (en cours), I5 (en file), B1 (en cours, 20 graines). Balayage LR Piste A à d_model=1024 (item [1] de la file de nuit) lancé ~15:38.

**Points "à arbitrer" (à trancher par un humain, pas décidés seuls cette nuit)** -- liste vide pour l'instant, sera remplie au fil des résultats ambigus.

---

## 05 Dec 2023
We got the error below only on GPU, the code worked well on CPU!
```RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [27, 4, 606, 8]], which is output 0 of AsStridedBackward0, is at version 19; expected version 18 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!```

Because of:
`self.k_cache[:, begin_insert_idx:end_insert_idx] = k`

Fixed it in commit `f96161d`, which should be improve for code cleanes and performance

Should improve device assignement, now I have to use: `with torch.device(device):`

## 06 Dec 2023

Oups The model don't learn!

Make the problem simpler:
- task : copy input sequence of 1010 to output 
- run model for 1 step
- use only crossentropy loss

The model learned:
- to pic the right embedding 0, 1
- identify the end of the sequence
- for some sample make perfect copy

Observation:
- the task require more compute than expected, the model observed 1.5e samples = 30.000 batchs * 512 sample/batch * lr = 0.001 per batch 
- at the beginning the loss go down quite easily, after it's constant loss for long time `5e6 samples` then start reducing quickly during that phase the accuracy also improve. My interpretation is that the solution we're looking is rare in the solution space, and difficult to be found via the loss landscape. but ounce around the `exploring stop` and `improvement start` until it get stuck again.
- Now the model learn the task some sample a perfectly predicted but vaerage accuray is 0.5 works better but I didn't manage to 

## 07 Dec 2023

Looking for solution to make the model reach 100% accuracy on copy task, but the compute seem too long.

I move for 0,1 sequence of number to 0,1,2 .., 10 sequence wich with my with give stringe signale to model during trining since the're not just 2 options

![loss plot](/logs/archive_exp_logs/copy_task_w1layer_loss_curve.png)

## 08 Dec 2023

I suspect something wrong with my model implem 

## 18 Dec 2023
I spent a lot of time debugging finally the issue was the model the batch and the sequence lenght was permutted in transformer pytiorch installation that's why my model wasn't leaning anything.
After issue was fixed the model learn quite quickly copy task, with flip, rolled number
Addin 2 step incresea significaly the performance on learning rolled number

After I reagange the model to make it behave like thinker (memeory+step)
I manage to reach 97 accuracy on addition of number on base 16 (max number was 1000)

from a previous training I observe that changing didn't affect performance on this task so I used fix step=1 and latent=4
With batchsize of 4096, learning_rate=0.02, 

I had to implement restart with best model when loss drop and gradient clipping, especially gradient clipping is very effective

I was worrying about the data loading timing, but it seem to be good
```
0.000336 second for data loading
0.011631 second for forward+loss
0.081530 second for backward+remaining
```

I manage to have 95% GPU utilisation even with 2.6Gb usage on Colab T4

I manage to reach 97accuracy for addition of base 16 number between 0-1000 (base10) with bacth size1024 and lr 0.005
![loss accuracy plot](/logs/archive_exp_logs/loss_acc_1addition_base16_max_number1000_hdim32_2step_8latent_lr0.005-0.0002.png)

# 19 Dec
I manage to run 3000 iteration in 2min with 95 accuracy on addition task but didn't reach 100% accuracy  

I observe it took me a lot of time to find the hyperameter and training setup for fast training  

To go faster I can:  
1. run many experiment in the backgroud while doing other stuff -> move to pytorch lighting will help  
2. find simpler problem eg. make copy just by lookig at memory

Project Update:
1. Moved to Pytorch Lighting
2. Avoid un-necessary computation by making optionnal the self attention in nn.TransformerDecoder
3. Update the CopyDataset for larger batch and more task
4. Tried TPU without success du to error on sub libray `import _XLA` 

Some ideas:
For faster training improve the loss, by make it lower for likely output regarding the problem:
1. for LLM task make, compare output distribution unregarding token location eg:
    1. global embedding average, loss = avg(output) - avg(target)
    2. loss on token neigborh
    3. compare all output to all target embedding :
        1. cluster targets embedding with in n centroid with knn, n==output embedding
        2. push the closest each output to move closer to his closest embedding, step after step
2. for number base task, we can make the loss to be condition on the sequence (or just use causal decoding)

# 20 Dec

model with hdim=16 and 9k parameters seam to perform the addition as well, just need more iteration to converge, number of latent and step seam not affect much the performance even the training are the all same while changing latent and step
![heatmap](/logs/archive_exp_logs/heatmap_16hdim_batch1024_20dec_exp.png)
![training curve](/logs/archive_exp_logs/loss_acc_16hdim_latent8_step4_batch1024_20dec_exp.png)

to go faster I should test the model with simpler task, such as copy and thier variants

```python
model_cfg = CfgNode(
    vocab_size = 31+1, n_probe = 1,
    max_latent=64, max_input_len=48, output_len=40,
    d_model=16, nhead=8, d_hid=16*4, nlayers=1, dropout=0
)
data_cfg = CfgNode(low=0, high=16, seq_len=20, batch=1, task=None)
run_cfg = CfgNode(max_iter=4_000, learning_rate=0.01, batch = 1024)
exp_cfg = CfgNode(
    n_latent = range(4, 8+1), # hyper parameters grid search
    n_step = range(1, 4+1),
)
```
the result below
![overview map](/logs/archive_exp_logs/copy_task-seq_len_20-hdim16_20dec_exp.png)

Observation, the model with higher capacity step=4 perform consitenly bad, probably it need more iterations to converge

# 21 Dec

I design a basic currilum learning and it make the copy task much easier to learn, the model reach 100accuracy with that eproach but plateau 60accuracy without it.

How I made the currilum, I vary the sequence lenght in the dataset following an uniform distrubtion, so that the model can easily learn from the short sequence and progressively learn how the longer one's

![with uniform curriculum](/logs/archive_exp_logs/loss_uniform_copytask_acc_16hdim_latent4_step8_batch1024_20dec_exp.png)
![with non uniform](/logs/archive_exp_logs/loss_non-uniform_copytask_acc_16hdim_latent4_step8_batch1024_20dec_exp.png)

Regarding this significan improvement I made a better currilum: by make a dynamic sampling the dataset distrution. With a risk of making the problem non-stationnary, but I think even if is non stationary the loss landscape should be easier the navigate during optimisation. 

# 22 Dec
I implemented a training process that progressivelly increase the difficulty level of the task, wich seem to give better result
![curriculum based](/logs/archive_exp_logs/loss_curriculum_copytask_acc_16hdim_latent4_step6_batch512_22dec_exp.png)

training setup
```
task:
    name = curriculum_copy
    sequence_len = 20
    vocab_size = 16
model:
    hdim = 16
    nhead = 4
    d_hid = 16*4  # feed forward project dimension
    prossing_layer = 1 
    ouput_decoding layer = 1 
running:
    latent = 4
    step = 6
training:
    batch = 512
    learning rate = 0.05
    check_point_auto_reset = 0
```

I observe a plateau,

Hypothesis:
1. the learning signal is weak since there are only ~3tokens failling at the end of the training, the level of success is too to created a good training signal we can observe the loss drop in the figure up here ☝
2. the learned positionnal encoding to be stuck for some last element to learn

An solution to the last hypothesis could a variable position encoding "kinda denoising position"
ideas: I could randomized postional encoding
    eg.1 vary the resolution and the offset of frequency base positional encoder  
    eg.2 sample while keeping order of learned postionnal embedding   


When I increased the seqlen I observe that the model manage to go over the previous performance plateau. It took around 10k iteration to reach %50 wich is approximate to 20seqlen prediction and after 30k step the model reach 30seqlen copy before have a peak loss (I should implement a better model checkpoint reload!)  
![curriculum based with 40 seqlen](/logs/archive_exp_logs/loss_curriculum_copytask_seq-len-40_16hdim_latent4_step6_batch512_22dec_exp.png)

At 50seqlen I have un error realted to CUDA even if I only use 0.3/15 GB of GPU MEM! 

So I increased the vocabulary size from 16 to 32, even with that the model dont seem to be at capacity
![curriculum based with 40 seqlen](/logs/archive_exp_logs/loss_curriculum_copytask-vocabsize_32_seq-len-40_16hdim_latent4_step6_batch512_22dec_exp.png)

hypothesis for plateau:
learning rate 0.01 too high, might explain the noisy loss curve even with 512 batch size

conclusion: having a plateau doesn't mean that the model is at capacity
I still suspect the position encore to be cause 

## varying position embedding
the model stuck at .2 accuracy with `varying position embedding`

![varying position embedding](/logs/archive_exp_logs/loss_varying-position-emb_copytask-vocabsize_16_seq-len-10_16hdim_latent4_step6_batch512_22dec_exp.png)

Sometimes you just have to train longer and/or make the task easier
after some iteration, I made it WORK!!

key_parameters
```
task:
    name = curriculum_copy
    sequence_len = 5 (and 10 in the 2nd experiment)
    vocab_size = 16
model:
    max_input_len  = 5 *4, # x4 space for varying position embedding
    max_output_len = 5 *4, # x4 space for varying position embedding

    hdim  = 32
    nhead = 8
    prossing_layer = 1 
    ouput_decoding layer = 2 # big change
```
_Note: I enable self attention in output TransformerDecoder, it was disable before since we where querying the model directly with the output query_

![varying position embed first work!](/logs/archive_exp_logs/loss_varying-position-emb-WORK_copytask-vocabsize_16_seq-len-5_input-output-5x4_32hdim_2layer-output_latent4_step6_batch512_22dec_exp.png)

for the second experiment I changed
```
sequence_len = 10
max_input_len  = 10 *3,
max_output_len = 10 *3,
```
![varying position embed 2nd work!](/logs/archive_exp_logs/loss_varying-position-emb-WORK_copytask-vocabsize_16_seq-len-10_input-output-10x3_32hdim_2layer-output_latent4_step6_batch512_22dec_exp.png)



# Jan 6

No experiment has been conducted recently, but I have implemented a full auto regressive flow (i.e., [Perceiver AR](https://arxiv.org/abs/2202.07765)). This ensures that the model will have a history from the beginning to the end of the multi-step forward pass, providing a stronger signal for easier optimization.
- With an appropriate mask, this will ensure there's no data leakage from future tokens given as input.
- Since there's more 'capacity' in the latent flow than in a single token flow, I expect the optimization to transfer the required computation to the latent flow.
- The capabilities learned in the latent flow could easily be transferred to the standard output, which has a weaker signal compared to the full flow autoregressive process.
:-) for longuer context we can use mistral layer to layer window shift 

I should implement a planned flow runner, that will help to implement different strategy or even random strategy

strategy: do all the time  
step | read_input | mem_lookup | mem_write  | output  
-----|------------|------------|------------|-------  
  1  |     1      |      1     |      1     |   1     
  2  |     1      |      1     |      1     |   1     
  3  |     1      |      1     |      1     |   1     

strategy: read ounce at the begging, output at the end with mem  
step | read_input | mem_lookup | mem_write  | output  
-----|------------|------------|------------|-------  
  1  |     1      |      0     |      1     |   0     
  2  |     0      |      1     |      1     |   0     
  3  |     0      |      1     |      1     |   0     
  4  |     0      |      1     |      0     |   1     


list of attributes:
- read_input
- mem_lookup
- mem_write
- static_mem_lookup
- output

_This is not yet implemented!_

# March 8
Empty brain!  
Should I implement, language model distillation task? as initially planned  
or keep going with toy task and model?

# Sept 18
Ca marche!

![learned converging compute](/logs/archive_exp_logs/ca-marche_with_scaled_loss_overstep-learned_cumsum.png)

```
    learning_rate=0.001,
    batch = 128,
    n_latent = [range(4,16+1,2)], # max latent -> 16
    n_step = [range(4,12+1)],     # max step   -> 12
```

At epoch 30k I paused the training, and incresed the compute capacity and this happened, the model generalized immediatly for all compute capacity
![learned converging compute after change](/logs/archive_exp_logs/ca-marche_with_scaled_loss_overstep-learned_cumsum_after-change.png)

![learned converging compute curve](/logs/archive_exp_logs/ca-marche_with_scaled_loss_overstep-learned_cumsum_curve_after-change.png)

Changelog when increased compute capacity
```
    learning_rate=0.01,
    batch = 1024,
    n_latent = [range(4,24+1,2)], # max latent -> 24
    n_step = [range(4,16+1)],     # max step   -> 16
```

hypothesis pertubation help, shake the model and might lead to convergence since the optimization is on going

### add pertubation in latent
```
# latent = self.embd_latent(pos[:,:n_latent]) # B, L, H
latent = self.embd_latent(pos[:,0:1]) + torch.normal(0, .1, size=(B, L, H)) # B, L, H
```

# Sept 19
With lower training iteration, we observe this, the same partern with lower accuracy
![accuracy_leverage_by_compute](logs/archive_exp_logs/ca_marche-no_scaled_loss-overstep-learned_cumsum.png)
![accuracy_leverage_by_compute](logs/archive_exp_logs/ca_marche-no_scaled_loss-overstep-learned-cumsum-curve.png)


## 2026-09-03/04 — Distillation onboarding: data prep, Teacher benchmark, first SFT + KD runs

Full run on Grid'5000 (cluster/job details live in `dev_notes/grid5000_usage.log.md`, not here).

### Data prep (reasoning / general / retrieval)

Ran all three `prepare_*_data.py` scripts against the real Hub datasets (previously only smoke-tested locally):
- `reasoning` (OpenR1-Math-220k): 38,057/93,733 kept (55,676 skipped — no verified `<think>` trace, or too long at `max_length=4096`), ~124 ex/s.
- `general` (WikiText-103 + TinyStories): 40,000/40,000 kept, 0 skipped, ~14s — TinyStories especially is very forgiving on length.
- `retrieval` (HotpotQA distractor): 90,446/90,447 kept, essentially no loss.

Added unfiltered `raw.jsonl` capture to all three scripts (writes every streamed example with a `kept: true/false` flag, not just the ones that pass filtering) — previously, skipped examples were discarded with no trace, which matters most for `reasoning` given its high skip rate. Re-ran `reasoning` with this enabled: same filtering outcome, `raw.jsonl` now available (93,733 rows) for future filter-criteria tuning.

### Teacher: Qwen3.8-27B-FP8

Chosen Teacher: `Qwen/Qwen3.8-27B-FP8`, a Qwen3.5 vision-language checkpoint (`Qwen3_5ForConditionalGeneration`) used purely as a text model. 30.89 GB on disk. Architecture detail that matters for anything touching hidden states: 64 layers total, only 16 "normal" attention, the other 48 are gated DeltaNet (linear attention with recurrent state) — see `learn/distill/qwen3.8-27b-notes.md` for the full writeup (quant landscape, reasoning_effort API, GPU-generation notes).

Bugs hit and fixed getting `bench_teacher.py`/`precompute_teacher_targets.py` to actually run this checkpoint:
- `--reasoning_effort none` is documented (Unsloth docs) as a valid 4th level alongside xhigh/medium/low, but this checkpoint's real chat template **rejects it with a hard `jinja2.exceptions.TemplateError`**, not a silent fallback as our own code comment assumed. Only xhigh/medium/low actually work; use `low` as the fastest available proxy.
- `model.config.num_hidden_layers` doesn't exist on `Qwen3_5Config` (a VLM wrapper) — the real value lives at `model.config.text_config.num_hidden_layers`.
- Missing `pillow`/`torchvision` block `AutoProcessor.from_pretrained` even though we never pass image input — the VLM image-processor class is still instantiated as part of loading.

Throughput investigation (Ampere, A100 40GB): first suspected missing `causal_conv1d`/`flash-linear-attention` optimized kernels (the 48 DeltaNet layers fall back to slow reference PyTorch ops without them) — compiled both from source (needed installing a CUDA toolkit via conda-forge/nvidia channels first, since the node has no `nvcc` by default) and confirmed they load and the "falling back" warnings disappear. **Throughput didn't change** (still ~0.3-0.6 tok/s). Real cause: `bench_teacher.py` forces `dtype=bfloat16` on load regardless of the checkpoint's native FP8 format, dequantizing to ~55.6 GB — too big for a 40 GB A100, so `transformers` silently offloads part of the model to CPU, and that CPU↔GPU traffic per generated token is what actually dominates. Not yet fixed in the script (should load without forcing bf16, or run on a GPU with native FP8 support instead). Switching to an L40S (Ada, compute capability 8.9, the lowest tier with native FP8 tensor cores) sidesteps the issue entirely — loaded fast, no offload, no dequantization warning.

Top-K=32 storage formula (`K*6+2` bytes/token) has now been validated exactly against real measurements three separate times (194.0 bytes/token measured every time) across different runs/GPUs.

### First SFT run (10M student, no KD)

`train_sft.py` had a real bug: it never moved the model or batch tensors to a CUDA device, so it would have silently trained on CPU even on a GPU reservation. Fixed. First real run on GPU: 9.44M-param random-init GPT-2-arch student (gpt2 tokenizer), 200 steps on the real `reasoning` data in 3.9s on an A100, loss 10.85→5.11. Confirms the full pipeline (real data → tokenizer → from-scratch model → GPU training loop) works end-to-end. This is plain next-token cross-entropy, no Teacher involved yet.

### First KD run (logit-level distillation)

Implemented in `train_sft.py`: `--teacher_targets`/`--kd_alpha` CLI args and a `topk_kd_loss()` function computing KL(teacher‖student) over the (K+1)-way categorical formed by the Teacher's Top-K token indices plus one merged "everything else" bucket, reconstructed via the residual log-sum-exp exactly as stored by `precompute_teacher_targets.py`. This is the only distribution shape exactly recoverable from Top-K-only storage — treating the untracked tail of the vocabulary as one lumped outcome rather than ignoring it or assuming it's zero.

Also fixed a latent bug: the student's model config always used `--base_config`'s own vocab size, silently ignoring `--tokenizer` — meant nothing crashed if you used a mismatched tokenizer, it would just produce token ids out of the embedding table's range. Now sets `config.vocab_size = len(tokenizer)` unconditionally.

Validated the loss function before spending any GPU time on it:
1. Local CPU smoke-test: tiny synthetic Top-K `.npz`, 1.7M-param gpt2-arch model, ran without shape errors, both CE and KD terms decreasing over 10 steps.
2. Numeric correctness check: built "teacher" targets directly from a student's own logits (so they should be identical distributions) and measured `KL ≈ 0` (`-1.6e-7`, i.e. numerical noise) — confirms the KL math itself is correct, not just that the code runs.

Real run: precomputed Top-K=32 targets for 200 `reasoning` examples using the actual Teacher on the L40S (had to drop `--max_length` from the default 4096 to 512 to avoid an OOM — the transient `(seq_len, vocab≈152k)` logits/mask tensors on top of the model's own ~40 GB footprint don't fit in 44 GB otherwise). Then trained a student with `--teacher_targets` pointed at that precomputed file, `--tokenizer` pointed at the Teacher's own snapshot (needed so Top-K indices refer to the same vocabulary as the student) — this pushed the student to **41.09M params** instead of the earlier 10M, because the Teacher's vocabulary's embedding table dominates parameter count at this width; the original "10M" target was implicitly calibrated for gpt2's much smaller (50k) vocab. Correction (2026-09-04): the Teacher's own tokenizer (`Qwen3.8-27B-FP8`, a Qwen3.5 VLM) has **248,077** tokens, not the ~151,936 figure cited in `README.md` for plain Qwen3 — that figure is correct for the *data-prep* tokenizer (`Qwen/Qwen3-0.6B`, used by `prepare_*_data.py`), a genuinely different, smaller tokenizer than the Teacher's own. The two are not interchangeable; KD training must use the Teacher's own tokenizer for index alignment (as it does), while data-prep length filtering used the smaller one — a minor, currently-unquantified discrepancy in effective token counts between the two stages. 200 steps in 37.9s: CE 11.38→~1.5-1.8, KD/KL 0.41→~0.29, best combined loss 0.673. Both loss components move sensibly, first genuine end-to-end KD result. Full config/metrics/log: `logs/EXP-003-kd-topk32-baseline/` (adopting the `logs/EXP-XXX/` convention already used elsewhere in this repo — going forward, this narrative log points to an EXP-XXX per run instead of inlining metrics).

Open follow-ups: decide whether to keep the ~41M student size (Teacher-vocab-aligned) or find a way to shrink it back toward the original 10M budget (smaller `n_embd`, or a separate small vocab with an explicit index-remapping layer against the Teacher's Top-K ids); scale the precompute step past the 200-example/512-token validation slice to the full 38,057-example `reasoning` set (needs the OOM addressed for the full 4096-token budget, likely via a bigger-VRAM GPU or a leaner residual computation).

### muP (hyperparameter transfer across width)

Decision (2026-09-04): given the project's stated target of scaling from ~10-41M up toward 500M-3B, tuning LR/init fresh at every tier is wasteful and, under standard parametrization, systematically misleading (optimal LR shifts with width, so values tuned small don't transfer). Implemented the two components of muP (Yang et al., *Tensor Programs V*) with the largest effect on transfer per the original paper: (1) hidden matmul weights (attention/MLP projections) get init std scaled by `1/sqrt(width_mult)` and Adam LR scaled by `1/width_mult`, both relative to a `--mup_base_width` reference; (2) the LM head is untied from the input embedding (a muP requirement — tied weights can't satisfy both the embedding's and the readout's scaling rules at once), zero-initialized, and its output logits divided by `width_mult` before the loss. Embeddings, LayerNorm, and biases keep the framework's default init and the base (unscaled) LR, per the muP table.

**Explicitly not implemented** ("muP-lite", not full muP): attention's `1/sqrt(d_head)` logit scaling is not patched to muP's `1/d_head` convention — would require reaching into the specific attention implementation's internals in a version-fragile way. Revisit if empirical transfer doesn't hold well across the widths actually used.

Validation done: local CPU smoke-tests (`--mup` alone, and combined with `--teacher_targets`/KD) ran without shape errors, losses decreasing sensibly. Sanity check on the zero-init readout: initial loss ≈ 10.82 ≈ ln(vocab_size≈50257), exactly matching the theoretical value for uniform logits at init — confirms the readout scaling wiring is correct.

**Real GPU width-transfer test (2026-09-04, `abacus21`/A100, real `reasoning` data)**: 4 learning rates (0.003/0.01/0.03/0.1) × 3 configs (narrow+muP at width 40, wide+muP at width 160, wide-without-muP at width 160), plain CE, 60 steps each. Result: a real if modest confirming signal — at the most aggressive LR (0.1), the no-muP wide model's loss drifted clearly worse (6.035) than the narrow-muP baseline (5.809), while the wide-muP model tracked the baseline almost exactly (5.811); 3 of the 4 LRs favored muP tracking the baseline more closely, one (0.01) was a coin flip. Not a dramatic explosion/instability demo, but a directionally consistent, codebase-specific result — an upgrade from the earlier inconclusive CPU toy test.

**Tied-head compromise (2026-09-04)**: canonical muP requires untying the LM head from the input embedding (they need different init/scaling rules, impossible to satisfy both on one shared tensor — see `learning_journal.md`'s "weight tying" entry for the full explanation). With this project's large Teacher-aligned vocabulary (248,077 tokens), untying doubles an already-huge table, which directly worked against a separate, related decision: **model size tiers now refer to core size (transformer blocks only), not total size**, since the vocab-sized head has nothing to do with the model capacity actually being scaled and dominates small cores badly (`README.md`'s "Student size" section has the full reasoning and the corrected 40M/150M tier numbers). Resolution: `train_sft.py --mup` now keeps the head tied by default (`--mup_untie_head` opts into canonical muP instead), still applies muP's core init/LR scaling and the logit/`width_mult` rescaling at the loss, just skips the zero-init/untie step — a documented, deliberate deviation from the paper, not the full recipe.

**150M-core tier run** (`logs/EXP-004-kd-150Mcore-mup/`): `n_layer=12, n_embd=1024, n_head=16` → 152.2M core + 254.0M tied head = 406.2M total (not the ~280M initially mis-estimated by reusing an earlier width's head size by mistake — corrected before writing anything down permanently). Reused the LR (0.003) found optimal for the muP base width (40) in the transfer test above, unchanged, across a 25.6× width jump — converged cleanly, best combined loss 0.288 (CE 12.19→0.60, KD 0.40→0.28), notably better than the 40M-core run's 0.673 on the same 200-example validation slice, with no signs of instability from the un-retuned LR.

Treat muP (the tied-head variant actually in use) as implemented, smoke-tested, and now backed by one real-data GPU transfer experiment showing the expected directional effect — not a fully rigorous, high-confidence validation (would want a bigger width ratio and a clearer LR-instability demonstration), but no longer "theory only."

**500M-core tier run** (`logs/EXP-005-kd-500Mcore-mup/`, job 4091486 on `abacus26` L40S): `n_layer=25, n_embd=1280, n_head=16` → 493.2M core + 317.5M tied head = 810.8M total. Same base LR (0.003) reused unchanged from the width-40 tuning run, now across an 80× width jump from the tuning base (32× from the 150M-core tier's own `width_mult`). Trained cleanly, no divergence: 200 steps in 77.3s, best combined loss 0.382 (CE 12.25→1.12, KD 0.41→0.28). Slightly worse than the 150M-core tier's 0.288 on the same 200-example validation slice (expected — capacity isn't the bottleneck at this data size, and this is still the tiny validation slice, not a real training run), but the LR transferred without retuning and without instability across three tiers now (40M → 150M → 500M-core), which is the actual thing muP was adopted to guarantee.

**LR/kd_alpha/weight_decay sweep, and a real limit of the muP-transfer claim (2026-09-04, `abacus30` V100×1, `logs/EXP-006-hp-sweep` group on W&B/MLflow)**: prompted by the observation that `lr=0.003`/`kd_alpha=0.5` had never actually been shown to be *good*, only shown to *transfer* — ran a full grid at the 40M-core tier (LR ∈ {1e-4..1e-1}, `kd_alpha` ∈ {0.25,0.5,0.75}, `weight_decay` ∈ {0,0.01,0.1}), all on the 200-example validation slice. Found a clear optimum at 40M-core: `lr=0.01` (best_loss 0.195, vs. 0.344 at the previously-used 0.003 — a 1.8× improvement), `kd_alpha=0.25` slightly better still (0.146, vs. 0.195 at alpha=0.5), `weight_decay` had negligible effect (0.192-0.199 across 0/0.01/0.1).

**Then re-tested this "improved" config at 150M-core to confirm transfer — it did not transfer.** `lr=0.01` (with the old `kd_alpha=0.5`) gave best_loss 1.276 at 150M-core, dramatically worse than the 0.288 already achieved with the original `lr=0.003`. A finer probe (`lr` ∈ {0.003, 0.005, 0.007, 0.01} at 150M-core, `kd_alpha=0.5` held fixed) showed a monotonically *increasing* loss with LR at this tier (0.288 → 0.306 → 0.374 → 1.276) — the exact opposite ranking from the 40M-core tier, where higher LR (up to 0.01) monotonically *decreased* loss. `kd_alpha=0.25` also transferred badly (0.423 at 150M vs. 0.288 at alpha=0.5, with `lr=0.003` held fixed to isolate the effect).

**Likely cause, not yet fixed**: the tier progression (40M→150M→500M-core) scales `n_layer` (4→12→25) *and* `n_embd` (160→1024→1280) simultaneously. muP's width-invariance guarantee applies to `n_embd` (width) only — it says nothing about depth, and deeper networks are well known to need lower LR for stability independent of muP (residual-stream accumulation across more layers). So the "transfer test" done across tiers was never a clean width-only test; depth was a confound the whole time. This explains why the original `lr=0.003`/`kd_alpha=0.5` — never verified as optimal at any single tier, just carried forward unchanged — turned out to be a better multi-tier compromise than the value that *was* verified optimal, but only at the shallowest, narrowest tier alone.

**Practical upshot (superseded below)**: no retraining needed — `EXP-003/004/005` already used the better (if accidentally so) hyperparameters for this depth+width co-scaling trajectory. The sweep's value was diagnostic, not a config upgrade: it turned an untested assumption into a confirmed (if not fully understood) one, and surfaced a real methodological gap — true muP guarantees would require holding depth fixed (isolating width) or adding a depth-specific correction (e.g. residual-branch `1/sqrt(n_layer)` scaling, "Depth-muP"-style) — neither implemented yet. Revisit before trusting LR transfer to much deeper tiers (1B/3B-core) if those also increase depth alongside width.

### Depth-muP-lite: implementing and testing the fix (2026-09-04, same session)

Implemented the fix flagged above: `apply_depth_mup_scaling()` in `train_sft.py` (new `--depth_mup`/`--mup_base_depth` flags) scales each transformer block's residual branch OUTPUT (attention and MLP, before the residual add) by `1/sqrt(n_layer / mup_base_depth)`, via `register_forward_hook` on each block's `.attn`/`.mlp` submodules rather than monkeypatching `GPT2Block.forward` (verified against the installed transformers version that `GPT2Attention` returns `(attn_output, present)` and `GPT2MLP` returns a plain tensor, so hooks can rescale the branch cleanly without touching the internal residual-add). "Lite" in the same sense as the existing width-muP-lite: only the forward-pass branch magnitude is corrected, not the branch's own init variance or a matching LR term — the single largest-effect piece, not the full canonical recipe.

Re-ran the exact case that broke transfer above (`lr=0.01`, `kd_alpha=0.5`, the config optimal at 40M-core) at both bigger tiers, with `--depth_mup --mup_base_depth 4` (4 = the 40M-core tier's own `n_layer`, so `depth_mult=1` there and the correction is a no-op at the tuning base — confirmed: 40M-core with `--depth_mup` gave the byte-identical 0.194570 as without it):

| tier (n_layer) | `lr=0.01`, no depth_mup | `lr=0.01`, with depth_mup | (reference: old `lr=0.003`, no correction) |
|---|---|---|---|
| 40M (4) | 0.195 | 0.195 (neutral, as expected) | 0.344 |
| 150M (12, depth_mult=3) | 1.276 | **0.469** (2.7× better) | 0.288 |
| 500M (25, depth_mult=6.25) | 2.356 | **0.320** (7.4× better, and *beats* the 0.382 old baseline) | 0.382 |

Depth-muP-lite substantially restores transfer at both deeper tiers, and at the largest depth ratio tested (500M, 6.25×) the "optimized-at-40M" LR combined with the depth correction actually **beats** the old untuned baseline — the first config found in this whole investigation that's simultaneously good at the small tuning tier AND competitive-or-better at the largest tier trained so far. At 150M (3× depth ratio) it closes most but not all of the gap to the baseline (0.469 vs. 0.288) — plausibly noise from the tiny 200-example single-seed validation slice, or a real residual gap that the "-lite" version (no init/LR-side depth correction) doesn't fully close; not yet distinguished.

**Practical upshot (current)**: `--mup --depth_mup --mup_base_depth 4 --lr 0.01` is now a credible candidate default for the model-size-tier roadmap — tune once at the smallest/shallowest tier as originally intended, transfer to both width and depth increases together. Given the remaining (if reduced) gap at 150M, this isn't yet a slam-dunk "always use this" result off a single 200-example slice; worth confirming on a larger validation slice (or the real training set) before committing it as the new default for the next tiers (1B/3B-core), but it's a real, working fix for the depth confound identified above, not just a diagnosis this time.

### ⚠️ Critical correctness bug found (2026-09-05): the FP8 Teacher's weights were never actually dequantized

While validating the FP8-vs-bf16 evaluation methodology (comparing Top-K
outputs from `Qwen/Qwen3.8-27B-FP8` against `Qwen/Qwen3.8-27B` bf16 on the
same 40-example slice), found **0% top-1 agreement** — nowhere near the
expected ~96-99% from the reference quantization literature. Root cause,
confirmed directly by inspecting the loaded model: `transformers`
(5.17.0.dev0) never applies the FP8 `weight_scale_inv` dequantization scale
for this checkpoint (`quant_method: fp8`, `fmt: e4m3`) — every scale tensor
is loaded then discarded as `UNEXPECTED`, `model.is_quantized` is `None`,
and every linear module is a plain `nn.Linear` holding the raw FP8 bytes
reinterpreted as bf16. Full writeup and evidence in
`learn/distill/qwen3.8-27b-notes.md`'s new "CRITICAL" section.

**This means every KD run above (`EXP-003/004/005`, the LR/kd_alpha/weight_decay
sweep, the Depth-muP-lite results in this same section) trained against
Top-K Teacher targets precomputed from this broken FP8 load path** — the
loss numbers are real, but the "Teacher signal" being distilled was
numerically incoherent, not real Qwen3.8-27B knowledge. The muP/Depth-muP
transfer conclusions (about LR/optimizer behavior across width and depth)
are probably still valid as relative comparisons — the targets were
garbage but *consistently* garbage across all three tiers, and the question
being tested was whether a given LR/depth-correction transfers, not how
good the resulting student is. But no number here should be read as
evidence of real distillation quality, and the combined-loss magnitudes
(0.19-2.36 across these tables) don't mean what they were assumed to mean.

**Fixed and verified same day.** Root cause: `load_model_and_tokenizer`
passed `quantization_config=None` explicitly, suppressing transformers'
auto-detection of the checkpoint's native FP8 scheme. Fix: omit the kwarg
unless bnb quantization is requested. Re-ran the comparison after the fix:
**98.92% top-1 agreement** vs bf16 (matches the ~98.9% 8-bit literature
reference) — confirms the fix and validates the comparison methodology.
Full writeup in `learn/distill/qwen3.8-27b-notes.md`'s "RESOLVED" section.
`EXP-003/004/005` above still predate the fix and used the broken load path
— still worth deciding whether to rerun them against real Teacher targets
before trusting any Teacher-signal-quality conclusion from them.

## 2026-09-12 — Indexed Attention Phase 0: hierarchy vs. flat go/no-go, and an n_facts scale cliff (`EXP-007`)

**Context**: first GPU validation of `HierarchicalMemory`/`IndexedThinker` (`dev_notes/indexed_attention_experiment_plan.md` Phase 0), beyond the CPU overfit tests in `tests/test_indexed_memory.py`. Grid'5000 Rennes, job 4104802 (`abacus3-1`, 4×A5000 24G besteffort, `~/micromamba/envs/teacher311`). Script: `learn/indexed_attention/train_kb_retrieval.py` (new file).

**LR does not transfer across scale (found before the main result)**: `lr=3e-3` (used in the CPU unit tests) fails outright once `n_facts`/`d_model` grow. A sweep at `n_facts=16, d_model=128` found `lr=3e-4` converges cleanly (98.6% held-out acc in 2 min) while `1e-4/3e-3/1e-2` all fail at chance — same muP-style width/LR miscalibration already documented in this log's distillation section, rediscovered independently here.

**Go/no-go result at `n_facts=16` (3 seeds each, held-out eval split, `lr=3e-4`, `d_model=256`, `n_step=3`, 8 min/run)**:
- `depth=4` (hierarchical): **99.6% / 99.5% / 100.0%** held-out accuracy
- `depth=0` (flat, Baseline C, spec §9): **7.0% / 6.25% / 5.86%**

Massive, unambiguous gap (>>2σ) — confirms Phase 0's hypothesis cleanly: the hierarchical, unified-softmax memory dramatically outperforms flat attention once the KB is large enough to dilute the flat baseline's signal. Decision per the plan's table: **go** — continue with `depth>0` as the default config into Phase 1bis.

**But: `n_facts=64` (256 leaves, same `d_model=256`) fails to learn at all, for BOTH `depth=0` and `depth=4`** — not a hierarchy-specific weakness, the whole register+SM+fusion mechanism plateaus at chance level. Isolated the axis with 4 short (5 min) diagnostic runs, one per GPU:
- `n_facts=64, d_model=128, lr=3e-4` → fails (0.2% acc)
- `n_facts=16, d_model=256, lr=3e-4` → succeeds (99.7%) — control confirming `d_model=256` alone isn't the problem
- `n_facts=64, d_model=256, lr=1e-4` → fails (0.5%)
- `n_facts=64, d_model=256, n_step=6` → fails (0.6%)

Neither `d_model` (128 vs 256), nor `lr` (1e-4 to 3e-3 tried across two sweeps), nor `n_step` (3 vs 6) move the needle — it's specifically **`n_facts`** (task scale) that breaks learning. Two more capacity-style levers tested and also ruled out: `n_register=4` (0.3% acc) and `batch_size=256` (0.3% acc) — neither helps either.

**Curriculum learning (matches this log's own 18 Dec 2023 ToyThinker/copy-task precedent — "having a plateau doesn't mean the model is at capacity", solved there by curriculum, not by hyperparameter tuning)**: `thinker-e9` (sister session) implemented fixed-shape padding/masking (`HierarchicalMemory.build(..., leaf_mask=...)`, commit `d7bff12`) so `data/kb_retrieval.py::KBRetrievalDataset(n_facts, max_facts, ...)` can vary real fact count while keeping the hierarchy shape (`block_size`/`depth`) fixed to the curriculum's target scale. `train_kb_retrieval.py` got a `--curriculum "16,32,64"` stage-promotion loop (promote on held-out accuracy threshold). Real GPU curriculum run (16→32→64, same job/node) in progress at time of writing — see next entry or `dev_notes/grid5000_usage.log.md` for the outcome.

**Role split for this work going forward** (per explicit user instruction): `thinker-e9` (sister session, "model-design") owns architecture/spec/plan changes (`core/`, `dev_notes/indexed_attention_spec.md`, `dev_notes/indexed_attention_experiment_plan.md`); this session owns experiment execution — Grid'5000 job orchestration, `learn/indexed_attention/` training-loop code, profiling/optimization, and reporting observations back for the plan to be updated.

### Curriculum result (same day, continued): confirms the mechanism, and re-confirms the flat baseline's failure

Ran the `--curriculum 16,32,64` promotion loop (`train_kb_retrieval.py`, promote at held-out acc ≥0.9, min 500 steps/stage) for real on GPU (job 4104802, `abacus3-1`), `depth=4` and `depth=0` in parallel, 18 min budget each:

- **`depth=4` (hierarchical): fully unblocked.** Promoted cleanly through all three stages and finished at `n_facts=64` (the original blocker) with **100% held-out accuracy**, `best_loss=0.00035`, in 8806 steps / 1082s. Confirms the Dec-2023 ToyThinker curriculum precedent transfers directly to this architecture: the plateau at `n_facts=64` was never a capacity/mechanism ceiling, just an optimization-landscape one that curriculum routes around.
- **`depth=0` (flat): never left stage 1.** After 22962 steps (>3× the steps `depth=4` needed for its *entire* 3-stage curriculum), held-out accuracy was still stuck at ~5%, never crossing the 0.9 promotion threshold even once — consistent with, not contradicting, the go/no-go result above (flat attention already failed at `n_facts=16` directly; giving it unlimited time at the same stage doesn't change that). Useful negative control: curriculum only helps when the underlying mechanism *can* eventually solve the harder stage — it's not a universal fix for any plateau.

**Practical upshot**: curriculum learning (train `depth>0` at `n_facts=16→32→64`) is now the validated path to scale this task past the `n_facts=64` cliff found in Phase -1, with zero architecture changes. Recorded as the reference training recipe for any future phase needing `n_facts>16`.

### Phase 1bis first pass (n_facts=16, 3 seeds/variant): no differentiation at this scale

Ran `n_slots=4` (vs. default `n_slots=1`), `use_ff=True` (2-layer GELU MLP in the main-loop fuse step), `detach_sm_keys=True` (stop-gradient on SM keys), and `level_dropout_p=0.1` (stochastic high-level dropping), 3 seeds each, same `n_facts=16` task/budget as the go/no-go run:

| Variant | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Baseline (`n_slots=1`, no FF/detach/dropout) | 99.6% | 99.5% | 100.0% |
| `n_slots=4` | 98.9% | 99.7% | 99.8% |
| `use_ff=True` | 99.5% | 100.0% | 100.0% |
| `detach_sm_keys=True` | (run duplicated by an orchestration mistake, discarded) | 100.0% | 99.7% |
| `level_dropout_p=0.1` | 99.8% | 100.0% | 99.8% |

All variants land in the same 98.9-100% band as the baseline — **no variant shows a distinguishable effect at this scale**, per the plan's own ambiguity rule (overlapping ±σ → "not concluded," not "no effect ever"). `n_facts=16` is a near-ceiling task for every configuration tried so far; a real test of `use_ff`'s composition-capacity hypothesis (the audit's priority #1) needs either the harder `n_facts=64` curriculum-trained setting or a task that actually requires computing over retrieved values (Phase 2 multi-hop), not more distractors at a task every variant already solves near-perfectly.

**Infra note for future large parallel batches**: piping `oarsh` stdout straight back over the orchestrating SSH connection is fragile — a transient connection drop (happened twice this session, unrelated to the remote job) kills the local pipe and loses all output, even though the remote training process itself survives untouched (confirmed via `ps`/`nvidia-smi` on the node). Switched to `... > ~/remote/path/logfile 2>&1 < /dev/null & disown`, launched via a short-lived `oarsh` call that returns immediately — the remote process then belongs to the OAR cgroup, not to the SSH session, and results are recovered with a plain `cat`/`tail` afterward regardless of local connection hiccups. Adopt this pattern by default for any run expected to outlive a single quick command.

### Multi-hop chain task (Phase 2 / Phase 1quater) and batch-size/LR co-scaling: session close-out

**Multi-hop (`data/kb_chain_retrieval.py::KBChainDataset`, thinker-e9's implementation)**, curriculum on `n_hops` (1→2→3, `n_distractors=1`, `depth=2`), 15 min budget each, comparing fixed `n_step=12` vs. randomized `n_step~Uniform(1,12)` per batch (Universal/Looped-Transformer-style depth randomization, for later extrapolation testing):

- `n_hops=1` promotes cleanly and fast in both runs (~step 1200, ~98% held-out acc) — the mechanism can do a single hop easily even under a large `N_step` budget.
- `n_hops=2` **plateaus at 32-34% held-out accuracy in BOTH runs**, never crossing the 0.9 promotion threshold despite 16000-22000+ further steps (well past `curriculum_min_steps=500`) — no measurable difference between fixed and randomized `N_step`, so depth-randomization isn't the fix here. Not chance level (~2-3% at this vocab size) but far from mastery.
- **Not yet diagnosed which of three explanations is correct** (flagged to thinker-e9, unresolved at session close): (a) `N_step=12` still insufficient for `n_hops=2` despite already being ~6x the hop count and above thinker-e9's own "~2-4x n_hops" heuristic, (b) `lr=3e-4` (carried over from the single-lookup task, never re-swept for the chain task specifically) is miscalibrated for this harder task -- same lesson as every other axis in this session, or (c) a genuine mechanism limit on 2+ hop composition through the SM buffer. Whichever it is, this is the most important open question for validating the project's central "iterative extraction beats single-pass" thesis -- next step should be an LR sweep specifically at `n_hops=2` (not reusing the retrieval-task LR) before concluding anything about (c).
- Added `--extrapolate_n_steps "8,16,24"` to `train_kb_chain.py`: in-memory re-evaluation at `N_step_test > N_step_train` on a fresh disjointly-seeded KB, no checkpoint needed (thinker-e9's suggestion, since this is a cheap go/no-go check, not something needing to persist/reuse a checkpoint elsewhere) -- ready for the next chain run but not yet exercised on a converged model (nothing converged past `n_hops=1` this round).

**Batch-size/LR co-scaling (prompted by a resource-utilization question mid-session)**: profiling (`nvidia-smi`) during the busiest 4-GPU-job stretch showed persistent 5-30% compute / 2-7% VRAM utilization even with 4-9 runs packed onto 4 GPUs simultaneously -- these ~0.7-1.6M-param models are nowhere near saturating an A5000 at `batch_size=64`. Per thinker-e9: don't reuse `lr=3e-4` (tuned for batch=64) at a larger batch without re-tuning -- same d_model/LR coupling lesson as Phase -1, apply the linear scaling rule (Goyal et al. 2017) as a starting point, then verify with a small sweep, don't trust it blindly. Sweep at `batch_size=256`, `n_facts=16`/`depth=3` (the Phase 0 go/no-go task): `lr=6e-4` (99.55% acc), **`lr=1.2e-3` = the linearly-scaled value (4x batch -> 4x lr): best, 99.9% acc**, `lr=2.4e-3` diverges (7.1% acc, loss stuck at 4.3). Confirms the linear scaling rule held exactly at this 4x batch jump. **`batch_size=256, lr=1.2e-3` is now the reference config for any future `learn/indexed_attention` run at this scale** wanting to use the node's idle capacity -- any accuracy/loss comparison against the earlier `batch_size=64, lr=3e-4` runs in this log should note the two changed together, not be read as an apples-to-apples curve comparison.

**Session-level summary of everything validated this session** (Phase 0 through here), for a reader who wants the punch line without the blow-by-blow above:
1. **Go/no-go (n_facts=16, 3 seeds)**: hierarchical (`depth>0`) massively beats flat attention (`depth=0`) -- 99.6-100% vs. 5.9-7.0%. Confirmed, not ambiguous.
2. **Curriculum on `n_facts`** (16->32->64) fully unblocks the `depth>0` mechanism at the `n_facts=64` scale that direct training could never solve (any `d_model`/`lr`/`n_step`/`n_register`/`batch_size` tried) -- 100% final accuracy. `depth=0` under the same curriculum never even clears `n_facts=16` (consistent negative control, not a new anomaly).
3. **Phase 1bis first pass** (`n_slots`, `use_ff`, `detach_sm_keys`, `level_dropout_p`) at `n_facts=16`: no variant distinguishable from baseline (all 98.9-100%) -- inconclusive at this scale, a real test needs either the `n_facts=64` curriculum-trained setting or a task that needs computation over retrieved values, not more distractors on an already-near-ceiling task.
4. **Multi-hop chain task**: mechanism handles 1 hop easily, plateaus hard at 2 hops (32-34%) regardless of fixed vs. randomized `N_step` -- open question, LR re-sweep at this task is the next diagnostic step, not yet a verdict on the architecture's multi-hop capability.
5. **Infra**: default to writing remote process output to a file on the node (not piping over the orchestrating SSH connection), and to reserving whole nodes (`gpu=4`) with `CUDA_VISIBLE_DEVICES` packing multiple independent runs per node rather than one GPU per job -- both adopted as standing defaults going forward given these models are small enough that node-level (not just GPU-level) parallelism is the actual bottleneck lever.


## 2026-09-13 — Distillation 500M-core MFU/batch-size sweep, and a missing-bf16 discovery (`abacus26` L40S, job 4104870)

Requested by thinker-e9 to replace the extrapolated MFU estimates in `learn/distill/README.md` with a real batch-size sweep at the 500M-core tier (810.8M total, `n_layer=25 n_embd=1280 n_head=16`). Found something more consequential than a batch-size curve: **`train_sft.py` had no mixed-precision at all** (`AutoModelForCausalLM.from_config` → fp32 by default, zero `autocast`/`GradScaler` in the file) — every MFU number in this project's history to date was an fp32-achieved-throughput compared against a **bf16** peak-FLOPS spec, not a like-for-like ratio.

Added `--bf16` to `train_sft.py` (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around the forward+loss computation; opt-in flag, no behavior change when unset). Results:
- fp32: batch=4 → 3,239 tok/s, batch=6 → 3,425 tok/s, **batch=8 OOMs** (44.39GB L40S already ~42GB used at batch=6).
- bf16: batch=6 → **5,748 tok/s — a real 1.68× speedup over fp32 at the identical batch size** (Tensor Core effect, confirms the missing-autocast finding was real and fixable). **batch=7 also OOMs under bf16.**
- **The batch ceiling (6) is identical in fp32 and bf16** — bf16 sped up compute but did not raise the memory ceiling, because the bottleneck is the Teacher-aligned tied vocab head (248,077 tokens): `topk_kd_loss`'s `logsumexp` over the full `(batch, block_size, vocab)` logits tensor, plus the Teacher's own fp32 top-K/residual tensors, dominate memory regardless of the model's own autocast dtype.

**Practical upshot**: `--bf16 --batch_size 6` (5,748 tok/s real) is now the best measured single-GPU config at this tier, giving ~392h (16.3 days) for `D=8.11B` tokens on one L40S — still past Grid'5000's ~1-week single-reservation limit, so checkpoint/resume across besteffort reservations (already implemented) remains necessary regardless. Full numbers and a chunked-loss idea (to remove the vocab-driven memory ceiling independently of precision, à la Liger-Kernel/"Cut Your Losses" — not yet implemented, next lever if more throughput is needed) are in `learn/distill/README.md`'s updated batch-size/precision sweep section.

## 2026-09-13 — n_hops=2 LR sweep: rules out miscalibrated LR as the plateau's cause

Follow-up to the multi-hop plateau found earlier (n_hops=2 stuck at 32-34% held-out acc, both fixed and randomized N_step=12), per thinker-e9's request: sweep LR specifically at `n_hops=2` (direct, no curriculum) rather than reusing the `n_hops=1`/single-lookup LR. `abacus3-1`, 4 GPU parallel, 4 min budget each, `n_distractors=2, vocab_size=64, depth=2, block_size=4, d_model=256, n_step=12, batch_size=64`:

| lr | final_acc |
|---|---|
| 1e-4 | 24.5% |
| 3e-4 | 25.3% |
| 1e-3 | 26.1% |
| 3e-3 | **diverges (loss NaN, 0% acc)** |

None of the tested LRs show a qualitatively different trajectory from the known 32-34% plateau (these are all in the same ballpark, plausibly still en route to that plateau within the short 4 min budget, not a real difference) — no LR value found so far unlocks mastery, and the highest value tested is unstable. **This weakens hypothesis (b) (miscalibrated LR)** as the explanation for the n_hops=2 plateau. Not a fully exhaustive sweep (4 min/point is short, and only 4 values tried), but no positive signal for "just needed a different LR" the way `n_facts=64`'s plateau turned out to be a curriculum problem rather than an LR one at first glance, or the way `n_facts=16`'s original failure *was* purely an LR problem. Remaining live hypotheses per the original three: (a) `N_step=12` still insufficient for 2-hop composition, or (c) a genuine mechanism limit on chaining through the SM buffer — next diagnostic step (not yet run) should isolate `N_step` directly (e.g. `N_step` sweep at a fixed, reasonable LR) before concluding on (c).

## 2026-09-13 — n_hops=2 LR sweep at full budget: (b) miscalibrated LR properly ruled out

Per thinker-e9's valid concern (the short 4-min sweep above wasn't budget-comparable to the 15-18min/16-22k-step runs that established the 32-34% plateau), re-ran the two most promising short-sweep LRs (`1e-3`, `6e-4`) at full budget (18 min, `abacus3-1`, job 4104890, `n_hops=2` direct, same config as the original plateau runs):

- `lr=6e-4`: 16,685 steps, **final_acc 25.2%** — still within/below the known plateau, no improvement.
- `lr=1e-3`: 16,984 steps, held-out acc oscillated 22-28% through training then **final measured acc 4.6%** (an unstable/collapsed final read, not a real improvement — consistent with `lr=1e-3` being the least stable value tried so far, one step from the `lr=3e-3` value that diverges outright).
- Extrapolation probe (`--extrapolate_n_steps 16,20,24`, in-memory, no checkpoint) on both: flat or slightly worse than the training-time N_step=12 accuracy (e.g. `lr=6e-4`: 24.8/25.0/24.5% at N_step_test=16/20/24 vs. 25.2% at N_step=12) — no sign that simply running more reasoning steps at inference recovers anything.

**Conclusion: (b) miscalibrated LR is now properly ruled out** at a budget comparable to the original plateau observation, not just a short sweep. Neither of the two candidate LRs exceeds 32-34%, and the higher one shows real instability rather than a hidden improvement. Remaining live hypotheses: (a) `N_step=12` still insufficient for 2-hop composition (next diagnostic: isolate `N_step` directly, e.g. a sweep at `N_step` ∈ {8, 16, 24, 32} at the already-known-stable `lr=3e-4`, matching Phase -1's methodology of sweeping one axis at a time rather than changing several together), or (c) a genuine mechanism limit on chaining through the SM buffer at 2+ hops -- not yet distinguishable from (a) without that N_step isolation.

## 2026-09-13 — n_hops=2 N_step sweep: (a) also ruled out, points to (c) a real mechanism limit

Isolating N_step directly at a fixed, stable LR (`lr=1.2e-3`, rescaled with `batch_size=256` per the linear rule to avoid reintroducing a batch/LR confound -- NOT the `lr=3e-4` originally suggested, since that was only validated at `batch_size=64`), `n_hops=2` direct, `N_step ∈ {8, 16, 24, 32}` × 2 seeds, 18 min budget each, 8 runs packed on 4 GPUs (2/GPU):

| N_step | seed 0 | seed 1 |
|---|---|---|
| 8 | 24.9% | 24.8% |
| 16 | 25.5% | 24.7% |
| 24 | 25.0% | 25.5% |
| 32 | 24.9% | 24.9% |

**Completely flat across a 4x range of N_step (8 to 32, i.e. 4x to 16x the hop count, well past thinker-e9's own ~2-4x heuristic and past the N_step=12 already tested)** -- all 8 runs land within a 0.8-point band (24.7-25.5%), no trend whatsoever. This rules out (a) N_step insufficiency as the explanation, following directly on ruling out (b) miscalibrated LR at full budget in the previous entry.

**With both (a) and (b) ruled out, (c) -- a genuine mechanism limit on 2+-hop composition through the SM buffer -- is now the best-supported explanation** for the n_hops=2 plateau (roughly 25-34% depending on the exact config tested across these sweeps, consistently well above chance ~1.5-3% but nowhere near the near-100% mastery seen at n_hops=1). This is a significant result for the project's central thesis (iterative extraction+processing should compose across hops) -- the mechanism handles single-hop retrieval essentially perfectly but does not yet compose reliably across two hops, independent of training budget, LR, or reasoning-step count tried so far.

**GPU utilization note** (per explicit user feedback on under-utilization mid-session): this sweep used `batch_size=256` (vs. the earlier default of 64) and packed 2 runs per GPU, measured at 31-41% compute / 8-14% VRAM per GPU during the run -- a real improvement over the 5-30%/2-7% seen in earlier single-run-per-GPU sweeps, though still with significant headroom (only 1-2.6GB of 24GB VRAM used per GPU). Density (processes/GPU) should be pushed further on the next batch of runs rather than batch size alone, per thinker-e9's guidance, to avoid re-opening the batch/LR confound question on an already-running sweep.

## 2026-09-13 — n_hops=2: use_ff and n_register don't clearly help either

Testing thinker-e9's priority-1 candidate (`use_ff=True`, the Phase 1bis variant Phase 1bis itself couldn't discriminate at n_facts=16) and n_register as candidate #2, at `n_hops=2` direct, `batch_size=256, lr=1.2e-3, n_step=12`, 2 seeds each, 18 min budget:

| Variant | seed 0 | seed 1 |
|---|---|---|
| `use_ff=True` | **diverges (NaN)** | 25.3% (= baseline) |
| `n_register=2` | 8.8% (worse) | 1.6% (much worse) |
| `n_register=4` | 25.0% (= baseline) | 25.3% (= baseline) |

None of these clearly break the ~25% plateau. `use_ff` is at best neutral (one seed matches baseline, the other diverges — plausibly an LR-stability interaction with the added FF capacity, not yet re-swept for this variant specifically) rather than a clean unlock. `n_register=2` is notably *worse* and unstable across seeds; `n_register=4` is neutral, same as baseline. No candidate tested so far (LR, N_step, use_ff, n_register) breaks the n_hops=2 plateau — (c) a genuine composition-mechanism limit remains the best-supported reading, though `use_ff`'s divergence at seed 0 leaves open whether a properly re-tuned LR for that variant specifically might behave differently (not yet tested: only the retrieval-task-tuned `lr=1.2e-3` was tried with `use_ff`).

## 13 Sep 2026 -- Teacher-target precompute sharding: A40/A100 (Ampere) are a bad fit for the FP8 Teacher checkpoint

Per the user's "as fast as possible" directive, sharded the 8000-example Teacher-target precompute (`precompute_teacher_targets.py`, K=32, max_length=1024) across 3 independent GPU jobs instead of running it sequentially on one node. Measured per-node throughput surfaced a real (not incidental) hardware-fit issue:

| Node (GPU) | Architecture | ex/s | Notes |
|---|---|---|---|
| abacus26 (L40S) | Ada Lovelace | 2.84-2.86 | native FP8 tensor cores |
| abacus27 (H100 NVL) | Hopper | 3.2-3.5 | native FP8 tensor cores |
| abacus4 (A40) | Ampere | 0.24 | **no native FP8 tensor cores** |

abacus4's shard was ~12x slower than the other two despite similar GPU memory headroom (58% util, only 126W draw on a ~300W TDP card -- clearly not compute-bound in the normal sense). Root cause: the Qwen3.8-27B-FP8 Teacher checkpoint is natively FP8-quantized; Ada (L40S) and Hopper (H100) have hardware FP8 Tensor Core support, Ampere (A40, and presumably A100) does not, so the `kernels` package's fine-grained FP8 path falls back to a much slower dequant/compute path on Ampere. This is a distinct failure mode from the earlier-documented "forcing `--dtype bfloat16` on a <56GB-VRAM GPU triggers CPU offload" collapse (both were previously conflated as "some GPUs are just slow for this") -- here `--dtype auto` was used correctly, and the model fit in VRAM without offload; the slowdown is purely an architecture/FP8-kernel-support mismatch.

### Full train/val curve, KD-run 500M-core (job 4105629, completed cleanly)

Completed all 13340 steps (`training_seconds=7669.2`, ~2h08, `best_loss=0.1227`), this time with the full periodic val curve preserved (`python -u` fix). Key points (step: val_ce / val_kd):

| step | val_ce | val_kd |
|---|---|---|
| 1 | 12.107 | 0.400 |
| 500 | 0.3145 | 0.671 |
| 1000 | 0.1909 | 0.706 |
| 2000 | 0.1315 | 0.732 |
| 3000 | 0.1134 | 0.750 |
| 5000 | 0.1004 | 0.778 |
| 8000 | 0.1062 | 0.786 |
| 10000 | 0.1038 | 0.802 |
| 13000 | 0.0974 (min) | 0.815 (max) |

**val_kd diverges almost immediately** (0.40->0.67 by step 500 alone, ~75% of its total eventual rise happens by step 2000-3000) and keeps climbing slowly and almost monotonically for the entire 13340-step run, never plateauing. **val_ce shows no comparable divergence** -- it oscillates in a noisy 0.10-0.13 band from step ~2500 onward, with its best value at the very last measured point (step 13000). Conclusion for model-design's question: KD-term memorization starts near-instantly and never stops climbing at this data scale (8000 examples); CE-based language-modeling generalization is unaffected across the whole run. Their suggested follow-up (try a lower `kd_alpha`, e.g. 0.1-0.2, to see if de-weighting the KD term changes the overall val_loss picture) is a reasonable next step, not yet run.

### GPU-scale attn_supervised grid (6 runs: 3 baseline, 3 attn_supervised) -- self-match fixed, task accuracy not

model-design's attention-supervision fix (auxiliary CE loss on q_proj/k_proj, no new params) was tested at `d_model=128` (vs. their CPU-scale `d_model=32` test) across 3 seeds each:

| variant | seed | final_acc | mean_rank (chance=1.50) | top1_rate (chance=0.25) |
|---|---|---|---|---|
| baseline | 0 | 0.246 | 1.250 | 0.336 |
| baseline | 1 | 0.264 | 1.264 | 0.401 |
| baseline | 2 | 0.256 | 1.590 | 0.272 |
| attn_supervised | 0 | 0.247 | **0.000** | **1.000** |
| attn_supervised | 1 | 0.245 | **0.000** | **1.000** |
| attn_supervised | 2 | 0.257 | **0.000** | **1.000** |

**Striking disconnect**: the self-match diagnostic goes from noisy/near-chance (baseline) to *perfect* (mean_rank=0, top1=100%, all 3 seeds) under attention supervision -- the auxiliary loss completely fixes the mechanistic problem it targets. But `final_acc` on the actual n_hops=2 chain task is essentially unchanged (baseline avg ~0.255, attn_supervised avg ~0.250) -- no better than the ~25-32% plateau documented throughout this project. This GPU-scale result (larger d_model, longer budget than model-design's CPU smoke test) does not reproduce their reported 72.9% accuracy at CPU scale -- a real discrepancy to flag, not just noise, since the self-match fix landed perfectly across all 3 seeds while accuracy stayed flat. Possible reading: perfect self-match among an episode's *own* candidate facts is necessary but not sufficient for the downstream task -- something else in the SM->output path (per model-design's own earlier hypothesis) may be the actual bottleneck once retrieval itself is no longer the failure mode.

**Actionable conclusion**: never schedule the Teacher-FP8 precompute (or presumably any FP8-checkpoint inference) on Ampere-generation GPUs (A40, A100) at this cluster -- restrict to Ada/Hopper (L40S, H100) or newer. The abacus4 job was killed mid-shard (besteffort preemption actually beat us to it) and its ~2520 remaining examples were re-split across the two already-idle Ada/Hopper nodes instead, which finished in ~8 additional minutes.

Follow-up: launched the real KD training run (`learn/distill/train_sft.py`, 500M-core tier, `--bf16 --mup --kd_alpha 0.5`, merged 8000-example Top-K32 Teacher targets) on the H100 node (fastest available at Rennes for this workload), 10 epochs (13,340 steps) budgeted at ~2h based on measured throughput, checkpointing every 500 steps to survive besteffort preemption.

### Real KD run result (500M-core, 8000 real examples, H100)

Completed cleanly, no preemption: **13,340/13,340 steps, `best_loss=0.1267`, `training_seconds=3110.6` (~51.8 min)** -- almost 2.5x faster than the ~2h05 estimate extrapolated from L40S throughput (0.552 s/step there vs. ~0.233 s/step actually achieved on the H100 NVL, a bigger gap than the ~15-20% suggested by the earlier precompute ex/s comparison -- KD training's compute mix, unlike single-example precompute inference, apparently favors H100 more strongly, plausibly batching/kernel-fusion effects rather than raw FP8 throughput alone).

Loss trajectory: 6.41 (step 1) -> 1.21 (step 95) -> ~0.15-0.18 (plateauing from roughly step 9000 onward, oscillating in that band through step 13340). Combined CE and KD components both bottomed out in the same range (`ce` ~0.10-0.15, `kd` ~0.17-0.22 at the end).

**Caveat worth flagging to model-design**: with only 8000 training examples and 13,340 steps at batch_size=6 (~10 full epochs), a loss collapse from 6.4 to ~0.15 is consistent with memorization/overfitting on this small a sample, not necessarily a generalizable KD signal -- the run validates the training *pipeline* (real data, real Teacher targets, checkpoint/resume, bf16, muP) end-to-end at this scale, but the loss curve itself shouldn't be read as "KD works well at 500M-core" without a held-out eval or a larger example count to rule out memorization.

### Held-out val check confirms memorization on the KD term

Per model-design's suggestion, ran the trained checkpoint against a held-out val split (from `prepare_reasoning_data.py`'s own `val.jsonl`, never seen in training -- distinct from `train_sample8000.jsonl`) using the same CE+KD loss (`learn/distill/eval_val_loss.py`, a new small script that reconstructs the exact architecture from the checkpoint's saved args/muP multipliers and runs a no-grad pass). First pass used a 40-example val slice (`val_sample40.jsonl`) whose Top-K32 Teacher targets happened to already exist from an earlier bf16-vs-fp8 precompute-dtype sweep session, letting this check run **without any GPU at all** (the eval only needs the small 810M student + precomputed targets, not the 27B Teacher -- ran on a plain CPU besteffort-free job while the three Ada/Hopper GPUs were all tied up by other users' jobs, see `grid5000_usage.log.md`).

Result: **val_loss=0.4585** vs. **train best_loss=0.1267** (~3.6x gap). Breaking down the two components separately is informative: `val_ce=0.0985` is actually in the same range as train's CE component (~0.10-0.15) -- plain next-token prediction generalizes fine -- but `val_kd=0.8185` is roughly 4-8x every train-time KD value logged (~0.10-0.22 range). **Conclusion: the loss collapse is memorization specifically of the fine-grained Teacher-logit alignment (the KD term), not of the underlying language-modeling task.** This matches the earlier caveat's prediction and settles the train/val question model-design asked for -- more examples (not just more steps) are needed before this run's loss curve says anything about real KD quality at this scale.

Caveat on this specific check: n=40 is a small val slice (chosen only because its Teacher targets already existed from an unrelated earlier sweep, avoiding a GPU-contended precompute just to get a first read); the qualitative CE-vs-KD split is unlikely to flip with more examples, but a tighter quantitative val_loss estimate would use a larger held-out slice (a 1000-example `val_sample1000.jsonl` is already prepared and staged for this, precompute pending GPU availability).

### Correction: "avoid Ampere for Teacher-FP8 precompute" was too broad -- the real constraint is VRAM, not architecture generation

Session-13's earlier `experiment.log.md` entry ("Teacher-target precompute sharding: A40/A100 are a bad fit for the FP8 Teacher checkpoint") concluded from the abacus4 (A40, 46GB) result alone that Ampere-generation GPUs should be avoided entirely for this workload. Investigating a Nantes site standby reservation (see `grid5000_usage.log.md`) turned up pre-existing logs from an earlier session's FP8-vs-bf16 comparison work on an **A100 80GB** (`ecotaxe` cluster) that contradict the blanket claim.

`transformers` itself explains the real mechanism on load: *"FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100) ... We will default to dequantizing the model to bf16"* -- A100 is compute capability 8.0, so it always dequantizes FP8->bf16 on load, exactly like A40. The dequantized model needs ~55.6GB VRAM (vs. ~30.9GB native FP8). **A40 (46GB) doesn't have enough VRAM for that, so it silently falls back to CPU offload -- a ~50-100x collapse, which is what the earlier 0.24 ex/s number actually measured.** A100 80GB has plenty of headroom for the same 55.6GB dequantized model, so no offload happens: `precompute_fp8_fixed.log` from that Nantes session shows the checkpoint loading in 12.6s and reaching a **steady-state throughput of ~3.1-3.6 ex/s** -- essentially on par with L40S (2.84-2.86 ex/s) and close to H100 (3.2-3.5 ex/s), not 12x slower.

**Corrected rule**: the deciding factor for this Teacher checkpoint's precompute speed is **available GPU VRAM relative to the ~56GB bf16-dequantized footprint**, not "Ampere vs. Hopper/Ada" as a category. A100-80GB (and presumably any other >=64GB-class Ampere card) is a fine precompute target; A40 (46GB) and any other <56GB card outside the native-FP8 Ada/Hopper set are not. Told model-design about this correction since the earlier (too-broad) version had already been passed along.

## 2026-09-13 — Contre-expertise: the n_hops>=2 plateau was a compressor bug, not a mechanism limit

Independent review of the whole Indexed Attention branch (spec + plan + this log + `core/` + diagnostics), requested by the user. It overturns the branch's current headline conclusion. **Read this entry before acting on any earlier multi-hop conclusion in this file.**

### 1. The plateau was compared against the wrong chance level

Every earlier entry reads the plateau as "well above chance (~1.5-3%), so the mechanism partially composes". That reference is the uniform-over-vocabulary rate, and the model never chooses among the vocabulary — it copies a value present in the episode's KB. The correct reference is the **conditional** chance level `1/n_facts`:

| Config | `n_facts` | `1/n_facts` | Plateau observed earlier |
|---|---|---|---|
| `n_hops=2, n_distractors=1` | 3 | 33.3% | 32-34% |
| `n_hops=2, n_distractors=2` | 4 | 25.0% | 24.7-25.5% |

Two exact matches on two different configs. Measured directly: **97-98% of the baseline's predictions land on some KB value**. The model had learned "emit a KB value" and was picking at random among them — a total failure, not partial composition.

Worse, the trivial-predictor controls now implemented (`learn/indexed_attention/eval_metrics.py`) show the plateau was **below** the best no-retrieval shortcut:

| Config | `random_kb` | `non_key` (skips every hop) | model at plateau |
|---|---|---|---|
| `n_hops=2, n_distractors=1` | 0.332 | **0.500** | 0.32-0.34 |
| `n_hops=2, n_distractors=2` | 0.253 | **0.330** | 0.247-0.255 |
| `n_hops=3, n_distractors=1` | 0.244 | **0.492** | 0.42-0.46 |

`non_key` exploits a real structural shortcut in `data/kb_chain_retrieval.py`: the chain's final answer never appears as a key, so guessing uniformly among non-key values needs zero hops. The `n_hops=3` baseline's "42-46%" was exactly this shortcut, not partial chaining.

### 2. Root cause: `LevelCompressor` could not represent a key->value association

`core/indexed_memory.py::LevelCompressor` pooled `parent_k` and `parent_v` with the **same** softmax weights. A fact block is `[KEY_MARK, key_id, VAL_MARK, val_id]`; to serve as a memory entry a node must be *findable by its key* (`parent_k ~ f(key_id)`) and *return its value* (`parent_v ~ g(val_id)`) — two opposite weightings over the same children. With one softmax the compressor can only pick one, or settle on a blurred compromise: brute-forceable at one hop (hence the clean ~98% at `n_hops=1`), unchainable beyond.

**Fix**: `decouple_kv=True` (now the default) adds a second learned pooling query `query_v`, costing `n_slots * d_model` parameters. `decouple_kv=False` keeps the old behavior as an ablation (`--shared_kv_pooling`).

**CPU evidence** (`d_model=32`, `n_step=8`, 3000 steps, `lr=1e-3`, batch 64, 2 seeds, everything else identical):

| Variant | seed 0 | seed 1 |
|---|---|---|
| shared pooling, `n_hops=2` | 35.4% | 26.4% |
| **decoupled, `n_hops=2`** | **100.0%** (loss 0.000) | **100.0%** (loss 0.000) |
| shared pooling, `n_hops=3` | 46.2% | 42.4% |
| **decoupled, `n_hops=3`** | **98.1%** | 46.4% |

Clean and reproducible at 2 hops. At 3 hops one seed out of two solves it at this budget — real progress over a baseline that never does, but **not yet a stable result**; seed variance at 3 hops is the first thing to characterize on GPU.

**Therefore hypothesis (c) ("a genuine mechanism limit on 2+-hop composition") is refuted.** The loop composes; the compressor could not supply anything composable. The `(a)`/`(b)` eliminations (N_step, LR) remain valid work but were answering a question whose premise was wrong.

### 3. The attention supervision was optimizing an orthogonal objective

This explains the "striking disconnect" logged above (perfect self-match, flat accuracy). `candidate_match_loss` supervised the query toward the fact's **KEY leaf** — but attending to a key leaf returns `v_proj` of that same key token, i.e. what the model already had. The auxiliary objective was fully satisfiable *and* useless: hence `mean_rank=0.000 / top1=1.000` on all 3 seeds with accuracy unchanged. Only a fact's **level-1 node** carries the key->value pair.

Both supervision scripts now default to `--supervise node` (targets `mem._levels_k[1]` at the target fact's index, gradient also reaching the compressor's pooling queries); `--supervise leaf` reproduces the old grid.

**Smoke observation, not an experiment** (CPU, 434 steps, 15s, `d_model=32`, `n_hops=2`): `--attn_supervised --supervise node` with decoupled pooling reached `final_acc=0.72` — versus a 25% plateau after 16,000+ GPU steps previously. Needs a real run at budget before being quoted as a result.

### 4. Phase 0's "hierarchy vs flat" result needs re-reading

`depth=0` (the "flat Baseline C") has no compressor at all, so its leaves are per-*token* K/V: `k_proj(embed(tok))` / `v_proj(embed(tok))`. Attending to a key token returns that key token. **A flat memory of raw leaves cannot represent a key->value association under any training budget** — which is why it sat at 5-7% and never left curriculum stage 1.

So the 99.6% vs 6.0% gap does **not** establish that hierarchical indexing beats flat attention. It establishes that block-level grouping is the only path to an associative entry in this implementation. The honest flat baseline is **`depth=1`** (one compression level, one node per fact, no multi-level index) — see the plan's Phase 0bis.

### 5. What changed in the repo

- `core/indexed_memory.py`: `LevelCompressor(decouple_kv=True)` default + `_pool()` helper; threaded through `HierarchicalMemory` and `Thinker`.
- `tests/test_indexed_memory.py`: `TestDecoupledKVPooling` (9 tests) pinning the property the old code violated; `test_compressor_matches_manual_reference` now parameterized over both modes. 70 tests green.
- `learn/indexed_attention/eval_metrics.py` (new): conditional chance, `pred_in_kb_rate`, trivial-predictor controls, `format_report`. Wired into `train_kb_chain.py` and `train_kb_retrieval.py` — every run now prints the chance-level block.
- `diagnose_attention_supervision.py` / `train_kb_chain_attn_supervised.py`: `--supervise node|leaf`, `--shared_kv_pooling`, plus a `node_selection_diagnostic` that tracks what actually matters.

No GPU runs were launched for this entry — the re-runs are queued in the plan (Phase 0bis / Phase 2-redo).

## 2026-09-13 — CPU-vs-GPU confound resolved: exact-config repro confirms it was hyperparameters, not scale (superseded by the compressor-bug finding above, kept for the record)

Before the Contre-expertise entry above landed, this session (experiment-manager) had been given a narrower, now-superseded task: the earlier GPU `attn_supervised` grid (`d_model=128`, 3 seeds) had failed to reproduce the CPU diagnostic's 99.2% task accuracy (self-match fixed perfectly on both, but GPU `final_acc` stayed at the ~25% plateau) — plan flagged reproducing the exact CPU config (`d_model=32, vocab_size=32, n_register=4, batch_size=64, lr=3e-4, n_step=16, 12000 steps`) at matched budget on GPU as the required next step before trusting either a scale effect or chasing a new architectural hypothesis.

Ran `learn/indexed_attention/train_kb_chain_attn_supervised.py` with that exact config, 3 seeds, on Rennes (`abacus22-1` A5000 job 4105879, then `abacus21-1` A100 job 4105917 after the first job's walltime cut off mid-grid):

| variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `attn_supervised` | 99.77% | 99.77% | 99.22% |
| baseline (no supervision) | 33.8% | 26.9% | 28.8% |

**Result: GPU fully reproduces the CPU numbers when the config is matched exactly** (99.2-99.8% vs. CPU's 99.2%; baseline lands in the same 25-34% plateau band documented throughout this project). This resolves the specific confusion flagged in the plan: the earlier `d_model=128` grid's failure to reproduce was **not** a d_model/GPU-scale effect — it was a confound of several simultaneously-changed hyperparameters (the GPU script's own defaults: `vocab_size=64` not 32, `n_register=1` not 4, `batch_size=256` not 64, `lr=1.2e-3` not 3e-4, `max_steps` capped by a 30-min walltime not a fixed 12000). Each run took ~25 min on GPU (not the few seconds this scale suggested — small-model wall-clock is dominated by per-step Python/kernel-launch overhead, not FLOPs), which is why the first job's default 1h walltime cut the grid short mid-way and needed a second, longer-walltime job to finish.

**Superseded context**: by the time this finished, the Contre-expertise entry above (relayed via the `model-design` sister session) had already found the real root cause of the `n_hops>=2` plateau (the compressor's shared-softmax key/value pooling bug) and shown the attention-supervision fix itself was targeting the wrong node (leaf, not level-1) — so this result closes the specific "why doesn't GPU reproduce CPU" question cleanly, but the broader `attn_supervised`-at-`d_model=128`-plateau finding it was chasing is no longer the live question. No further action needed on this thread; recorded for completeness since it was a fully-executed, valid diagnostic in its own right.

Infra note: `micromamba` isn't on `$PATH` in a non-interactive `oarsh`/`nohup` shell (no `.bashrc` sourcing) — use the full path (`~/micromamba/micromamba run -n <env> ...`) in any unattended launch script rather than assuming `micromamba` resolves. Also, running a script by relative path from `~/thinker` still needs `PYTHONPATH=~/thinker` set explicitly — Python puts the *script's own* directory (`learn/indexed_attention/`) on `sys.path[0]`, not the cwd, so `import data.foo` / `import core.foo` fail with `ModuleNotFoundError` otherwise despite `cd`ing to the repo root first.

## 2026-09-13 — Toy model: the on-the-fly medium-term memory was never actually tested

Same blind spot as the compressor bug found the same day: the mechanism exists in the code, nothing ever forces its use, and no metric in place would have revealed it.

`core/toy_model.py:205-207` has the machinery (`memory = latents if i >= read_step else [x] + latents`, plus an `n_memory` FIFO cap), but `read_step = n_step - 1` is **hardcoded** in every training script (`scripts/train.py:77`, `scripts/th1nker_runner.py:1044` — whose comment, "remove on output step", confirms the intent was to force only the OUTPUT to read from memory), and `n_memory` was never set (default `1e4`, unbounded). Neither knob appears anywhere in this log.

Tracing the loop: the input stays in memory for every latent-compute step; only the final output query reads `latents` alone. **The model never had to carry information across iterations without being able to re-read the input.**

Consequence: the known results (97% base-16 addition, Dec 2023; the `cumsum` compute-extrapolation, Sept) are fully compatible with "the model re-reads the input each step and the latent is just a workspace". They neither demonstrate nor refute that the on-the-fly memory carries information — the question is simply open. The Sept compute-extrapolation result itself still stands; it does not depend on this point.

Caveat to lift before investing: this comes from reading the code and tracing the loop, not from an execution. The Sept run may have come from `notebooks/Th1nker_runner.ipynb` rather than the two scripts checked. Verify which path that run used.

Full analysis, the decisive experiment (a `read_step` sweep) and four further proposals: **`dev_notes/toy_model_memory_experiments.md`**. Nothing implemented, no runs launched; briefed to `model-design` for implementation and `experiment-manager` for execution.

Why it matters beyond the toy branch: `Thinker` has the same structure (SM built by APPEND per step while the KB stays permanently queryable — i.e. `read_step = n_step` throughout), so this conditions the SM's design in the main architecture, at CPU cost instead of GPU.

## 2026-09-13 — `read_step` sweep executed: [RETRACTED — see next entry] the on-the-fly memory does carry real information, no cliff found anywhere

Ran the decisive experiment flagged above. Two corrections landed from `thinker-5b`/`model-design` before/during execution, both worth keeping in mind when reading the result: (1) `read_step` alone isn't the only varying quantity — the real write budget before the input disappears is `(read_step+1) * n_latent` vectors, so a naive low-`read_step` failure could be an information-theoretic impossibility rather than a verdict on the memory mechanism (`capacity_budget()` in `learn/toy_memory/eval_metrics.py` checks and flags this — `capacity_constraining` was `False` on every single run below, so this was never actually binding at these settings); (2) any direct-training cliff must be re-tested with `--read_step_curriculum` before concluding a mechanism limit, since this project has twice already seen a plateau that looked architectural dissolve under curriculum alone (18 Dec 2023 copy-task; `n_facts` Phase -1/0 this same week).

**Mandatory sanity check first** (per protocol): `--task copy --read_step 6 --n_step 6` (input always visible, the trivial case) — reached 100% exact-match by step 200, confirming the pipeline itself is sound before reading anything into a low-`read_step` result.

**Full grid: `{copy, cumsum}` x `{seq_len=8/n_latent=8, seq_len=32/n_latent=32}` x `read_step` in `{0..6}` x 3 seeds (84 runs), plus 12 extrapolation runs (trained at `read_step=6`, evaluated at `read_step=0`)** — 96 runs total, `d_model=64`, `n_step=6` fixed. Run on Rennes `paradoxe-5`, CPU (52 cores, `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1` per worker, 48 concurrent workers), normal (non-besteffort) queue — moved here mid-session from an initial GPU launch on `abacus21-1` per `dev_notes/compute_scheduling.md`'s Tier A/C split (~100K-param toy models don't need a GPU and shouldn't occupy one reserved for distillation).

**Result: 96/96 runs reached exact_match=1.0000. Zero cliff at any `read_step`, any scale, either task, or the extrapolation condition.** `seq_len=8` (the original scale) turned out to be a floor case per `model-design`'s own calibration — copying 8 tokens is achievable in a single compute step regardless of memory quality, so it doesn't discriminate the question at all; `seq_len=32` is the real test and it's just as clean: `copy` at `read_step=0`/`seq_len=32` (the hardest transport-only condition — input excluded from memory after the very first compute step) still hits 100%, and so does `cumsum` at the same setting (transport **and** running-sum accumulation, the harder task). The extrapolation condition (train with the input always visible, evaluate with it withdrawn immediately) also reached 100% on every seed — the representation generalizes to a read_step regime never seen during training. No cell anywhere showed a cliff, so the curriculum-retest branch of the protocol was never needed.

**Reading this correctly** (per `model-design`'s own framing, worth restating so it isn't over- or under-claimed): this directly falsifies the "the latent is just a scratchpad, the model always re-reads `x`" null hypothesis that motivated the whole experiment — with `x` structurally unavailable after step 0, the only way to reach 100% is to have written a usable representation of it into the FIFO-appended latents before it disappeared, and for `cumsum` that representation has to support a running computation, not just a stored copy. That's a real, clean, positive result for `core/toy_model.py`'s on-the-fly memory mechanism at this scale (`n_latent=8/32`, `n_step=6`, vocab_size=16). It does **not** by itself say anything about `Thinker`'s SM (different architecture, different scale) — it closes the toy-model-specific blind spot the `n_hops>=2` investigation's sibling finding opened, nothing more.

Infra notes: (1) besteffort GPU reservations on this cluster keep getting preempted mid-grid with zero warning (hit again this same session on the unrelated Phase 0bis nf256 rerun) — a CPU-only workload that doesn't need a GPU is strictly better served by the normal queue once that's an option, not just cheaper. (2) A background driver script killed with `pkill -f <script.sh>` does NOT stop its already-forked child process, and once the driver is dead no further queued commands in its sequential loop get launched — confirming a lingering single run is running orphaned (reparented to PID 1) rather than assuming the whole batch is still progressing; kill both the driver and the live child explicitly.

## 2026-09-13 — RETRACTION: the 96/96=100% result above was a label-leakage artifact, not a memory result

`model-design`/`thinker-5b` found and fixed a real bug in `core/toy_model.py::ToyThinker.forward` right after the entry above was written: the output query is built as `embd_out_pos(pos) + embd_vocab(target)` with `target` **unshifted** — position *i*'s query already contains the embedding of `target[i]`, the exact token being predicted at that position. The pre-norm residual stack (`attn_compute`) lets that embedding survive untouched to the output, and the tied head (`F.linear(output, embd_vocab.weight)`) reads it straight back out — **completely independent of `memory`/`x`/`read_step`**. Confirmed directly: a trained model still predicts `targets` perfectly even when `inputs` is swapped for an unrelated random sequence at eval time. This is exactly why all 96 cells converged to 1.0000 with zero exceptions regardless of `read_step` — the grid was measuring the shortcut, not the memory.

**Every number in the entry above is void.** `core/toy_model.py`'s on-the-fly memory question is still open — back to where `dev_notes/toy_model_memory_experiments.md`'s original write-up left it, none of this session's 96 runs said anything valid about it either way.

Fix (`learn/toy_memory/train_toy_memory.py`, `core/toy_model.py`): standard teacher-forcing shift — position *i*'s query now gets `target[i-1]`, a reserved BOS id at position 0, model built with `vocab_size+1`. `evaluate()` now runs a mandatory leak check every time (inputs swapped for an unrelated random sequence, same targets — must fall back to chance) and prints/reports it (`leak_check`) so this exact failure mode can never again pass silently. Verified the fix doesn't break the learnable task: `read_step=6` (control), `seq_len=8`, 4000 steps converges cleanly to 100% by ~step 1000 with `leak_check` pinned at chance (~0.062-0.064) throughout — the task is still learnable, it just needed real budget now that the free shortcut is gone (the previous grid's likely-much-shorter effective budget won't necessarily be enough post-fix).

**Redo required, not yet done**: the full 96-cell grid needs re-running with the fixed script and a real per-cell budget (>=1000-2000 steps, not whatever the previous grid used), checking `leak_check` in every single output before reporting any number. Per explicit instruction, Exp. 2/3 stay paused until Exp. 1 gives a clean signal on the corrected measurement path. See `dev_notes/grid5000_usage.log.md` for the relaunch.

## 2026-09-14 — Redo complete, on the corrected (leak-free) measurement path: `copy` carries real information at scale, `cumsum` needs more capacity than this model has

Multi-round redo after the retraction above, with two more real bugs found and fixed along the way (both on `thinker-5b`/`model-design`'s side, in `train_toy_memory.py`) — worth recording since they explain why the numbers below took several iterations to become trustworthy:

1. **`seq_len=32` under-budgeted at first**: a flat `--max_time_minutes`/`--max_steps` budget shared across all 4 curriculum stages let the cheap early stages (seq_len 8/16/24) eat the time meant for the one stage that matters (32) — some cells never got there at all. Fixed with `--final_stage_min_steps` (initially only guaranteeing the *final* stage, later extended to guarantee it at *every* stage after a second round of the same failure mode at intermediate stages).
2. **`n_latent` mismatch between an isolated LR-sweep diagnostic and the real curriculum grid**: a quick LR sweep for `cumsum` used the script's own default `n_latent=8`, but the actual curriculum grid runs at `n_latent=32` throughout (including its `seq_len=8` first stage) — two different configs despite sharing "`seq_len=8`" in both. Caused an initial, wrong conclusion ("lr=1e-3 already works fine for cumsum") that a same-day re-sweep with the correct `n_latent=32` overturned (`lr=1e-3` gives 0-36% at `n_latent=32`, wildly unstable; `lr=3e-3` gives a clean 96-100% across 3 seeds).

**Final numbers, `leak_check` verified clean (~0.06, at `vocab_chance`) on every single cell reported below, no exceptions:**

- **`copy`, both scales (`seq_len` 8 and 32), all `read_step` 0-6, both direct and `--read_step=6`-trained/`eval_read_step=0` extrapolation**: converges cleanly and reliably, including the hardest condition (`read_step=0`, input excluded from memory after the very first compute step) at the harder scale. A handful of individual cells needed a much longer budget (up to 90 min instead of 30) to actually promote through the full `seq_len` curriculum to stage 4 before converging (confirmed via `final_stage_seq_len` in each log, not just trusting the reported number) — once genuinely evaluated at `seq_len=32`, `copy` reaches ~90-100% on the overwhelming majority of cells. **This is the clean positive result for the on-the-fly memory's core claim**: with `x` unavailable after the first step, the only way to reach these numbers is a real, FIFO-written, correctly-held representation of the input — not a scratchpad illusion, and not scale-limited within what was tested.
- **`cumsum` (transport + running-sum + modulo)**: `seq_len=8` converges well once the right LR is used (`lr=1e-3` sufficed there, before the `n_latent` confusion above). At `seq_len=32`, extensive isolation work (per-stage budget ruled out, LR re-swept and confirmed at the correct `n_latent`, curriculum-vs-direct-training both tried) still leaves near-total failure: only 3/24 curriculum cells ever reached the target scale, and even those landed at 0-4.6% exact-match; a direct (no-curriculum) run at the same corrected LR did better on 1/3 seeds (65.6%) but the other two failed differently, not identically — one (seed1) made real progress (91% token_acc by step 2800) then **diverged** (loss spiking to 19711+), the other (seed2) stayed flat/at-chance the whole run. **Mixed picture, not a clean uniform wall**: the diverging seed looks like `lr=3e-3` (stable at `seq_len=8`) being too aggressive specifically at `seq_len=32` (longer sequence, more accumulated gradient signal per step) rather than a capacity ceiling per se, while the flat seed looks more like a genuine ceiling. Per `model-design`: don't flatten this into "cumsum needs more capacity" as an established fact — a concrete next lever (lower LR or a warmup schedule for longer sequences specifically) is identified but not yet tried, left open for a future session rather than resolved tonight.

**Net conclusion for the toy-model memory question this whole thread was chasing**: `core/toy_model.py`'s on-the-fly medium-term memory does carry real, usable information across compute steps — `copy` proves transport works, cleanly, at scale. Whether it can also carry information *useful for further computation* (not just storage/retrieval) remains open at `seq_len=32` — `cumsum`'s failure there is a mixed signal (one seed diverged after real progress, suggesting an LR/schedule issue specific to longer sequences; another stayed flat, suggesting a real ceiling), not yet resolved into a single clean explanation. Concrete next lever identified (lower LR or warmup for longer sequences) but not tried tonight — reprioritized back to Indexed Attention per explicit steer, `model-design`'s call on whether/when to revisit.

Infra lessons worth keeping for next time this project runs a curriculum-based toy experiment: (1) always verify `final_stage_*` (or equivalent "did it actually get where the cell name claims" field) in the log before trusting any number from a curriculum run, not just the reported final metric; (2) an isolated hyperparameter-diagnostic run and the real grid it's meant to inform must match on *every* config flag, not just the ones that seem relevant (`n_latent` here) — a name like "seq_len=8" is not a complete config description on its own; (3) besteffort GPU/CPU-normal-queue preemption hit this thread repeatedly (jobs `4106143`, `4106188`, `4106211` all ended mid-grid) — always check `final_stage`/completion markers per cell after any relaunch, don't assume a job's own `Terminated` (not `Error`) state means all its work finished.

## 2026-09-14 — Toy-memory Exp. 2: `n_memory` ablation finds no accumulation signal

`--task copy --read_step 0 --n_memory {1,2,4,8,10000}`, `seq_len=8/n_latent=8/lr=1e-3` (the validated Exp.1 scale, deliberately not `seq_len=32` which is still being characterized separately), 3 seeds/cell, `leak_check` clean (near vocab_chance) on all 15 cells, no exceptions:

| n_memory | seed0 | seed1 | seed2 |
|---|---|---|---|
| 1 | 98.1% | 99.9% | 94.5% |
| 2 | 67.2% | 100% | 96.4% |
| 4 | 97.2% | 100% | 100% |
| 8 | 100% | 99.7% | 94.1% |
| 10000 | 100% | 99.7% | 94.1% |

**`n_memory=1` already matches `n_memory=10000`** — all five values land in the same 94-100% band (the one low point, 67.2% at `n_memory=2`/seed0, looks like ordinary seed noise given `n_memory=1` and `n_memory=8` show no comparable dip on any seed, not a capacity effect). **No accumulation-across-write-slots signal detected at this scale**: whatever `copy` at `read_step=0` needs from the FIFO memory, a single most-recent slot already provides it just as well as ten thousand. This doesn't contradict Exp.1's `copy` result (real transport through the memory, confirmed there) but it does mean the specific "accumulates useful content across multiple write slots" sub-claim isn't supported by this test — at this scale, the mechanism is behaving more like a single recurrent state than a genuine multi-slot memory. Caveat: only tested with `copy` (a pure-transport task) at `seq_len=8` (the floor/easy scale per Exp.1's own calibration) — an accumulation signal might still be hiding behind a task/scale too easy to need more than one slot; not chased further this session, `model-design`'s call on whether/how to follow up (e.g. Exp.3 causal attribution, or a harder task/larger scale first).

## 2026-09-14 — Étape 4 (hardened generator) complete: real multi-hop chaining confirmed at n_hops 2/3/4

`train_kb_chain.py`, hardened generator (`n_distractors` scaled per `n_hops`: 6/5/4), `depth=2, block_size=4, d_model=256, n_step=12, batch_size=256`, 3 seeds/cell, `leak_check`/shortcut controls verified on every cell:

**LR sweep at `n_hops=2`** (16 cells: `lr` ∈ {1e-4, 3e-4, 6e-4, 1.2e-3, 3e-3} × 3 seeds, minus one):

| lr | seed0 | seed1 | seed2 |
|---|---|---|---|
| 1e-4 | 100% / +0.871 | 100% / +0.868 | 100% / +0.865 |
| 3e-4 | 99.98% / +0.865 | 100% / +0.850 (dup: +0.868) | 100% / +0.850 |
| 6e-4 | fails, ~0% margin | fails | fails, ~0% margin |
| 1.2e-3 | fails (~13-20%, margin negative) | fails | fails |
| 3e-3 | fails (6.2%, margin -0.084) | fails (4.0%, margin -0.104) | (not run) |

**Sharp, non-monotonic LR window**: only `lr ∈ {1e-4, 3e-4}` work (both ~100%, margin ~0.85-0.87); everything from `6e-4` up through `3e-3` fails outright (near-zero or negative margin) — not a gradual degradation, a cliff immediately above `3e-4`. Confirmed on 3 seeds at every failing value, not a fluke.

**n_hops=3** (`n_distractors=5`, `lr=3e-4`, 3 seeds): 100% / +0.838, 100% / +0.837, 100% / +0.820.
**n_hops=4** (`n_distractors=4`, `lr=3e-4`, 3 seeds): 99.98% / +0.795, 99.69% / +0.784, 99.77% / +0.798.

**Conclusion: all 3 n_hops levels (2, 3, 4) beat the no-retrieval shortcuts decisively and consistently across every seed (margin +0.78 to +0.87, no exceptions)** — per the standing decision table, this confirms real multi-hop chaining at the hardened-generator scale, not shortcut exploitation. Per the same table, relaunched Phase1bis (`use_ff`/`n_register`) at the hard `n_hops` (see next entry) rather than treating this as final — a ceiling this close to 100% could mask a use_ff/n_register effect that would only show up under more pressure (harder task or the same task with less capacity), so the ablation is a robustness/scaling-curve check, not a rescue of a failing result this time.

## 2026-09-14 — Phase1bis `use_ff`/`n_register` ablation at hard n_hops (3, 4) — launched, not yet analyzed

Reusing `n_hops=3`/`n_hops=4`'s exact working config (`lr=3e-4`, matching `n_distractors`), varying `use_ff ∈ {False, True}` × `n_register ∈ {1, 4}` (baseline cell `use_ff=False, n_register=1` already covered by the previous entry's 3-seed result), 2 seeds/cell, 12 runs total across 2 GPUs. Results pending at time of writing — see follow-up entry once complete.

## 2026-09-14 — Phase1bis pool_n_head/k_dim ablation complete (12/12): opposite LR-robustness ranking than initially suspected

`kdim128_decoupled` (asymmetric K/V dim=128, still decoupled) vs. `poolhead4_shared` (`--shared_kv_pooling --pool_n_head 4`), `n_hops=2, n_distractors=2, d_model=256`, 3 LR values × 2 seeds:

| variant | lr=3e-4 | lr=6e-4 | lr=1.2e-3 |
|---|---|---|---|
| `kdim128_decoupled` | ~100% both seeds | **100% (s0) / 33.4% (s1)** — inconsistent | fails both seeds (~20%, margin negative) |
| `poolhead4_shared` | ~100% both seeds | **99.2-99.8% both seeds** — robust | fails both seeds (~26-29%, margin negative) |

**Correction to the working hypothesis formed mid-sweep**: `poolhead4_shared` (multi-head pooling, still `shared_kv_pooling`) turned out to be the more LR-robust variant at `lr=6e-4` (both seeds solidly above shortcut), while `kdim128_decoupled` (asymmetric K/V width) is the inconsistent one at that same LR (one seed at ceiling, the other barely above chance) — the opposite ranking from an early read of the first few results streaming in. Neither variant survives `lr=1.2e-3` — same cliff already seen for the plain decoupled baseline in the Étape 4 LR sweep above, so this isn't a `pool_n_head`/`k_dim`-specific robustness gain in that direction, just noise/inconsistency specifically at `kdim128_decoupled`'s `6e-4` cell. Given the ceiling effect at `lr=3e-4` (both variants ~100%, no differentiation) and the mixed/inconsistent picture at `6e-4`, this doesn't cleanly match either of the pre-supplied decision-table branches ("poolhead4_shared ≈ full decouple" or a clear win/loss) — flagging to `model-design` as ambiguous rather than forcing a read.

## 2026-09-14 — `sm_cap` ablation complete: bounded (cap=1) buffer matches unlimited — same architectural signal as toy-memory Exp.2

`train_kb_chain.py`, `n_hops=2, n_distractors=2, lr=3e-4, d_model=256`, 3 seeds/cell:

| sm_cap | seed0 | seed1 | seed2 |
|---|---|---|---|
| 1 | 99.84% / +0.657 | 100% / +0.652 | 100% / +0.662 |
| none (unbounded) | 100% / +0.664 | 100% / +0.664 | 99.98% / +0.676 |

**`sm_cap=1` (a single-slot short-term-memory buffer) is indistinguishable from `sm_cap=None` (unbounded FIFO)** — both land in the same ~99.8-100% / margin +0.65-0.68 band, no seed showing a gap larger than ordinary run-to-run noise. This is the same qualitative finding as toy-memory Exp.2's `n_memory` ablation (`experiment.log.md`, earlier entry): whatever the short-term-memory buffer contributes here, one slot already provides it as well as an unbounded one — no evidence of the buffer accumulating useful content across multiple retained writes at this task/scale. Per the standing decision table, this is the "sm_cap=1 ≈ None" branch — reporting directly to `model-design` as a cross-workstream architectural finding (toy-memory + indexed-attention now agree), not filing it as inconclusive.

**Correction (same day, from `model-design`/user, applies to this entry AND the `disable_sm` follow-up below): this result does not actually settle whether the SM is useful.** `kb_chain_retrieval.py` is Markovian by construction — resolving hop $i{+}1$ never needs anything older than hop $i$'s value, so a single current state ($R$) suffices structurally for any chain length, `n_hops=4` included, independent of whether the multi-slot memory mechanism is good or not. **`sm_cap=1 ≈ sm_cap=None` (and whatever `disable_sm` shows) confirms this specific task cannot reveal the SM's usefulness either way — it is not evidence against the SM's architectural role (spec §-1), it's a limitation of this test's design.** Full detail in `dev_notes/indexed_attention_spec.md` §9.1. Read as "inconclusive on SM usefulness, task is Markovian," not as "SM is superfluous" — the earlier framing above ("important architectural finding") overstated what this result actually shows, before this correction arrived. A real test needs a task with independent facts to combine that are separated in time by a forced distraction phase (NTM/DNC-style recall) — not yet properly specified, `model-design`'s call on when to design and launch it.

## 2026-09-14 — Phase 3 real-text: LR sweep result + a real blocker for "continue plus longtemps"

`train_real_text.py`, `depth=1`, `data/distill/general_realtext/train.jsonl` (2700 docs), 2 seeds/LR:

| lr | seed0 final_loss | seed1 final_loss |
|---|---|---|
| 1e-4 | 6.99 | 6.80 |
| 3e-4 | 6.49 | 6.36 |
| 1e-3 | 6.05 | 5.89 |
| 3e-3 | 5.89 | 5.58 |
| 1e-2 | 6.81 | 6.10 |
| 3e-2 | 16.10 | 16.91 |

Clean, monotonic loss decrease from `1e-4` to `3e-3`, then degradation at `1e-2` (worse than `3e-3` but still training) and outright divergence at `3e-2`. **`lr=3e-3` is the optimum, cleanly bracketed on both sides** — not just the best of the originally-tested range.

**Blocker found while trying to act on "loss décroît proprement → continue plus longtemps"**: `train_real_text.py`'s `LockstepLaneBatcher` does a **single, non-repeating pass** over the 2700-document dataset — it exhausts the data and the generator stops after ~1537-1538 steps (~37-40s wall-clock) regardless of `--max_time_minutes`. Confirmed directly: a run launched with `--max_time_minutes 18` (vs. the sweep's 12) still stopped at `num_steps=1537`, `elapsed=0.65m` — i.e. it was a duplicate of the `lr=3e-3 seed=0` sweep cell, not a longer run. **"Continue plus longtemps" is not currently possible with this script/dataset as-is** — needs either (a) a multi-epoch/repeat mechanism added to `LockstepLaneBatcher` (currently `refill()` returns `False` and stops once `next_doc_ptr` exhausts `doc_order`, no wraparound), or (b) more real-text data than 2700 documents. Flagging to `model-design` rather than silently working around it — this directly blocks Phase 3's next planned step (Baselines A/B/C also still not implemented per the script's own docstring, separately).

## 2026-09-16 — `n_facts_curriculum` bug found and fixed: curriculum could never promote past stage 1 (n_facts=1)

Resuming the pending work flagged by commit `af2e1fb` (n_facts curriculum for associative recall, implemented in response to the 0/9-seeds-escaping finding at direct `n_facts=4` training): launched the actual sweep on Rennes (`paradoxe-9`, job 4111083, CPU) -- `--n_facts_curriculum 1,2,3,4 --n_facts 4 --latent_reset_at_query`, `lr ∈ {1e-3, 3e-3, 1e-2}` × 3 seeds, 30 min/cell.

**9/9 cells got stuck at `final_stage_n_facts=1 (stage 1/4)` after the full 30-minute budget, regardless of LR** -- not an LR effect (all three values, spanning a 10x range, did the exact same thing), a promotion-logic bug. Root cause (`learn/toy_memory/train_associative_recall.py`, promotion check): the curriculum promotes on `acc_excl_last >= args.curriculum_promote_acc`, but `acc_excl_last` is structurally `nan` at `n_facts=1` (there is no "query doesn't target the last-written fact" case when there is only one fact ever written -- `evaluate()`'s own code returns `float("nan")` when the underlying list is empty). `nan >= threshold` is always `False` in Python, so a curriculum whose first stage is `n_facts=1` could never promote out of it -- confirmed directly, all 9 cells show identical `step=79021`-ish stopping with zero `CURRICULUM PROMOTE` lines anywhere in the logs.

**Fix**: fall back to raw `acc` for the promotion check specifically when `acc_excl_last` is `nan` (only true at `n_facts=1`, where the shortcut/genuine-retrieval ambiguity `acc_excl_last` exists to resolve doesn't apply anyway -- there's only one possible query target). Verified locally (`PYTHONPATH=. python learn/toy_memory/train_associative_recall.py --n_facts_curriculum 1,2,3,4 --n_facts 4 --latent_reset_at_query --lr 3e-3 --curriculum_min_steps 20 --eval_every 20 --max_time_minutes 1`): promotes cleanly, `n_facts=2` at step 20, `n_facts=3` at step 60. Not yet committed.

**Relaunched** the same 9-cell sweep with the fix, same node/job (4111083, ~2h50 walltime remaining). Decision table for the result: curriculum reaching `n_facts=4` with `acc_excl_last` well above chance on a majority of seeds/LRs -> curriculum escapes the rare-solution trap the direct 9-seed sweep couldn't, proceed to the `n_memory` resweep already queued at the end of §5ter (with `--latent_reset_at_query`, LR revalidated fresh per standing rule since curriculum changes training dynamics); still 0/N genuine escapes even once `n_facts=4` is reached -> the trap is deeper than a landscape/curriculum issue, flag to `model-design` rather than continuing to tweak hyperparameters.

**Result, 30 min/cell budget**: with the fix, promotion now fires everywhere (unlike the flat 0/9 before) -- real, graded progress:

| lr | seed0 | seed1 | seed2 |
|---|---|---|---|
| 1e-3 | stage 3/4, acc_excl_last=0.873, recency_match_excl_last=0.106 | stage 2/4, acc_excl_last=0.449 | stage 2/4, acc_excl_last=0.494 |
| 3e-3 | stage 3/4, acc_excl_last=0.344 | stage 2/4, acc_excl_last=0.549 | stage 2/4, acc_excl_last=0.545 |
| 1e-2 | stage 2/4, acc_excl_last=0.536 | stage 2/4, acc_excl_last=0.271 | stage 3/4, acc_excl_last=0.050 (collapsed after promoting) |

**Read**: no cell reached the target `n_facts=4` stage within 30 min -- this looks budget-limited (per the standing LR-revalidation rule's protocol: check budget before reading anything into a plateau), not a repeat of the trap, since every cell is making real graded progress unlike the flat direct-training result. `lr=1e-3` looks best: highest `acc_excl_last` (0.873 at stage 3/4, seed0) with a large gap over `recency_match_excl_last` (0.106) -- most of that accuracy is genuine content-based retrieval, not the recency-echo shortcut. `lr=1e-2` looks worst/least stable (seed2 promoted to stage 3 then collapsed to near-chance, 0.050). Not treating this as conclusive yet -- extending budget on the most promising LR before reading a verdict, per the same rule (exhaust budget before concluding a wall).

**Follow-up launched**: `lr=1e-3`, 3 seeds, `--max_time_minutes 75` (same node/job, ~1h27 walltime was left), same curriculum/`--latent_reset_at_query` config -- goal is to actually reach `n_facts=4` and read `acc_excl_last` there. Results pending, see next entry.

**Follow-up result**: with `--max_steps` no longer binding (100k-127k steps reached, 38 min/seed, well past the 30-min budget that let the first sweep's `lr=1e-3 seed0` reach stage 3/4), **all 3 seeds stayed stuck at stage 2/4** this time -- including seed0, which had escaped to stage 3/4 (`acc_excl_last=0.873`) in the first, shorter sweep. This run's seed0 instead plateaus at `acc_excl_last=0.5435`, no better than seeds 1/2 (`0.4949`, `0.5311`). Budget is now clearly not the limiting factor (2-4x more steps, same outcome) -- this reads as genuine run-to-run instability in whether training escapes the stage-2 local solution, not a budget-starved curriculum. Two runs, same `lr`/seed/curriculum config, different `OMP_NUM_THREADS` (2 in the first sweep vs 4 here) is the only known difference, and CPU floating-point reduction order is thread-count-sensitive -- plausible mechanism for the divergent outcome despite "same seed", not proof.

**Reading**: matches this project's repeated "solution rare in the loss landscape" signature (toy-model `read_step` curriculum, Indexed Attention `n_facts=64` curriculum, both logged earlier) rather than a budget or LR problem -- per the standing LR-revalidation rule, budget and LR are now both exhausted as explanations (ample budget tried, `lr=1e-3` was the best of 3 values tried with real budget). Pausing this thread here rather than continuing to relaunch single-shot variants: the open question is now "what fraction of seeds/restarts escape stage 2 at this config", which needs a wider-seed characterization (cheap, parallelizable) or a curriculum-design change (e.g. relaxed `--curriculum_promote_acc`, more stages, longer `--curriculum_min_steps` dwell) rather than more time on the same 3 seeds -- flagging to `model-design` for a call on which lever to pull next.

## 2026-09-16 — Baselines A/B/C real-text grid (LFM2/OLMo/Qwen), a real `tokenizer.vocab_size` bug found, 13/15 done

Relayed by `model-design` on behalf of domguia: Baselines A (`--n_step 1`) / B (`--disable_kb`) / C (main, `depth=1`) x {lfm2, olmo} x {`d_model=128`, `d_model=1024` spec §13 sizing}, plus `qwen`/`d_model=1024` (checked first, not a duplicate). 15 cells via `tools/exp/`, see `grid5000_usage.log.md` for the launch/infra details.

**Bug found**: both `qwen` cells that actually build/use the KB embedding path (`n_step=1` and the main/`depth=1` run) crashed identically at the very first forward pass -- `IndexError: index out of range in self` in `self.embed(kb_tokens)`. Root cause: `train_real_text.py` sized the embedding table (and the output head) from `tokenizer.vocab_size`, but that attribute reports the tokenizer's **base** vocab, excluding added/special tokens -- confirmed directly (`Qwen/Qwen3-0.6B`: `vocab_size=151643` vs `len(tokenizer)=151669`, a 26-token gap; `LiquidAI/LFM2-350M` and `allenai/OLMo-2-0425-1B` both have zero gap, so their 12 completed cells are unaffected). The `disable_kb` cell for `qwen` happened to succeed anyway -- it never exercises the `self.embed(kb_tokens)` path that hit the id above `vocab_size-1`. **Fix**: use `len(tok)` instead of `tok.vocab_size` everywhere the embedding/head is sized (`learn/indexed_attention/train_real_text.py`). Verified with a 5-step smoke run on `qwen`/`d_model=1024` post-fix (`params=321.32M`, no crash). Not yet committed. Stale `state`/`claims` files for the two crashed cells deleted and re-queued through the same worker pool (still using the fixed code now synced to the node) rather than a fresh grid -- run_id is a hash of (script, config), unaffected by an internal code fix, so this is a legitimate resume, not a duplicate.

**Preliminary read on the 13 completed cells (`final_loss`, lower is better; all still far from converged -- 1000-5500 steps out of the 200000-step ceiling, this is an early-training snapshot, not an asymptotic comparison)**:

| tokenizer | d_model | A (`n_step=1`) | B (`disable_kb`) | C (main, `depth=1`) |
|---|---|---|---|---|
| lfm2 | 128 | **4.948** (5493 steps) | 6.383 (5386 steps) | 5.525 (4233 steps) |
| olmo | 128 | **5.405** (3393 steps) | 6.113 (3314 steps) | 6.153 (2987 steps) |
| lfm2 | 1024 | **4.828** (1915 steps) | 6.966 (1936 steps) | 7.678 (1549 steps) |
| olmo | 1024 | **5.902** (1303 steps) | 9.803 (1344 steps) | 8.447 (1081 steps) |

**Baseline A (flat retrieval, no loop) has the lowest loss in all 4 cells -- beating the main model (C) at this budget, on both tokenizers, at both `d_model`.** Read with real caution before treating this as "iteration doesn't help": each baseline gets a different number of gradient steps for the same wall-clock budget, because `n_step` directly multiplies per-step compute (A's `n_step=1` vs C's default `n_step=6` -- confirmed in the step counts above, A consistently reaches ~1.5-2x more steps than C in the same 30/45 min). This comparison confounds "does the loop help" with "same wall time, fewer steps" -- not a clean ablation as currently run. B (`disable_kb`, loop but no memory) is worst everywhere as expected (no memory access should hurt), which is at least a sane sanity check on the harness itself. **Not drawing an architecture conclusion from this table** -- flagging the wall-time-vs-step-count confound to `model-design`; a fair comparison needs either matched step count (not wall time) or accounting for the per-step cost difference explicitly.

**Update: 15/15 cells done** (the 2 re-queued `qwen` cells finished cleanly, well before the reserving job's 6h walltime ran out and reclaimed the node). Final row for `qwen`/`d_model=1024`, same shape as the rest of the table above:

| tokenizer | d_model | A (`n_step=1`) | B (`disable_kb`) | C (main, `depth=1`) |
|---|---|---|---|---|
| qwen | 1024 | **4.897** (2686 steps) | 11.726 (1544 steps) | 6.101 (2267 steps) |

Same pattern as lfm2/olmo: A lowest loss, B highest, C in between -- consistent with the wall-time/step-count confound already flagged (A's `n_step=1` reaches ~1.2-1.7x more steps than C here too). Grid fully collected; no new read beyond what's already flagged to `model-design` (step-count confound needs resolving before any "does the loop help" claim).

## 2026-09-19 — I1 (deadline plan): Phase1bis `use_ff`/`n_register` at hard `n_hops` (3,4), collected from job 4106501 -- 11/12, no architectural signal

Resumed job 4106501 (launched 2026-09-14, `abacus11`, 12 runs across 2 GPUs -- see the 2026-09-14 launch entry above) via its raw logs on the Rennes home (`~/thinker/logs/phase1bis_hardhops/*.log`; this grid predates `tools/exp/`, so `collect.py` has no `grid.jsonl` for it and the logs had to be read directly). All 12 processes did run (`GPU0_DONE` present; `GPU1_DONE` missing, matching the incomplete cell below) -- this was genuinely unread output, not lost work.

`train_kb_chain.py --depth 2 --block_size 4 --d_model 256 --n_step 12 --batch_size 256 --lr 3e-4`, `use_ff ∈ {False, True}` × `n_register ∈ {1, 4}` (baseline cell `use_ff=False, n_register=1` already covered by the earlier 3-seed `n_hops` sweep, not rerun here), 2 seeds/cell:

**n_hops=3, n_distractors=5** (baseline for reference: 100%/+0.838, 100%/+0.837, 100%/+0.820):

| variant | seed0 | seed1 |
|---|---|---|
| `ff0_nreg4` | 99.94% / +0.8324 | 99.77% / +0.8307 |
| `ff1_nreg1` | 95.88% / +0.8025 | 99.98% / +0.8460 |
| `ff1_nreg4` | 99.88% / +0.8318 | 99.98% / +0.8328 |

**n_hops=4, n_distractors=4** (baseline for reference: 99.98%/+0.795, 99.69%/+0.784, 99.77%/+0.798):

| variant | seed0 | seed1 |
|---|---|---|
| `ff0_nreg4` | 99.79% / +0.8050 | 97.42% / +0.7667 |
| `ff1_nreg1` | 99.55% / +0.7963 | 99.45% / +0.8007 |
| `ff1_nreg4` | 99.77% / +0.8067 | **incomplete -- walltime cut the run off mid-training (log ends at step 3000, `eval_acc(held_out)=0.9949`, no final chance-level report/`margin_over_shortcut` line ever printed)** |

**Read (11/12 cells)**: every completed variant lands in the same tight band as the already-established baseline -- 95.9-100% accuracy, `margin_over_shortcut` +0.77 to +0.85, all comfortably above every no-retrieval shortcut. No variant (`use_ff`, `n_register=4`, or both together) separates from the baseline or from each other outside ordinary seed-to-seed noise (the single lowest point, `ff1_nreg1 n_hops=3 seed0` at 95.88%/+0.8025, is still far above shortcut and not clearly worse than the baseline's own seed spread). **Verdict: `use_ff`/`n_register` show no detectable effect at these hard `n_hops`, i.e. the ceiling from Étape 4 is real and robust, not masking a use_ff/n_register-driven difference** -- closes the ablation the 2026-09-14 entry flagged as still open, pending the one missing cell.

**Not yet final**: `h4_ff1_nreg4_s1` needs a ~15-min GPU relaunch (same `run_id`-equivalent config, legitimate resume not a duplicate) to complete the 12/12 picture, but given how tightly every other cell already clusters, a single additional seed is very unlikely to change the "no effect" read. Per this session's explicit instruction, no GPU reservation made for this alone -- deferred, to be bundled with the next GPU reservation once that's decided.

## 2026-09-19 — I4 (deadline plan): soft attribution measure -- real signal above both controls, but on a different checkpoint than the historical 0,000 audit (important caveat)

Goal (relayed): the hard top-1 argmax audit on `memory.attend` gave 0,000 at both hops despite task accuracy far above chance -- inconclusive by design (spec §5.3, softmax retrieval is soft/diffuse, no hard selection), never followed up with the soft correlation test the plan flagged since 09-13. CPU only, no training intended.

**Checkpoint provenance caveat -- read before trusting the numbers below**: the saved checkpoint `diagnose_no_ff_composition.py` required (`/tmp/diagnose_n_hops2_no_ff_checkpoint.pt`) was stale against current `core/indexed_memory.py` (`missing key: memory.compressor.query_v` -- an architecture change since it was saved). Regenerating it meant re-running `diagnose_no_ff_composition.py`'s training loop (CPU, ~12 min, no GPU) under **current** code, which includes the `decouple_kv=True` default fix (`experiment_plan.md` line 16) that the *original* 0,000-audit checkpoint (2026-09-13, pre-fix) did not have. Consequence: the regenerated checkpoint converges to **99.6-100% task accuracy**, not the ~23-32% plateau the historical top-1-hard audit and the earlier soft-measure attempt (`experiment_plan.md`, "Mesure douce, corrigée deux fois" section) were run against. **This result answers "does `o_kb` correlate with the correct value once the task is actually solved", not "...on the specific broken checkpoint that gave 0,000" -- that exact checkpoint is gone and not reproducible without checking out the pre-`decouple_kv`-fix code.** Flagging this precisely rather than presenting it as a direct retraction of the old number, per this project's own rule against accuracy claims without full context.

Extended `learn/indexed_attention/diagnose_no_ff_soft_retrieval.py` with the two required trivial controls (previously had only the correct-value cos_sim and an in-KB rank test, no baseline to compare against): `control_random_kb` (cos_sim to a wrong candidate value drawn from the same episode's own KB) and `control_non_key` (cos_sim to a value never inserted into this episode's memory at all -- a random vocab token pushed through the same `v_proj(embed(x) + source_bias(KB))` transform). Both stay in `o_kb`'s actual vector space (the module's own prior fix), and both are reported as full distributions (mean/std/p10/p50/p90), not just a mean. `n_eval=256`.

| | correct value | control random_kb | control non_key | gap (correct − random_kb) | gap (correct − non_key) | top-1 (chance=25%) |
|---|---|---|---|---|---|---|
| hop1 (natural R0) | mean=+0.650, p50=+0.661 | mean=+0.222, p50=+0.260 | mean=+0.162, p50=+0.153 | +0.428 | +0.488 | **93.4%** |
| hop2 (forced R, clean mid_val query) | mean=+0.585, p50=+0.609 | mean=+0.307, p50=+0.273 | mean=+0.187, p50=+0.178 | +0.279 | +0.399 | **63.3%** |

**Read, on this healthy (post-`decouple_kv`-fix) checkpoint**: `o_kb` correlates clearly and consistently with the correct value's true `v_proj` vector, well above both controls at both hops -- not an artifact of comparing against a degenerate/near-zero baseline (both controls sit at a real, non-trivial +0.16 to +0.31 themselves, from generic KB/vocabulary structure, and the correct-value signal still clears them by a wide margin). Weaker at hop 2 than hop 1 (63.3% vs 93.4% top-1) despite both being fed a "clean" query, consistent with the second retrieval being the harder one throughout this project's diagnostics -- but still 2.5x chance, a real signal, not noise. **Per the decision table's first branch, this is a real and worth-reporting finding**: on a model that actually solves the task, retrieval is genuinely diffuse-but-correlated, i.e. a hard top-1-argmax metric would systematically under-measure real KB attribution even in a healthy model -- worth noting in the paper as a methodological point about measuring soft retrieval, independent of the checkpoint-provenance caveat above. **What remains open**: whether the *original* plateaued (~23-32% accuracy) checkpoint would show the same pattern (real-but-diffuse) or the flat/anti-correlated pattern the historical weight-inspection round actually found (`experiment_plan.md`, "Triangulation complète" -- below-chance top-1 in K-space specifically on that broken checkpoint) -- this run does not settle that, since it's a different model. Re-running on the historical broken regime (if worth the time before the deadline) would need reproducing that pre-fix training run specifically, not just any n_hops=2 checkpoint.

**Second reading, added 2026-09-19 (relayed by `ff2attn`)**: the hard top-1-argmax metric is not a blind/noisy measure -- on this same healthy checkpoint it scores 93.4% at hop 1, well above its own 25% chance floor, so the metric clearly *can* detect real attribution when it's there. That sharpens what the historical 0,000 (not ~25%, not noise around chance -- exactly zero) on the pre-fix broken checkpoint actually means: a blind/uninformative metric would scatter around chance, but an exact zero means the mechanism never once pointed at the correct leaf -- a positive signal of systematic failure, not an absence of signal. This is independently consistent with the already-established root cause of that era's plateau (`experiment_plan.md` line 16: `LevelCompressor` pooling `parent_k`/`parent_v` with the *same* softmax, making key->value association structurally unrepresentable) and with the historical weight-inspection round's own below-chance top-1 in K-space on that same broken checkpoint. Two independent lines now converge on the same explanation. **Practical consequence: reproducing the broken pre-fix regime specifically (to directly settle whether it shows real-but-diffuse vs. truly-flat retrieval) is not worth its cost at 6 days out** -- the checkpoint-provenance caveat above stays as the accurate scope statement of what this run measured; this paragraph is an additional, independent inference on top of it, not a replacement.

## 2026-09-19 — Piste A: step-matched real-text baselines rerun -- A still beats C, gap widens at larger d_model (early-training snapshot, not asymptotic)

Goal: the 2026-09-16 Baselines A/B/C grid (`experiment.log.md` above) confounded "does the loop help" with "same wall time, fewer gradient steps" -- Baseline A (`n_step=1`) reached 1.2-2x more steps than C (`n_step=6`, main config) in the same budget. Reran **step-matched** instead: `--max_steps 1000` fixed identically for every cell (chosen conservative -- the slowest historical cell, C at `d_model=1024`, reached only 1081-1549 steps in 30-45 min wall-clock on 2026-09-16, so 1000 is comfortably reachable by every variant), `--max_time_minutes 60` generous/non-binding. 36 cells (A/B/C x {lfm2, olmo} x {`d_model`=128, 1024} x 3 seeds), via `tools/exp/`, `abacus18-1` (3x RTX 6000), all 36 done in ~16 minutes wall-clock (much faster than the original wall-time-matched grid, expected -- 1000 steps is a low, fixed bar every cell clears quickly rather than running until a time budget expires).

**Mean `final_loss` at step 1000 (lower is better), 6 cells averaged per (baseline, d_model) cell -- both tokenizers, 3 seeds pooled**:

| d_model | A (`n_step=1`) | C (main, `n_step=6`) | B (`disable_kb`) |
|---|---|---|---|
| 128 | **6.49** | 6.88 | 7.44 |
| 1024 | **6.04** | 8.43 | 9.66 |

**[CORRECTION, same day, before this was ever read as final -- LR was never revalidated for this grid]**: every cell in this grid used `train_real_text.py`'s bare default `--lr 3e-4` -- confirmed by checking the grid config directly, `lr` was never set in `gridgen`'s `--fixed`/`--sweep`. That is the exact "changed axis, reused an old LR" pattern this project's own standing rule exists to catch (already burned twice: `decouple_kv` at `lr=1.2e-3`, `pool_n_head` bimodality at `lr=6e-4`). **Pulled the actual loss curves from the `RunLogger` state history (`progress()` every 20 steps) before trusting the table above, per that same rule -- and it matters**:

- `d_model=1024`, C (main, `n_step=6`), seed 0, lfm2 (`train_real_text-5c73e1061b`): loss is **not** descending cleanly -- `11.6 -> 12.3 -> 12.0 -> 16.6 -> 11.6 -> 9.0 -> 10.1 -> 10.5 -> 23.1 -> 12.5` (steps 0-900, every 100). Wild oscillation with a spike to 23 at step 800 -- textbook LR-too-high instability, not a converging run.
- `d_model=1024`, A (`n_step=1`), seed 0, lfm2 (`train_real_text-f6a6b848f6`): smooth, roughly monotonic descent, `11.4 -> 7.8 -> 7.1 -> 6.8 -> 6.5 -> 6.2 -> 6.3 -> 6.2 -> 6.1 -> 6.0`.
- `d_model=128`, same comparison: **both** A and C descend smoothly (C: `11.5 -> 8.9 -> 8.2 -> 7.8 -> 7.4 -> 7.3 -> 7.3 -> 7.2 -> 6.9 -> 6.99`), no instability on either side.

**So the widening A-C gap at `d_model=1024` is very likely an artifact of `lr=3e-4` being too high specifically for C's config at that scale (`n_step=6`, more compounded gradient paths per step than A's `n_step=1`), not a genuine "the loop scales worse" finding.** `d_model=1024` is a real axis change (bigger model) on top of `n_step` already differing between A and C -- exactly the situation the LR-revalidation rule targets, and this grid skipped it entirely.

**Retracting the "real negative result" framing above -- this is NOT yet a validated result.** What actually still stands: at `d_model=128`, both curves are stable and A beats C at matched steps by a modest, currently-uncorrupted margin (0.38) -- worth keeping as a tentative, small-scale-only observation. The `d_model=1024` comparison (and its "gap widens" claim) is retracted pending an LR sweep for **A and C separately at `d_model=1024`** (their optimization landscapes differ -- `n_step=6` vs `n_step=1` is not the same mechanism, per the standing rule's own guidance) before any conclusion, positive or negative, is drawn at that scale.

**Both wall-time-matched (2026-09-16) and this step-matched attempt remain open questions for the `d_model=1024` / main-config comparison specifically** -- not yet a defensible "A beats C" claim at that scale until LR is recalibrated there. The `d_model=128` step-matched reading is the only piece of this entry currently safe to cite.

## 2026-09-19 — A1/A2 (Phase 12 pre-check): a measurement artifact caught, real signal is quasi-orthogonality

`learn/indexed_attention/diagnose_olmo_ffn_memory_geometry.py`, `allenai/OLMo-2-0425-1B` pretrained weights, CPU, no training, ~2 min wall-clock (`paradoxe-27`, job 4121204). Goal: does unifying the FFN-key space (`W_gate`) with the attention-key space (`W_K`) cost a real distillation, or is it nearly free -- decides S1's branch (i) vs (ii) and informs S3's feasibility, before any conversion code is written.

**Measurement artifact caught before it produced a wrong conclusion**: the "energy of `W_gate` projected onto `span(W_K)`" metric read **exactly 1.0000 at every layer**, and the top-10 principal-angle cosines between the two subspaces are **exactly 1.0** too. This looks like "total overlap" but isn't -- `AutoConfig` confirms OLMo-2-1B uses full MHA (`num_attention_heads=16 == num_key_value_heads=16`, no GQA), so `W_K` has shape `(2048, 2048)` and is full column rank: `span(W_K)` is trivially the **entire** 2048-dimensional ambient space. Any vector at all -- `W_gate`'s rows included, or pure noise -- projects onto it with 100% energy and 0-degree principal angles. **This specific pair of metrics is uninformative whenever the reference subspace (`W_K` here) has rank >= `d_model`**, which is exactly OLMo's case; reporting the 1.0 values as "unification is nearly free" would have been a real, wrong conclusion in the opposite direction from what the data actually supports.

**The metric that survives this trap**: pairwise cosine similarity between individual `W_gate` key vectors and individual `W_K` key vectors (computed on a 512x512 random subsample, doesn't depend on either matrix's rank). Mean `|cos|` = **0.017-0.021 across all 16 layers**, p90 `|cos|` = 0.044 (layer 0). The expected mean `|cos|` between two independently random unit vectors in `R^2048` is `sqrt(2/(pi*2048))` approx **0.0176** -- OLMo's actual `W_gate`/`W_K` key vectors sit right at that random-vector floor, no measurable directional alignment beyond chance.

**Read (A1)**: `W_gate` (FFN "keys") and `W_K` (attention keys) are **quasi-orthogonal**, not overlapping -- the substantive finding the flawed energy/angle metric obscured. Per the decision table's second branch: **unifying the K/V space across the three memories is not nearly-free; it costs a real distillation to budget**, and S1 should keep "conversion first, anneal into the shared space" (start from S0's exact equivalence, not from a from-scratch shared-space init) rather than treating unification as elegant-and-cheap.

**A2 (cross-layer FFN redundancy)**: stacking all 16 layers' `W_gate` keys (131072 vectors total, `d_model=2048`) gives effective rank **1699/2048 (83%) at 90% energy, 2000/2048 (98%) at 99% energy** -- close to the full ambient dimension, i.e. **very little global redundancy relative to `d_model`** despite the huge nominal key count (131072). Adjacent-layer mean `|cos|` (0.0207) and distant-layer mean `|cos|` (0.0178) are both near the same random-vector floor as A1 and barely distinguishable from each other -- **no clear low/high stratification signal at this coarse per-vector-pair-cosine level** (doesn't rule out subspace-level stratification a finer analysis might catch, just not visible here).

**Read (A2)**: a shared "universal layer" memory (S3) is **not** cheaply compressible from redundancy -- the keys already nearly span the available `d_model`-dimensional space, so unioning them (per the plan's own "union, never average" rule) would need close to the full ambient rank, not a small shared subspace. Combined with A1's quasi-orthogonality finding, **both free pre-checks point the same direction: the elegant/cheap versions of unification (S1 branch (ii), and S3's shared layer) are not free lunches at this model's scale** -- the harder, distillation-requiring paths are the ones actually supported by the geometry. Full per-layer numbers in `runs/olmo_ffn_geometry/result.json` (Rennes home).

**[CORRECTION, same day, before design work started from this reading]**: `model-design` caught that **both A1 and A2's conclusions above outrun what their metrics actually measure**, independent of the rank artifact already caught:
- A1: mean pairwise cosine between individual `W_gate`/`W_K` row vectors doesn't distinguish "incompatible key spaces" from "same space, different rotation" -- two arbitrary orthonormal bases of the *same* `R^d` also average to the random-vector cosine floor. And since `W_K` is already established as full rank (its span is literally all of `R^2048`), the FFN keys live in that same ambient space by construction -- subspace membership was never the right question. The real question is **score scale/geometry under a real query**: do `q.k_ffn` and `q.k_attn` land at comparable magnitudes for an actual residual-stream query, or does one source dominate a unified softmax purely on scale (spec Sec.3.2's "mass vs. peak" risk) regardless of relevance?
- A2: rank of 131072 vectors in `R^2048` is bounded at 2048 *by construction* -- "not fully redundant" was guaranteed before any data was measured, uninformative about S3. The real question is per-key **substitutability**: does a key in layer L have a near-duplicate in another layer? A few thousand near-duplicates among 131k keys would be decisive for S3 and completely invisible in a mean-cosine or rank statistic -- needs a nearest-neighbor **tail** statistic instead.

Both re-measured directly on real activations/nearest-neighbor tails (`diagnose_olmo_ffn_memory_geometry_v2.py`, CPU, ~2 min) rather than left as an open retraction -- see the follow-up entry immediately below for the corrected numbers. The "quasi-orthogonal" / "not compressible" readings above are **retracted as stated**; whatever the v2 entry below says supersedes this one.

## 2026-09-19 — A1/A2 v2: corrected measures -- score-scale mismatch confirmed real, S3 near-duplication genuinely absent

`diagnose_olmo_ffn_memory_geometry_v2.py`, `allenai/OLMo-2-0425-1B`, real activations from `data/distill/general_realtext/train.jsonl` (512-token sample), CPU, `paradoxe-27`, ~2 min.

**A1 (score-scale geometry, real queries)** -- per layer (0, 4, 8, 11, 15), `q = q_proj(hidden)`, `k_attn = k_proj(hidden)` (512 real per-token keys), `k_ffn = W_gate` (8192 static rows), scores scaled by `1/sqrt(head_dim)` on both sides for a like-for-like comparison:

| layer | \|\|k_attn\|\| | \|\|k_ffn\|\| | score_attn (mean/p99/max) | score_ffn (mean/p99/max) | unified-softmax mass on FFN |
|---|---|---|---|---|---|
| 0 | 15.8 | 2.36 | 0.26 / 1.07 / 2.65 | -0.01 / 0.18 / 0.65 | **92.4%** |
| 4 | 16.6 | 1.92 | 0.14 / 0.83 / 9.88 | -0.00 / 0.16 / 0.67 | **93.2%** |
| 8 | 21.9 | 1.83 | 0.39 / 1.42 / 84.7 | 0.00 / 0.21 / 0.87 | 75.8% |
| 11 | 27.4 | 1.74 | 0.58 / 2.46 / 31.2 | 0.00 / 0.25 / 1.32 | 82.0% |
| 15 | 70.3 | 1.49 | 6.50 / 27.8 / 83.8 | -0.01 / 0.53 / 1.53 | **0.29%** |

**Read**: this is the real, decisive answer A1 v1 couldn't give. `k_attn` norms and scores **grow sharply with depth** (norm 15.8->70.3, score mean 0.26->6.50, max up to 84.7 by layer 15) while `k_ffn` scores stay essentially flat and near-zero at **every** depth (mean ~0, max never above 1.5). Concatenated under one softmax, this produces a wildly depth-inconsistent mixing ratio that has nothing to do with relevance: **FFN keys capture 92-93% of the softmax mass in early layers** (winning purely on count -- 8192 keys at near-zero score each still out-accumulate 512 keys at slightly-less-near-zero score) and then **collapse to 0.29% by layer 15** (attention's few huge-magnitude scores become winner-take-all). **This is exactly the "mass vs. peak" risk spec Sec.3.2 flagged, now confirmed as a real, large, depth-dependent effect on an actual pretrained model -- not a hypothetical.** Directly supports keeping **branch (i)** (separate reads with source-appropriate kernels, e.g. `relu`/`sigmoid` for the KB, softmax for the sequence, per the plan's already-stated default) rather than (ii) (naive unified softmax), or at minimum a per-source scale calibration before any unified softmax is attempted.

**A2 (nearest-neighbor substitutability)** -- adjacent-layer pairs, distant pairs, and intra-layer self-control (self-match excluded):

| pair type | mean(max_cos) | frac > 0.5 | frac > 0.7 | frac > 0.9 |
|---|---|---|---|---|
| adjacent (avg of 15 pairs) | 0.12-0.19 | 0.09-1.2% | ~0-0.06% | ~0% |
| distant (0-15, 0-8, 7-15, 3-12) | 0.084-0.091 | **0.0%** | 0.0% | 0.0% |
| intra-layer self-control (0,8,15) | 0.20-0.21 | 0.5-4.0% | 0-0.8% | 0-0.26% |

**Read**: this is a properly-posed substitutability tail statistic, and it's clean -- **near-duplicate FFN keys are genuinely rare everywhere**, including within a single layer's own 8192 keys (the intra-layer control, which should show the *highest* self-similarity of any comparison, still has essentially 0% of keys with a near-duplicate at `cos > 0.9`). Adjacent layers are mildly more self-similar than distant ones (0.12-0.19 vs 0.084-0.091 mean max-cosine) -- a small, consistent signal in the direction Geva's stratification predicts, but nowhere near "near-duplicates exist to exploit." **This reading survives the correction and can be kept**: a shared/universal-layer memory (S3) cannot be built by deduplication or nearest-neighbor merging -- any real compression there would need to be learned (distillation/projection), not found for free in the raw key geometry.

Full numbers: `runs/olmo_ffn_geometry/result_v2.json`.

## 2026-09-19/20 (nuit) — Piste A LR sweep at d_model=1024: A and C do NOT share an optimal LR, and step-matched != optimization-progress-matched

`pistea_lrsweep_d1024` (A: `n_step=1`, C: main/`n_step=6`, both `d_model=1024`, `tokenizer=lfm2`, `max_steps=1000` matched, 3 seeds/cell), launched to resolve whether the retracted "A beats C, gap widens to 2.39" reading (above) survives a proper LR sweep instead of the shared, unvalidated `lr=3e-4`.

**Wave 1 (`lr` in `{1e-5, 3e-5, 1e-4, 3e-4}`) mean `final_loss`**:

| lr | A (`n_step=1`) | C (`n_step=6`, main) |
|---|---|---|
| 1e-5 | 7.014 | 7.761 |
| 3e-5 | 6.587 | **7.672 -- clean bracketed minimum for C** (worse on both sides: 1e-5 and 1e-4) |
| 1e-4 | 6.181 | 7.881 |
| 3e-4 | **6.050 -- best A so far, still at the range's top edge** | 8.691, unstable (one seed 9.289) |

**C's optimum is real and bracketed at `lr=3e-5`** -- not a boundary artifact. **A's optimum is not yet bracketed** (still improving at the range's top edge); wave 2 (`lr` in `{1e-3, 3e-3}`) is resolving it.

**Central finding, independent of A's still-open optimum**: C's best LR (~3e-5) is **roughly 10x lower** than A's (>=3e-4) -- the looped mechanism (`n_step=6`) needs a substantially lower, narrower LR window than the flat baseline (`n_step=1`) at this scale. This on its own explains why the original shared-`lr=3e-4` comparison was invalid (C was unstable at that LR, A was fine).

**[Methodological correction from `model-design`, catches a second confound the step-matching alone didn't remove]**: pairing `max_steps` removes the wall-clock/step-count confound (the original 2026-09-16 issue), but **introduces a different one once the two architectures' optimal LRs differ by an order of magnitude: at matched step count but each at its own optimal LR, the two architectures do not travel the same distance through parameter space.** C at `lr=3e-5` for 1000 steps moves far less than A at `lr=3e-4` for 1000 steps -- so a 1000-step comparison, however carefully LR-tuned, is not actually an apples-to-apples "does the loop help" measurement; it confounds "architecture quality" with "optimization progress made in the same step budget." **This is a limit of the comparison, not a result -- the 1000-step gap (currently 1.62 at each architecture's best-known LR) must not be reported as a negative finding on its own.** The only way to actually settle "does the loop help" is a longer, per-architecture-optimal-LR run where the loss curves either converge, cross, or stay apart -- see the extended-budget follow-up (next entry, queued once A's LR is bracketed).

Not yet concluding anything from this entry (charte d'autonomie: pas de conclusion sur résultat partiel) -- wave 2 still running.

**À arbitrer**: none yet from this thread -- flagged here as a placeholder since the eventual extended-budget read (curves crossing late vs. staying apart) may itself land in "ambiguous, needs a human call" territory depending on what the curves actually do.

**Wave 2 complete (`lr` in `{1e-3, 3e-3}` added) -- both optima now cleanly bracketed**:

| lr | A (`n_step=1`) mean | C (`n_step=6`) mean |
|---|---|---|
| 1e-5 | 7.014 | 7.761 |
| 3e-5 | 6.587 | **7.672 -- C's optimum, bracketed** |
| 1e-4 | 6.181 | 7.881 |
| 3e-4 | **6.050 -- A's optimum, bracketed** | 8.691 |
| 1e-3 | 6.495 (worse than 3e-4) | 91.6 (exploding) |
| 3e-3 | 10.47 (diverging) | NaN (fully diverged) |

**A's optimum is `lr=3e-4`** (worse on both sides: 1e-4 and 1e-3), **C's optimum is `lr=3e-5`** (worse on both sides: 1e-5 and 1e-4) -- confirmed 10x apart, both now solidly bracketed rather than open questions. At each architecture's own best LR: A=6.050, C=7.672, gap=1.622. **Per the methodological correction above, this gap is NOT reported as a "does the loop help" verdict** -- it's the necessary input to the extended-budget follow-up (next entry), which is the actual test.

Extended-budget follow-up (item [2], each architecture at its own bracketed optimum: A `lr=3e-4`, C `lr=3e-5`, `d_model` in {128, 1024}, 8000 steps, 2 seeds, full curve logged every 20 steps) launched on `abacus18` once the sweep freed it. Results pending, see next entry.

## 2026-09-19/20 (nuit) — I7: N_step generalization + read-step accuracy curve -- [RETRACTED as a positive result] training-margin confound identified

Relayed overnight: does Thinker's loop generalize to `N_step` values it was never trained at, without retraining -- or did it learn a step-indexed stopping heuristic (spec Sec.7.2bis: `t` may be knowable via SM state/size, but must never be parameterized)? Two eval-only measurements on the same unrolled pass, `diagnose_nstep_generalization.py`.

**Scope caveat, read first**: Étape 4's actual hardened-generator checkpoints (`n_hops` in `{2,3,4}`, `d_model=256`, GPU) were **never saved to disk** -- confirmed tonight, `train_kb_chain.py` never called `torch.save` until this session added `--save_checkpoint_path` (committed). This run instead reuses the small CPU checkpoint already available from I4/I6 (`n_hops=2`, `d_model=32`, trained `N_step=16`) -- **not** the real Étape 4 scale the night's brief asked about. A proper rerun at real scale is queued once GPU frees from item [2]; this is a partial, small-scale first signal, not the final answer.

**I7a -- eval at `N_step_test` != trained `N_step=16`, zero retraining** (chance=25%):

| N_step_test | 2 | 4 | 6 | 8 | 12 | **16 (trained)** | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| acc | 1.2% | 3.1% | 33.0% | 73.1% | 98.6% | **99.8%** | 99.8% | 98.4% |

**I7b -- single long unroll to `N_step=24`, accuracy read out at every intermediate step `t`** (same eval batch):

| t | 1 | 4 | 8 | 12 | **16 (trained)** | 20 | 24 |
|---|---|---|---|---|---|---|---|
| acc | 2.0% | 5.3% | 73.1% | 99.0% | **99.4%** | 99.0% | 98.2% |

**[CORRECTION, `model-design`, same night]**: **the "no drift past trained `N_step`" read above is retracted as a positive result -- a training-margin confound was found.** The checkpoint is `n_hops=2` trained at `N_step=16`: the task needs 2 hops, so the model spent **~14 of its 16 training steps already stable/idle** (already trained to hold its answer well past task completion). Testing up to `N_step=24` is then only 8 steps beyond a *margin of 14 it was already trained on* -- not a real extrapolation test. "No drift to 24" is close to trivial in that regime: it shows the model extrapolates slightly past an idle margin it was explicitly given, not that it learned a genuine content-based implicit stop from a tight budget. **The real test needs a LOW training margin** -- e.g. `n_hops=4` trained at `N_step=6` (only 2 steps of slack), then evaluated at 12/24/48 -- that is the condition that would actually distinguish "learned to hold" from "learned to occupy a fixed budget." **Marking this "à arbitrer, confound identified" rather than a result.** Folded into the queued real-scale rerun (`--save_checkpoint_path`, Étape 4 configs): **training margin (`N_step - n_hops`) is added as an explicit variable of that rerun**, not just `N_step` alone -- this single design fix addresses both the scale caveat above and this confound at once.

**Read**: clean and, on this small checkpoint, genuinely positive for the thesis. Below the trained `N_step`, accuracy is low -- expected and uninformative (the task structurally needs enough hops before it *can* be solved, this isn't the interesting direction). **Above the trained `N_step` (the real test) is where it matters: accuracy does not collapse or drift -- it plateaus at ~98-99% from `t=12` onward and holds flat through `t=24` (1.5x the trained budget), both in the zero-retrain generalization sweep (I7a) and in the single continuous unroll with per-step readout (I7b).** Per the decision table's first branch: this is the "over-iteration is safe" signature -- an implicit, content-based stabilization rather than a model that was simply read at the one externally-fixed step it happened to be tuned for. Matches the framing from tonight's spec discussion (Sec.7.2bis): the SM growing by append lets the model's *state* reflect progress without any step index ever being a parameter, and this result is consistent with the model actually using that state to stabilize rather than drifting once past its "trained" length.

**Not a paper-ready claim, and not a result at all as things stand**: `n_hops=2` toy CPU scale, wrong training margin (see correction above). **Queued**: rerun with training margin (`N_step - n_hops`) as an explicit swept variable, at Étape 4's real hardened scale, using the new `--save_checkpoint_path` -- this is the version that will actually answer the question.

## 2026-09-19/20 (nuit) — I6: linear phase probe on the register -- works, but mostly explained by scale drift (control_c), as re-framed after spec Sec.7.2bis

Same checkpoint/caveat as I7 above (`n_hops=2`, `d_model=32`, toy CPU scale -- not Étape 4's real config). Re-framed mid-night: after the spec correction that `t` *may* be knowable via SM state (only *parameterizing* it is forbidden), a working probe is the *expected* outcome, not the interesting one -- the weight goes to control (c): does a single scalar (`||R_t||`) already explain the predictability, meaning "phase" is just scale drift rather than real multidimensional structure?

`diagnose_register_phase_probe.py`, 512 eval episodes x `N_STEP=16`, multiclass logistic regression, 5 probe seeds, chance=6.25%:

| probe | mean accuracy | read |
|---|---|---|
| main (`R_t` (32-d) -> `t`) | **36.9%** (std 0.5%) | clearly above chance |
| control_a (shuffled `t` labels) | 6.1% | at chance, as required -- probe isn't overfitting |
| control_b (frozen `R_0` repeated `N_STEP` times) | 2.5% | at/below chance, as required -- not exploiting a fixed per-example signature |
| **control_c (`\|\|R_t\|\|` alone, 1 scalar) -> `t`** | **48.9%** (std 0.8%) | **higher than the full 32-d probe** |

Confusion matrix (main probe) is banded near the diagonal, not scattered -- 38.8% of all errors are adjacent-step confusions (`|true-pred|==1`), consistent with a smooth/progressive trajectory rather than discrete jumps between phases.

**Read**: both trivial controls pass cleanly (a and b at/below chance, ruling out overfitting or a fixed-signature artifact), so the main result is real. But **control (c) alone (the register's norm, a single number) predicts `t` even better than the full 32-dimensional register state** -- per the decision table's third branch, this means **"phase" here is substantially, and possibly entirely, a scale-drift signature, not evidence of rich multidimensional phase structure**. Reporting this honestly as the weaker-but-real result the table calls for, not inflating it: the register's norm growing (or otherwise moving) systematically with `t` is enough on its own to nearly match the full-vector probe's performance, so the interesting direction/content structure the main probe might add beyond norm is small at this scale, if present at all. Consistent with the norm-growth pattern already seen elsewhere tonight (A1's `||k_attn||` also grew markedly with depth, a different mechanism but the same general "activations scale up over repeated/deep computation" motif worth keeping in mind project-wide).

**Same scale caveat as I7**: small CPU checkpoint, not Étape 4's real config -- queued for rerun at real scale alongside I7 once a checkpoint exists there, so both mechanistic reads (N_step generalization + phase-vs-norm) land on the same, thesis-relevant model.

**[Note, `model-design`]** the full-probe-underperforms-the-norm-alone result (36.9% < 48.9%) is not a contradiction, it's expected: `||R_t||` is a **nonlinear** function of `R_t` (a norm, not a linear projection), so a *linear* probe on the raw 32-d vector cannot reconstruct it -- the two probes aren't nested the way "more information should never do worse" intuition suggests. Recorded so this pair of numbers doesn't read as inconsistent later.

**Design tension flagged for later, not to act on now**: if phase is mostly carried by `||R_t||` growth, then **any future normalization of the register (an RMSNorm on `R_t`, or a norm-control on memory writes) would destroy this step signal** -- a real architectural tradeoff to weigh explicitly before adding such normalization, not something to discover after the fact.

## 2026-09-19/20 (nuit) — I3 étape 1 complete: attention supervision resolves hard-hops chaining, but bimodal -- success rate drops sharply with LR

`i3_attnsup_lrsweep`, `train_kb_chain_attn_supervised.py --attn_supervised --supervise node`, `n_hops=3, n_distractors=5` (hardened generator), `depth=2, block_size=4, d_model=256, n_step=12`, `lr` in `{1e-4, 2e-4, 3e-4, 4e-4}` x 3 seeds, `abacus11` (2x A5000), ~15 min/cell, 12/12 done.

| lr | seed0 | seed1 | seed2 | successes |
|---|---|---|---|---|
| 1e-4 | 99.84% | **17.99%** | 99.18% | 2/3 |
| 2e-4 | **23.16%** | 99.22% | 98.32% | 2/3 |
| 3e-4 | 29.30% | 30.86% | 98.77% | 1/3 |
| 4e-4 | 26.74% | 24.43% | 28.22% | 0/3 |

**Read**: every cell lands cleanly in one of two bands -- resolved (~98-100% acc) or collapsed (~18-31%, near the failed/no-supervision range) -- no intermediate outcomes, a textbook bimodal signature already seen elsewhere in this project (`pool_n_head`, `kdim128_decoupled`). **The success RATE, not just the absolute performance level, degrades monotonically with `lr`**: 2/3 at `1e-4` and `2e-4`, 1/3 at `3e-4`, 0/3 at `4e-4`. Qualitatively, this answers the open question I3 was launched for: **attention supervision does resolve hard-hops chaining (`n_hops=3`) when it lands in the right basin** -- it is not architecturally incapable at this scale -- but **no LR tested here is reliable** (best is 2/3 seeds, i.e. a 1/3 failure rate even at the safest point tested).

**Per the standing rule (bimodality found -> needs >=5 seeds to trust a cell, not 3)**: this result on its own cannot yet support "supervision reliably beats no-supervision at `n_hops=3`" for étape 2's comparison -- 1/3 to 2/3 seeds succeeding is exactly the regime that rule exists to catch. **À arbitrer**: whether to (a) extend the LR sweep further down (`3e-5`, `5e-5` -- the failure rate trend suggests lower might be safer, matching this project's repeated "narrow, low, non-monotone LR window" pattern) before committing étape 2's LR, or (b) accept the current best (`1e-4`/`2e-4`, both 2/3) and run étape 2 with enough seeds (>=5, ideally more given the observed 1/3 failure rate) to average over the bimodality honestly. Proceeding with **(a) first** since it's cheap and GPU is available -- extended-low-LR cells queued on `abacus11`.

**[Confirmed direction + stopping rule, `model-design`]**: continuing the low-LR extension is the right call -- a monotonically-decreasing success rate as `lr` increases is the same signature `pool_n_head`'s fine sweep showed before it eventually found a narrow stable window, so the window is plausibly below `1e-4`. **Stopping rule, to bound the search**: if no `lr` in the extended sweep reaches `>=4/5` seeds succeeding, stop looking and report the **success rate itself** as the finding ("supervision resolves hardened `n_hops=3` in k/n seeds, cleanly bimodal, no LR window tested is fully reliable") rather than continuing to hunt for a perfect window -- that is a real, publishable result on its own, an indefinite hunt is not, and it costs GPU needed elsewhere.

**Also: don't lose the actual question I3 was launched to answer.** The bimodality measured so far is in the **supervised arm alone** -- it says nothing yet about redundant-vs-necessary without the **unsupervised arm at the same LRs**. Étape 2 must run **both arms** (`--attn_supervised` on and off) at the same `lr` values and seed count, and compare **success RATES** (fraction of seeds resolving the task), not mean accuracies -- a mean across a bimodal distribution is not a meaningful summary statistic here.

**Extension complete (26/26), stopping rule triggered -- reporting the success rate, not chasing a perfect window further**:

| lr | 2e-5 | 5e-5 | **1e-4** | **2e-4** | 3e-4 | 4e-4 |
|---|---|---|---|---|---|---|
| n seeds | 5 | 5 | 5 | 5 | 3 | 3 |
| success rate | **0/5** | 0/5 | **2/5 (40%)** | **2/5 (40%)** | 1/3 | 0/3 |

**Correction to the "lower is safer" hypothesis this extension was launched to test**: it's falsified -- `2e-5` and `5e-5` (the newly-added, lower values) do *worse* than `1e-4`/`2e-4`, not better (0/5 success at both, versus 40% at the window's center). The success rate actually peaks at `1e-4`-`2e-4` and degrades on **both** sides -- a genuinely bracketed window this time, not an open-ended search, but its peak reliability is only 40%. **No `lr` reached the `>=4/5` bar** -- per the stopping rule above, halting the LR hunt here rather than probing further (e.g. `1.5e-4`) and reporting this as the finding: **hardened `n_hops=3` attention supervision resolves the task in at most 2/5 seeds at its best LR window (`1e-4`-`2e-4`), cleanly bimodal (resolved ~98-100% or collapsed ~18-31%, nothing between), with no LR tested giving reliable success.** This is the number to carry into étape 2, not a single "best" accuracy.
