# Experiment Log

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

