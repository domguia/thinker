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
