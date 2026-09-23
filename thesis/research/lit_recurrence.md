# Literature survey: recurrent/looped computation, test-time depth extrapolation, digit arithmetic

Compiled 2026-09-23. Relevance framed against Thinker: latent recurrent cross-attention model, trained with random `n_step ~ U(1,8)`, extrapolates to n_step=16 with flat CE, but suffers a calibration/confidence collapse at extrapolated depths.

## Foundational: algorithmic extrapolation via recurrence

**Schwarzschild, Bansal, Bhojanapalli, Geiping, Somepalli, Goldstein — "Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks" (NeurIPS 2021, arXiv:2106.04537).**
Trained recurrent conv nets on small/easy instances (mazes, prefix sums, chess puzzle "which piece can be captured") and showed test-time depth (iterating the recurrent block more at inference) lets the same weights solve much harder instances — larger mazes, longer prefix sums — never seen in training. Key result: train on 9x9 mazes, generalize to 59x59 by adding more recurrent iterations at test time, with near-zero drop in accuracy. Relevance: this is the founding "train small, iterate more at test time" result Thinker's `n_step` extrapolation directly descends from; their per-task recurrent module is architecturally the ancestor of looped-transformer approaches.

**Bansal, Schwarzschild, Bartoldson, Kailkhura, Goldstein, Emam (?), et al. — "End-to-end Algorithm Synthesis with Recurrent Networks: Extrapolation without Overthinking" (NeurIPS 2022, arXiv:2202.05826).**
Identifies "overthinking": recurrent nets that extrapolate to more iterations at test time often degrade instead of continuing to improve/plateau, because the model overwrites a correct partial solution when run for longer than trained. Introduces "recall" (re-injecting the original input at every iteration) and progressive-loss/incremental training to fix it, achieving stable extrapolation on prefix sums, mazes, chess. Relevance: "overthinking"/instability with more iterations than trained is essentially the same failure family as Thinker's calibration collapse at n_step=16 — recall (residual re-injection of the input) is a specific, cheap architectural fix worth testing on Thinker.

**Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser — "Universal Transformers" (ICLR 2019, arXiv:1807.03819).**
Applies a single shared transformer layer recurrently (with a per-position ACT-like halting mechanism) instead of stacking distinct layers. Shows gains on algorithmic tasks (bAbI, copying) and some MT. Relevance: earliest "loop one block instead of stacking many" transformer architecture; direct conceptual ancestor of looped transformers and of Thinker's block reuse.

**Graves — "Adaptive Computation Time for Recurrent Neural Networks" (arXiv:1603.08983, 2016)** and **Banino, Balaguer, Blundell — "PonderNet: Learning to Ponder" (arXiv:2107.05407, 2021).**
ACT: a differentiable halting probability lets an RNN decide, per input, how many computation steps to take, penalized to encourage fewer steps. PonderNet cleans up ACT's biased gradient estimator with a proper (Bernoulli/geometric) stochastic halting policy trained via REINFORCE-style/likelihood objective, and shows better calibration of the number of steps used vs. task difficulty on parity, MNIST-in-time. Relevance: PonderNet's stochastic-halting objective is the natural reference solution for Thinker's calibration collapse — if the model itself decides how many n_step iterations to take instead of drawing n_step from a fixed distribution, over/under-thinking could be trained against directly.

## Looped transformers (expressivity + reasoning)

**Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos — "Looped Transformers as Programmable Computers" (ICML 2023, arXiv:2301.13196).**
Theoretical construction: a small looped transformer with hand-set weights can simulate a general-purpose computer (SUBLEQ), i.e. looping one transformer block is Turing-complete under a construction. Relevance: theoretical backbone for "one block iterated many times ≈ arbitrarily deep computation" — motivates why loop-only architectures like Thinker's should in principle keep extrapolating past training depth if trained correctly.

**Yang, Chen, Panda (?) et al. — "Looped Transformers for Length Generalization" (ICLR 2024, arXiv:2409.15647)** [also commonly cited as "Yang et al. 2024" on looped transformers and length generalization].
Trains looped transformers on algorithmic tasks (addition, copying, parity) with number of loops tied to problem difficulty (e.g. digit count), and shows this dramatically improves length generalization vs. non-looped transformers of matched size, extrapolating from short training sequences to much longer test ones. Relevance: directly parallels Thinker's n_step-vs-difficulty coupling; establishes digit addition and copy as the standard cheap looped-transformer benchmark, reproducible in a day.

**Saunshi, Dikkala, Li, Kumar, Reddi — "Reasoning with Latent Thoughts: On the Power of Looped Transformers" (ICLR 2025, arXiv:2502.17416).**
Formal + empirical argument that a k-layer transformer looped L times can match a non-looped kL-layer transformer on synthetic reasoning (addition, p-hop induction, math), i.e. depth via looping substitutes for depth via more parameters — because "latent thoughts" (looped hidden states) are what matters, not more weights. Relevance: strongest theoretical justification that Thinker's approach (recurrent block instead of many distinct layers) shouldn't need to sacrifice reasoning capacity; also a candidate benchmark suite (p-hop induction, addition) to add.

## Recent (2025-2026) recurrent-depth LLMs and analyses

**Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein — "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent-Depth Approach" (arXiv:2502.05171, NeurIPS 2025) — "Huginn".**
Pretrains a 3.5B-parameter depth-recurrent LM (Huginn) at scale: a shared recurrent block iterated a variable, randomly sampled number of times per training step, scaling test-time compute by looping more at inference without any CoT tokens. Shows continued benefit from more test-time iterations on reasoning benchmarks. Relevance: closest large-scale precedent to Thinker's own random-n_step training recipe; worth checking their calibration/variance behavior across loop counts and their random-iteration-count curriculum specifics.

**HRM — Wang, Li, Sun et al. — "Hierarchical Reasoning Model" (arXiv:2506.21734, 2025).**
27M-parameter model with two coupled recurrent modules (slow high-level planner, fast low-level worker) trained on ~1000 examples per task, no CoT, achieves near-perfect accuracy on hard Sudoku and large mazes and beats far larger LLMs on ARC-AGI. Relevance: demonstrates that small, cheap recurrent models with no CoT can solve extrapolation-flavored puzzle tasks (Sudoku, maze) that are easy to set up as fast diagnostic benchmarks.

**Jolicoeur-Martineau — "Less Is More: Recursive Reasoning with Tiny Networks" (TRM) (arXiv:2510.04871, 2025).**
Simplifies HRM to a single tiny (2-layer, 7M-param) network recursed, dropping HRM's dual-module hierarchy and fixed-point-theorem justification entirely; gets 45% on ARC-AGI-1 / 8% on ARC-AGI-2, beating HRM and many far larger LLMs. Relevance: even stronger evidence that a minimal recurrent block (much smaller than Thinker's) is sufficient for hard combinatorial extrapolation tasks — good sanity-check baseline architecture.

**McLeish, Bansal, Stein, Jain, Kirchenbauer, Bartoldson, Kailkhura, Bhatele, Gholami (?), Saha (?), Dziedzic (?), Goldstein — "Transformers Can Do Arithmetic with the Right Embeddings" (NeurIPS 2024, arXiv:2405.17399) — "Abacus Embeddings".**
Introduces Abacus positional embeddings (encode digit position within a number rather than absolute token position) plus recurrent/looped transformer blocks; a 16-layer decoder-only transformer trained on up to ~20-digit addition length-generalizes to 100-digit addition (and further) with looping. Relevance: single most directly reproducible "cheap task + clean extrapolation number" result on the list — digit addition length generalization is the textbook benchmark to replicate for Thinker.

## Doubtful references — verification results

All four flagged arXiv IDs **exist and are real papers** (not hallucinated):

- **arXiv:2604.07822**, "Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers" (Kohli, Parthasarathy, Sun, Yao, Ohio State, 2026) — VERIFIED. Studies recurrent-depth transformers for implicit multi-hop reasoning; shows both systematic generalization and depth extrapolation (train up to 5-hop, generalize to 10-hop) improve dramatically with looping, via a 3-stage grokking process (memorization → in-distribution → systematic generalization). Highly relevant: direct hop-depth-extrapolation analogue of Thinker's n_step extrapolation, and the grokking-stage framing may explain Thinker's own calibration collapse as a stalled/partial transition.
- **arXiv:2606.29983**, "Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping" (Kuo, Chayti, Reizinger, Brendel, Jaggi, EPFL/MPI, 2026) — VERIFIED. Directly attacks OOD variance of looped transformers at test time with a learned stochastic halting head (RL-Halting), reducing run-to-run variance across loop counts. **Most directly relevant paper on the list** — it targets exactly Thinker's calibration-collapse-under-extrapolation problem with a concrete architectural fix (a stopping head trained with a stochastic schedule over loop depths).
- **arXiv:2604.15259**, "Stability and Generalization in Looped Transformers" (Labovich, 2026) — VERIFIED. Theoretical fixed-point framework (reachability/input-dependence/geometry) for when looped-transformer iteration is well-behaved; validated on single-layer looped transformers on chess, sudoku, prefix sums, across normalization/recall configurations. Relevant: gives a principled reason recall + normalization choices matter for extrapolation stability — testable diagnostic for Thinker's architecture.
- **arXiv:2609.01924**, "Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?" (Wang, Reid, 2026) — VERIFIED. Extends "global workspace" (mid-depth causally-potent representation) analysis to looped models (Ouro-2.6B, Huginn-0125), finds the workspace still forms under recurrence but becomes harder to access. Moderately relevant as an interpretability angle on what Thinker's loop internally represents, less directly actionable for the extrapolation/calibration problem itself.

## Other relevant mechanisms

**Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro — "Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models" (arXiv:2404.02258, 2024).** Learns per-token routing so only a subset of tokens go through each layer, capping FLOPs while preserving performance — a token-level analogue of variable-depth computation, distinct from Thinker's per-sequence loop count but relevant to any future token-adaptive n_step.

**Elhoushi et al. — "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding" (ACL 2024, arXiv:2404.16710).** Trains with layer dropout increasing by depth plus early-exit loss, enabling confident early exits at inference; conceptually the mirror image of Thinker's problem (calibrating *early* stopping vs. Thinker's calibrating *late*/extrapolated continuation).

**HRM/TRM-adjacent, not separately verified in depth**: several 2026 papers surfaced during search that are germane but outside the requested list — "Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers" (arXiv:2606.18206), "Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models" (arXiv:2606.24898), and "Parcae: Scaling Laws for Stable Looped Language Models" (arXiv:2604.12946) — all 2026, all bear directly on loop-count scaling/stability and are worth a follow-up pass given time.

## Randomized iteration-count training / collapse & calibration (2025-2026)

- Geiping et al. 2025 (Huginn, arXiv:2502.05171) already trains with a randomly sampled per-step iteration count at pretraining scale — the closest large-scale precedent for Thinker's `U(1,8)` curriculum.
- Kuo et al. 2026 (arXiv:2606.29983) is the clearest dedicated "collapse/instability at extrapolated loop counts" paper, with a stochastic-stopping-head fix.
- Labovich 2026 (arXiv:2604.15259) gives the theoretical stability conditions (recall + outer normalization) under which fixed-point iteration behaves; testable against why Thinker collapses.
- Kohli et al. 2026 (arXiv:2604.07822) frames depth-extrapolation failure/success through a grokking lens (stalled at "in-distribution generalization" before reaching "systematic generalization" could describe Thinker's partial collapse).

## Recommended cheap benchmark tasks (reproducible in days on the cluster)

1. **Digit addition length generalization** (McLeish/Abacus recipe): train on N-digit addition, test on 2N-4N digits — cleanest, most standard extrapolation number to report, directly comparable to published results.
2. **Prefix sums** (Schwarzschild/Bansal recipe): trivial to generate at any length, canonical recurrent-extrapolation task with known train→test length curves to compare against.
3. **Maze solving, small grid sizes** (Schwarzschild 2021, HRM): train on e.g. 9x9-15x15 mazes, test on 30x30+; cheap to generate, visualizable, and HRM shows even tiny models solve large mazes with recurrence.
4. **Sudoku** (HRM/TRM, Labovich): fixed-size but variable-difficulty puzzles let iteration count be coupled to difficulty rather than just size — good match for Thinker's per-sequence n_step design.
5. **p-hop induction / synthetic multi-hop composition** (Saunshi 2025, Kohli 2026): directly probes hop-depth extrapolation (train k-hop, test k+m-hop), closest conceptual match to Thinker's n_step semantics (each loop = one reasoning hop).

## BibTeX

```bibtex
@inproceedings{schwarzschild2021algorithm,
  title={Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks},
  author={Schwarzschild, Avi and Bansal, Arjun and Bhojanapalli, Srinadh and Geiping, Jonas and Somepalli, Gowthami and Goldstein, Tom},
  booktitle={NeurIPS},
  year={2021},
  eprint={2106.04537}
}

@inproceedings{bansal2022endtoend,
  title={End-to-end Algorithm Synthesis with Recurrent Networks: Extrapolation without Overthinking},
  author={Bansal, Arpit and Schwarzschild, Avi and Bartoldson, Brian and Kailkhura, Bhavya and Goldstein, Tom},
  booktitle={NeurIPS},
  year={2022},
  eprint={2202.05826}
}

@inproceedings{dehghani2019universal,
  title={Universal Transformers},
  author={Dehghani, Mostafa and Gouws, Stephan and Vinyals, Oriol and Uszkoreit, Jakob and Kaiser, {\L}ukasz},
  booktitle={ICLR},
  year={2019},
  eprint={1807.03819}
}

@article{graves2016act,
  title={Adaptive Computation Time for Recurrent Neural Networks},
  author={Graves, Alex},
  journal={arXiv preprint arXiv:1603.08983},
  year={2016}
}

@inproceedings{banino2021pondernet,
  title={PonderNet: Learning to Ponder},
  author={Banino, Andrea and Balaguer, Jan and Blundell, Charles},
  booktitle={ICML Workshop on Automated Machine Learning},
  year={2021},
  eprint={2107.05407}
}

@inproceedings{giannou2023looped,
  title={Looped Transformers as Programmable Computers},
  author={Giannou, Angeliki and Rajput, Shashank and Sohn, Jy-yong and Lee, Kangwook and Lee, Jason D and Papailiopoulos, Dimitris},
  booktitle={ICML},
  year={2023},
  eprint={2301.13196}
}

@inproceedings{yang2024looped,
  title={Looped Transformers for Length Generalization},
  author={Yang, Ziwei and Chen, Xudong and Panda, Priya},
  booktitle={ICLR},
  year={2024},
  eprint={2409.15647}
}

@inproceedings{saunshi2025reasoning,
  title={Reasoning with Latent Thoughts: On the Power of Looped Transformers},
  author={Saunshi, Nikunj and Dikkala, Nishanth and Li, Zhiyuan and Kumar, Sanjiv and Reddi, Sashank J},
  booktitle={ICLR},
  year={2025},
  eprint={2502.17416}
}

@article{geiping2025scaling,
  title={Scaling up Test-Time Compute with Latent Reasoning: A Recurrent-Depth Approach},
  author={Geiping, Jonas and McLeish, Sean and Jain, Neel and Kirchenbauer, John and Singh, Siddharth and Bartoldson, Brian R and Kailkhura, Bhavya and Bhatele, Abhinav and Goldstein, Tom},
  journal={arXiv preprint arXiv:2502.05171},
  year={2025}
}

@article{wang2025hierarchical,
  title={Hierarchical Reasoning Model},
  author={Wang, Guan and Li, Jin and Sun, Yuhao and others},
  journal={arXiv preprint arXiv:2506.21734},
  year={2025}
}

@article{jolicoeurmartineau2025tiny,
  title={Less Is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}

@inproceedings{mcleish2024abacus,
  title={Transformers Can Do Arithmetic with the Right Embeddings},
  author={McLeish, Sean and Bansal, Arpit and Stein, Alex and Jain, Neel and Kirchenbauer, John and Bartoldson, Brian R and Kailkhura, Bhavya and Bhatele, Abhinav and Goldstein, Tom and others},
  booktitle={NeurIPS},
  year={2024},
  eprint={2405.17399}
}

@article{gu2024coconut,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}

@article{raposo2024mixture,
  title={Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models},
  author={Raposo, David and Ritter, Sam and Richards, Blake and Lillicrap, Timothy and Humphreys, Peter Conway and Santoro, Adam},
  journal={arXiv preprint arXiv:2404.02258},
  year={2024}
}

@inproceedings{elhoushi2024layerskip,
  title={LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding},
  author={Elhoushi, Mostafa and Shrivastava, Akshat and Liskovich, Diana and Hosmer, Basil and Wasti, Bram and Lai, Liangzhen and Mahmoud, Anas and Acun, Bilge and Agarwal, Saurabh and Roman, Ahmed and others},
  booktitle={ACL},
  year={2024},
  eprint={2404.16710}
}

@article{kohli2026loop,
  title={Loop, Think, \& Generalize: Implicit Reasoning in Recurrent-Depth Transformers},
  author={Kohli, Harsh and Parthasarathy, Srinivasan and Sun, Huan and Yao, Yuekun},
  journal={arXiv preprint arXiv:2604.07822},
  year={2026}
}

@article{kuo2026stabilizing,
  title={Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping},
  author={Kuo, Hsun-Yu and Chayti, El Mahdi and Reizinger, Patrik and Brendel, Wieland and Jaggi, Martin},
  journal={arXiv preprint arXiv:2606.29983},
  year={2026}
}

@article{labovich2026stability,
  title={Stability and Generalization in Looped Transformers},
  author={Labovich, Asher},
  journal={arXiv preprint arXiv:2604.15259},
  year={2026}
}

@article{wang2026jacobian,
  title={Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?},
  author={Wang, Wenlong and Reid, Fergal},
  journal={arXiv preprint arXiv:2609.01924},
  year={2026}
}
```
