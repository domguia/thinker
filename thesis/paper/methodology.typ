
= Training Methodology

== Multi-Task Algorithmic Learning
We trained Thinker on a suite of algorithmic tasks of increasing difficulty:
- *Sequence Copy/Flip/Roll:* Basic manipulation of numerical sequences.
- *Multi-digit Addition:* Performing arithmetic in various bases (e.g., base-16).
- *Cumulative Sum (Cumsum):* Learning to track and update running totals.
- *Factorization (Planned):* Testing memory usage for high-complexity tasks.

== Curriculum Learning and State Stability
Directly training on long sequences or complex tasks often leads to optimization plateaus. We implemented a curriculum learning strategy [11] where the task difficulty progressively increases based on the model's current accuracy. This is augmented by a "best-model-restart" strategy similar to the R-max algorithm [25] used in reinforcement learning to handle sparse rewards in proof search [26].

To ensure that the latent state $Z_t$ remains meaningful across many steps, we apply a "progressive supervision" loss. This is conceptually similar to the "Chain-of-Thought" (CoT) prompting [19] used in externalized reasoning, but here the reasoning is internalized in the latent space. We periodically project the latent state $Z_t$ into the output space, ensuring it maintains a coherent "partial proof state" throughout the computation. This prevents the latent representation from drifting into uninterpretable noise—a key requirement for theorem proving where a proof state must remain logically consistent over many tactic applications, as seen in modular systems like Lego-Prover [20].

== Scaled Loss and Gradient Clipping
To stabilize training across multiple recurrent steps, we employed gradient clipping and a supervised fine-tuning (SFT) approach often used to align models with formal specifications [14]. We also experimented with scaled losses over steps to encourage early convergence while maintaining late-step accuracy. Our results show that the capacity of the model to solve complex problems is directly proportional to the dimensionality of the latent space $L$. By increasing $L$ at inference time—a technique inspired by Multi-head Latent Attention [8] and recent findings in neuro-symbolic geometry [31]—we can dynamically increase the "resolution" of the model's internal proof state, allowing it to handle more complex premises and intermediate subgoals without retraining.
