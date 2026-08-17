
= Training Methodology

== Multi-Task Algorithmic Learning
We trained Thinker on a suite of algorithmic tasks of increasing difficulty:
- *Sequence Copy/Flip/Roll:* Basic manipulation of numerical sequences.
- *Multi-digit Addition:* Performing arithmetic in various bases (e.g., base-16).
- *Cumulative Sum (Cumsum):* Learning to track and update running totals.
- *Factorization (Planned):* Testing memory usage for high-complexity tasks.

== Worked Example: Base-16 Addition
To make the recurrence introduced in the Architecture section concrete, we walk through a single forward pass on the base-16 addition task, reusing the same notation ($Z_t$, $K_"in"$, $K_"mem"$).

Consider adding two 3-digit base-16 numbers, e.g. $"2A7" + "1F4"$. The digit sequence (including an operator token and padding) is tokenized and embedded into $E$, from which the *static* input cache $K_"in", V_"in"$ is computed once via $"Linear"_"in"$—this cache never changes for the rest of the forward pass, and represents the two operands the model can always attend back to.

The latent state $Z_0$ is initialized (e.g. from a learned starting vector), and the memory cache starts empty. The model then iterates:
- *Step 1:* $Q_1$ attends only over $K_"in"$ (the memory cache is still empty), effectively "reading" the least-significant digits of both operands. The resulting $Z_1$ is projected into $K_"mem", V_"mem"$—it now holds an intermediate result, conceptually the sum of the least-significant digits together with a carry flag.
- *Step 2:* $Q_2$ attends over $[K_"in", K_"mem"]$, so the model can both re-read the input digits and retrieve the carry information written at step 1. This is what lets the model propagate a carry from one digit position to the next without re-deriving it from scratch.
- *Steps 3…$N_"step"$:* the same read-input/read-memory/write-memory cycle repeats, one digit position at a time, until every digit (including a possible final carry) has been processed.

After $N_"step"$ iterations, the final latent state $Z_{N_"step"}$ is decoded into the output digit sequence (here, "4 9 B"). Because $K_"mem"$ grows by exactly one entry per step, the number of steps the model needs scales with the number of digit positions, not with the input length in tokens—which is why increasing $N_"step"$ at inference time (Section 4.3) lets the same trained model handle longer numbers than it was trained on, as long as it has learned the underlying per-digit carry propagation rule rather than a fixed-length lookup.

== Curriculum Learning and State Stability
Directly training on long sequences or complex tasks often leads to optimization plateaus. We implemented a curriculum learning strategy [11] where the task difficulty progressively increases based on the model's current accuracy. This is augmented by a "best-model-restart" strategy similar to the R-max algorithm [25], which favors near-optimal exploration under sparse rewards by encouraging revisits to promising states. This restart-on-improvement approach also parallels the selective, value-guided search strategies used in Monte-Carlo Tree Search [26], which biases computation toward states most likely to improve the outcome.

To ensure that the latent state $Z_t$ remains meaningful across many steps, we apply a "progressive supervision" loss. This is conceptually similar to the "Chain-of-Thought" (CoT) prompting [19] used in externalized reasoning, but here the reasoning is internalized in the latent space. We periodically project the latent state $Z_t$ into the output space, ensuring it maintains a coherent "partial proof state" throughout the computation. This prevents the latent representation from drifting into uninterpretable noise—a key requirement for theorem proving where a proof state must remain logically consistent over many tactic applications, as seen in modular systems like Lego-Prover [20].

== Scaled Loss and Gradient Clipping
To stabilize training across multiple recurrent steps, we employed gradient clipping and a supervised fine-tuning (SFT) approach often used to align models with formal specifications [14]. We also experimented with scaled losses over steps to encourage early convergence while maintaining late-step accuracy. Our results show that the capacity of the model to solve complex problems is directly proportional to the dimensionality of the latent space $L$. By increasing $L$ at inference time—a technique inspired by Multi-head Latent Attention [8] and recent findings in neuro-symbolic geometry [31]—we can dynamically increase the "resolution" of the model's internal proof state, allowing it to handle more complex premises and intermediate subgoals without retraining.
