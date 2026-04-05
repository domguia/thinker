
= Architecture
The Thinker model is a recurrent, latent-based transformer that decouples computational steps from fixed architectural depth. Unlike standard transformers, Thinker utilizes a constant number of latent vectors that iteratively "read" from a high-dimensional input and "write" to a persistent memory cache.

#figure(
  image("../../visual-explanation.png", width: 80%),
  caption: [System overview: The Thinker architecture utilizes iterative cross-attention to maintain a dynamic proof state, selectively reading from premises and writing to a persistent latent memory cache.],
)

== Formal Formulation
Let $d$ be the hidden dimension and $L$ the number of latent vectors. An input sequence $X \in \{1, dots, V\}^{T_{in}}$ is embedded into $E \in RR^{T_{in} times d}$. The model initializes a latent state $Z_0 \in RR^{L times d}$ and two empty Key-Value (KV) caches: the *input cache* $(K_"in", V_"in")$ and the *memory cache* $(K_"mem", V_"mem")$.

=== Input Encoding
The input cache is populated once at the beginning of the computation:
$ K_"in", V_"in" = "Linear"_"in" (E) $

=== Recurrence and Thinking Steps
For each thinking step $t in \{1, dots, N_"step"\}$, the model updates its state as follows:
$ Q_t = "Linear"_q ("LayerNorm"(Z_{t-1})) $
The keys and values at step $t$ are formed by concatenating the fixed input cache and the dynamic memory cache:
$ K_t &= [K_"in", K_"mem, t-1"] \ V_t &= [V_"in", V_"mem, t-1"] $
The attention output $A_t$ is computed using iterative cross-attention:
$ A_t &= "Softmax"((Q_t K_t^T) / sqrt(d)) V_t $
$ Z_t &= Z_{t-1} + "MLP"("LayerNorm"(A_t)) $

=== Memory Update
The memory cache is updated at each step by projecting the current latent state:
$ K_"mem, t" &= [K_"mem, t-1", "Linear"_"mem"(Z_t)] \ V_"mem, t" &= [V_"mem, t-1", "Linear"_"mem"(Z_t)] $
This recursive structure allows the model to access not only the static input but also its own intermediate computational results from previous steps.

== Computational Complexity
Standard transformers exhibit $O(T^2)$ complexity, making them inefficient for long sequences. Thinker reduces this bottleneck through its latent bottleneck. Each thinking step has a complexity of $O(L dot (T_"in" + t dot L))$, where $L$ is the (typically small) number of latents. For $N_"step"$ iterations, the total complexity is $O(N_"step" dot L dot T_"in" + N_"step"^2 dot L^2)$. In practice, where $L << T_"in"$, Thinker scales linearly with input length, offering significant efficiency gains over standard attention mechanisms.
