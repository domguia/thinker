= Literature Review

== Recurrent Architectures and Universal Transformers
The quest for computational universality in neural networks began with Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks. The introduction of the Transformer architecture by Vaswani et al. (Google Research) [6] revolutionized the field but introduced a fixed architectural depth. Dehghani et al. (Google Brain) [1], in their seminal ICLR 2019 paper, addressed this by applying the same transformer block recurrently. Recent work by Bulatov et al. [7] on Recurrent Memory Transformers (NeurIPS 2022) and the theoretical analysis by Giannou et al. [2] have further refined this, showing that a single looped transformer layer—as detailed in their ICLR 2024 publication—can simulate a Turing-complete computer.

== Latent Bottlenecks and Efficiency
Processing long sequences in transformers is computationally expensive due to self-attention complexity. Jaegle et al. (DeepMind) [3, 4] introduced latent bottlenecks in the Perceiver family, reducing complexity to $O(L dot T)$. This paradigm is essential for theorem proving, where the model must maintain a compact "proof state." More recently, the DeepSeek-AI team [8] proposed Multi-head Latent Attention (MLA) to compress Key-Value caches, achieving over 90% memory reduction.

== Algorithmic Reasoning and Extrapolation
Algorithmic reasoning requires models to generalize beyond their training distribution. Bansal, Schwarzschild, and the Maryland team [5, 11] identified the "overthinking" problem in recurrent systems. The "Ca Marche" effect observed in our work aligns with the emergent reasoning capabilities of large-scale models like DeepSeek-R1 [12].

== AI for Formal Theorem Proving
The integration of LLMs with formal proof assistants (Lean, Coq) is a primary frontier. Early systems like GPT-f by Polu and Han (OpenAI) [13], presented at NeurIPS 2020, used transformers to suggest tactics. Modern approaches by Xin et al. (DeepSeek-AI) [14] and Trinh et al. (Google DeepMind) [15] combine large-scale pre-training with Reinforcement Learning. DeepSeekMath [24] introduced Group Relative Policy Optimization (GRPO) to optimize reasoning. Furthermore, systems like TongGeometry by Zhang et al. [31] (Nature Machine Intelligence 2025) demonstrate that guided tree search—a strategy popularized by Silver et al. [16]—can outperform IMO gold medalists.

== Reasoning via Search and Scaling Laws
The "OpenAI o1" and "DeepSeek-R1" [12] models have demonstrated that reasoning can be scaled at inference time. This "Inference-time Scaling Law," explored in COLM 2024 by Zelikman et al. [17] (Quiet-STaR), suggests that spending more compute on "thinking" is more effective than increasing model size. Recent studies on AlphaGeometry 2 [32] and Goedel-Prover by Liu et al. [33] further emphasize this transition toward search-based formal mathematics.
