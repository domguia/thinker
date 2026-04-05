= Discussion

== Internalized vs. Externalized Reasoning
A central debate in AI reasoning is whether computational steps should be externalized (as in Chain-of-Thought prompting) or internalized within hidden states. While CoT provides interpretability and human-readable proofs, it is limited by the overhead of token generation and the fixed sequence length of the underlying model. Internalized reasoning, as implemented in Thinker, allows for much deeper, high-frequency "thinking" that can explore thousands of potential logic paths in a latent space before committing to a symbolic output.

== Scaling Through Recurrence
Our findings on the "Ca Marche" effect suggest that for complex reasoning tasks, scaling the *computational budget* ($N_"step"$) is as important as scaling the *parameter count*. This has profound implications for hardware design, moving toward architectures that prioritize high-bandwidth latent updates over massive feed-forward weights.

== Towards Formal Proof Integration
The ultimate goal of this research is *improving AI model reasoning via theorem proving*. We have demonstrated that Thinker can learn the computational primitives required for such tasks. The next logical step is integrating the model with formal proof kernels like Lean 4 [28] or Coq [29].

=== Tactic Selection as Recurrent Transitions
In formal theorem proving, a "proof state" is transformed into subgoals through tactics. Thinker's latent state $Z_t$ can be trained to represent this proof state. Each thinking step $t$ would then correspond to the internal search for a valid tactic. By utilizing feedback from the proof assistant—which we've seen in recent models like DeepSeek-Prover-V1.5 [14] and frameworks like LeanDojo [22]—the model can learn to prune invalid proof paths in its latent space. This process can be further optimized by integrating Hypertree Proof Search [21] or RL environments like Lean-Gym [23].

=== Neuro-Symbolic Hybridization
While Thinker's latent space handles the "intuition" of the proof (selecting promising directions), the final symbolic output must still be verified by a formal kernel. Our architecture provides a bridge: a neural engine capable of deep, iterative exploration, constrained by the rigorous requirements of formal logic. To foster this exploration, we can employ techniques like intrinsic motivation [27] or imitation learning via DAgger [30], allowing the model to learn from its own successful and failed proof attempts. This hybridization is the most promising path toward AI systems that are not only capable of complex reasoning but also guaranteed to be correct—a necessity in fields ranging from mathematics to software verification.
