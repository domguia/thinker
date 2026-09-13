# Future Experiments for Thinker Paper

**§2's "Memory Cache Importance" and "Input Re-reading" items are now an active chantier, not just a wishlist entry** — see `dev_notes/toy_memory_experiment_plan.md` (2026-09-13): `core/toy_model.py`'s two existing runners hard-code `read_step = n_step - 1`, so the input was always re-readable at every compute step except the last, meaning neither ablation below was ever actually tested by any result obtained so far. Exp. 0/1 there implement exactly these two ablations with a proper eval methodology.

To reach publication standards (e.g., NeurIPS, ICLR), the following experiments are required to validate the architectural claims and provide a rigorous empirical foundation.

## 1. Baselines Comparison
Compare **Thinker** against standard architectures on the same algorithmic tasks (Copy, Addition, Sort):
*   **Standard Transformer (GPT-style)**: Fixed depth (e.g., 6 layers).
*   **Universal Transformer (UT)**: Recurrent depth with the same number of parameters as Thinker but attending to the full sequence.
*   **Perceiver AR**: To see the impact of the latent bottleneck vs. standard autoregressive attention.

## 2. Ablation Studies
*   **Memory Cache Importance**: Train a version of Thinker where the model can only update its current latent state $Z_t$ but cannot attend to past latents $K_M, V_M$. This tests if the "medium-term memory" is actually utilized.
*   **Input Re-reading**: Compare "read once at the beginning" vs. "read input at every thinking step."
*   **Latent Size Ablation**: How does performance scale with $L$ (latent size) for the same task?

## 3. Extrapolation & Generalization
*   **Sequence Length Extrapolation**: Train on sequences of length 20. Test on lengths 40, 80, and 160.
*   **Step Extrapolation**: Demonstrate that increasing $N_{step}$ at inference time continues to improve accuracy for complex tasks (e.g., factorization) beyond the steps seen during training.

## 4. Interpretability
*   **Attention Heatmaps**: Plot cross-attention maps for the "Addition" task. Does the model attend to the corresponding digits and the "carry" bit at appropriate steps?
*   **Latent Trajectory Analysis**: Use PCA/t-SNE to visualize the latent state evolution $Z_t \to Z_{t+1}$ across steps. Show that the state converges to a "solution manifold."

## 5. Computational Efficiency
*   **FLOPs vs. Accuracy**: Plot the computational cost (FLOPs) vs. Accuracy. Show that Thinker is more efficient than a standard Transformer for long sequences due to the $O(L \cdot T)$ cross-attention complexity.
