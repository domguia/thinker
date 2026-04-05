
= Experiments and Results

== Copy and Manipulation Tasks
Initial experiments on the copy task revealed that the model initially struggles with long sequences but quickly converges once it "discovers" the copy algorithm. Curriculum learning was essential for reaching 100% accuracy on sequences of length 40+.

== Addition and Cumsum
The model reached over 97% accuracy on base-16 addition for numbers up to 1000. Interestingly, we observed that increasing the number of processing steps ($N_"step"$) and latent size ($N_"latent"$) at inference time, beyond what was seen during training, often led to immediate performance improvements without further fine-tuning.

== "Ca Marche": Emergent Generalization
A key finding in our "Sept 18" experiment was the model's ability to generalize to higher compute capacities. After training with a maximum of 12 steps, increasing the step count to 16 and latent size to 24 resulted in an immediate jump in accuracy and a smoother convergence curve. This suggests the model learns a robust computational primitive that scales with available resources.

#figure(
  image("../../logs/archive_exp_logs/ca_marche-perf_increase_with_compute-30k_iteration_cumsum_curve.png", width: 90%),
  caption: [The "Ca Marche" effect: Accuracy improves as the number of thinking steps and latent dimensions are increased at inference time, demonstrating the model's ability to scale its internal reasoning process.],
)

#figure(
  image("../../logs/archive_exp_logs/hp_heatmap_20dec_exp.png", width: 90%),
  caption: [Hyperparameter Evaluation Heatmap: The relationship between latent dimensions (vertical) and thinking steps (horizontal). Increased latent capacity and depth consistently lead to higher precision in the internalized reasoning process, aligning with the requirements for complex proof state management.],
)
