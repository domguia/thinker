
#set page(paper: "a4", margin: 2cm)
#set text(size: 11pt)
#set heading(numbering: "1.")

#let abstract(body) = {
  align(center)[
    #set text(weight: "bold")
    Abstract
  ]
  v(0.5em)
  body
}

#align(center)[
  #text(size: 18pt, weight: "bold")[Thinker: A Recurrent Cross-Attention Transformer for Algorithmic Reasoning] \
  #v(1em)
  #text(size: 12pt)[Drafted by Gemini CLI Agent] \
  #text(size: 10pt)[March 26, 2026]
]

#v(2em)

#abstract[
Modern Large Language Models (LLMs) often struggle with multi-step algorithmic reasoning, typically requiring explicit "Chain-of-Thought" prompting to perform complex computations. We propose *Thinker*, a recurrent, latent-based transformer architecture designed to internalize computational steps within a hidden "thinking" process. By employing weight sharing across processing steps and utilizing cross-attention over input and past latent states (memory), Thinker achieves a flexible computational budget that can be scaled at inference time. We demonstrate the model's effectiveness on various algorithmic tasks, including multi-digit addition and sequence manipulation. Our results show that the model not only learns the required algorithms but also exhibits immediate generalization to increased computational capacity after training.
]

#v(1em)

= Introduction
The standard transformer architecture processes input in a fixed number of layers, regardless of the task's complexity. For algorithmic tasks like long division or number factorization, the required computation scales with the input size or task difficulty. To address this, we introduce the *Thinker* model, which separates the reasoning process from the fixed architectural depth through recurrence.

Inspired by architectures like Perceiver [1] and Universal Transformers [2], Thinker uses a shared processing block that iterates over a set of latent vectors. This hidden state serves as a "working memory" where the model can perform multiple steps of computation before producing an output.

= Architecture

#figure(
  image("../visual-explanation.png", width: 80%),
  caption: [Visual explanation of the Thinker model architecture, showing the interaction between input, latent state, and processing blocks.],
)

== Core Components
The Thinker model consists of four primary components:
1. *Input Encoding:* Inputs are embedded and stored in a Key-Value (KV) cache.
2. *Latent State:* A set of learned latent vectors that represent the model's internal state.
3. *Processing Block:* A shared transformer-style layer (cross-attention + MLP) that updates the latent state.
4. *Memory Cache:* A KV cache that stores previous latent states, allowing the model to look back at its own "thoughts."

== The Thinking Process
At each step $t$, the model updates its latent state $L_t$ by attending to the input cache $K_I, V_I$ and the memory cache $K_M, V_M$:
$ L_(t+1) = "ProcessBlock"(L_t, [K_I, K_M], [V_I, V_M]) $

The shared weights across steps allow the model to learn a general "thinking" operator that can be applied $N$ times, where $N$ can vary depending on the task's complexity.

== Cross-Attention Latent Flow
Unlike standard transformers that attend to all previous tokens at each layer, Thinker maintains a constant number of latents that cross-attend to the input. This significantly reduces the computational complexity relative to input length while allowing the model to selectively "read" relevant parts of the input when needed.

= Training Methodology

== Multi-Task Algorithmic Learning
We trained Thinker on a suite of algorithmic tasks of increasing difficulty:
- *Sequence Copy/Flip/Roll:* Basic manipulation of numerical sequences.
- *Multi-digit Addition:* Performing arithmetic in various bases (e.g., base-16).
- *Cumulative Sum (Cumsum):* Learning to track and update running totals.
- *Factorization (Planned):* Testing memory usage for high-complexity tasks.

== Curriculum Learning
Directly training on long sequences or complex tasks often leads to optimization plateaus. We implemented a curriculum learning strategy where the task difficulty (e.g., sequence length, number magnitude) progressively increases based on the model's current accuracy. This "challenge factor" ensures the model always receives a useful training signal.

== Scaled Loss and Gradient Clipping
To stabilize training across multiple recurrent steps, we employed gradient clipping and a best-model-restart strategy. We also experimented with scaled losses over steps to encourage early convergence while maintaining late-step accuracy.

= Experiments and Results

== Copy and Manipulation Tasks
Initial experiments on the copy task revealed that the model initially struggles with long sequences but quickly converges once it "discovers" the copy algorithm. Curriculum learning was essential for reaching 100% accuracy on sequences of length 40+.

== Addition and Cumsum
The model reached over 97% accuracy on base-16 addition for numbers up to 1000. Interestingly, we observed that increasing the number of processing steps ($N_"step"$) and latent size ($N_"latent"$) at inference time, beyond what was seen during training, often led to immediate performance improvements without further fine-tuning.

== "Ca Marche": Emergent Generalization
A key finding in our "Sept 18" experiment was the model's ability to generalize to higher compute capacities. After training with a maximum of 12 steps, increasing the step count to 16 and latent size to 24 resulted in an immediate jump in accuracy and a smoother convergence curve. This suggests the model learns a robust computational primitive that scales with available resources.

= Discussion
The Thinker architecture demonstrates that "thinking" can be viewed as a differentiable process of latent refinement. By internalizing the computational steps, the model avoids the verbosity of Chain-of-Thought while retaining its benefits. The use of a shared block and flexible latent memory provides a path toward models that can adapt their "effort" to the difficulty of the task at hand.

= Conclusion
We presented Thinker, a recurrent latent-based transformer for internalized algorithmic reasoning. Through cross-attention, shared weights, and curriculum learning, the model successfully learns to perform complex numerical computations. Future work will explore applying this architecture to large-scale language modeling and multi-modal tasks, where the ability to dynamically allocate computation could lead to more efficient and capable AI systems.

#v(2em)

= References
- [1] Jaegle, A., et al. (2021). Perceiver: General Perception with Iterative Attention. *arXiv:2103.03206*.
- [2] Dehghani, M., et al. (2018). Universal Transformers. *arXiv:1807.03819*.
- [3] Bansal, A., et al. (2022). End-to-end Algorithm Synthesis with Recurrent Networks. *arXiv:2202.05826*.
- [4] Schuurmans, D. (2023). Memory Augmented Large Language Models are Computationally Universal. *arXiv:2301.04589*.
