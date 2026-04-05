
#let abstract(body) = {
  align(center)[
    #set text(weight: "bold")
    Abstract
  ]
  v(0.5em)
  body
}

#abstract[
Modern Large Language Models (LLMs) often struggle with multi-step algorithmic reasoning and formal logic, typically requiring explicit "Chain-of-Thought" prompting to perform complex computations. We propose *Thinker*, a recurrent, latent-based transformer architecture designed to internalize computational steps within a hidden "thinking" process, providing a robust foundation for automated reasoning and theorem proving. By employing weight sharing across processing steps and utilizing iterative cross-attention over input and past latent states, Thinker maintains a stable representation of the "reasoning state." This enables the model to achieve algorithmic extrapolation, solving complex problems by expending more computational effort at inference time. We demonstrate Thinker's effectiveness on various computational tasks, showing that it can synthesize strategies that scale to problems of higher complexity than those seen during training—a critical requirement for deep mathematical reasoning and formal verification.
]
