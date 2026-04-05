# Thinker
The trained computer

We want to train a model that does numeric computation such as 1 + 2 = 3,
what do we need for computation:
1. `input`
2. reusable `computer` unit -> repeat transformer block
3. `memory` -> concatenated embeddings accessed via cross attention
4. `algorithm`, that gives the desired `output`

in our case `computer` & `algorithm` will be merged in the `model`  
`memory` will be intermediates latent states concatenated

the `algorithm` we decided to learn are (from easy to difficult) : 
1. copy input to output, with some variations
2. addition
3. multiplication
4. number factorisation

Task 3 in particular will help to test how this method performs in variable complexity. Since the last task highly relies on memory to reduce computation we will observe how the model model will use the given memory.

Checkout :
- [toy_model.py](core/toy_model.py)
- on going [experiment log](dev_notes/experiment.log.md)
- [LLM as Data Compressor](core/compressor/)

## LLM as Data Compressor
This sub-project explores using LLMs to compress data by finding optimal prompts.
- **Goal:** Achieve high compression ratios by storing only a short prompt and correction ranks.
- **Environment:** Use the `thinker` conda environment.
- **Usage:**
  ```bash
  export PYTHONPATH=$PYTHONPATH:.
  conda run -n thinker python scripts/compress_demo.py --text "Your text here" --n_prompt 5 --n_steps 100
  ```
- **Experiment Skill:** A project-specific skill `compressor-experiment` is available to automate and log experiments. Use it to run batch trials or systematic tests.
- **Dependencies:** `transformers`, `accelerate`, `torch` (CPU version recommended for local development).

Based on the observed result we could re-use the same approach on Language Modeling Task following the [original ideas](https://www.figma.com/file/MNe376umkTm5iCpg9kSmcq/thinking-transformer?type=design&node-id=328-196&mode=design).

**About the model**  
The model is a cross-attention latent-based transformer (like Perceiver):
1. layer weight sharing to allow reuseable compute block
2. hidden latent vector as information passing
3. cross attention on input
4. cross attention on past latent (wider information passing)

[here's a visual](https://www.figma.com/file/MNe376umkTm5iCpg9kSmcq/thinking-transformer?type=design&node-id=328-196&mode=design)

![visual explanation](visual-explanation.svg)

[here's a draft of the initial idea](dev_notes/ideas/ideas-draft.md)

## Project Structure

```text
thinker/
├── core/                       # 🧠 Core architecture and tools
│   ├── models.py               # Main model configurations
│   ├── toy_model.py            # Primary ToyThinker model
│   ├── layers.py               # Basic model layers (SwiGLU, RMSNorm, FlexDecoderLayer, RoPE)
│   └── utils.py                # Core utilities (e.g. CfgNode)
├── data/                       # 🗃️ Datasets and curriculum logic
│   └── numbers.py              # Generative/Curriculum datasets
├── scripts/                    # 🚀 Entrypoint scripts for execution
│   ├── train.py                # Setup for automated (15 min budget) autoresearch training
│   ├── th1nker_runner.py       # Standard runner
│   ├── run_lightning.py        # Lightning-based runner
│   ├── generate_embeddings.py  # Utility scripts
│   └── visualize.py            # Log visualization
├── notebooks/                  # 📓 Rapid prototyping and Colab entrypoint
│   └── Th1nker_runner.ipynb
├── dev_notes/                  # 📝 Development notes, DB structures, past experiments
│   ├── ideas/                  
│   └── experiment.log.md
├── docs/                       # 📚 Documentation
│   └── ToDo.md
├── inspirations/               # 💡 External references (Autoresearch, AdderBoard)
└── program.md                  # 🤖 System instructions for automated research agents
```

Similar ideas:
1. Looped Transformers - [paper](https://arxiv.org/pdf/2311.12424) - [x_post](https://twitter.com/DimitrisPapail/status/1747302035077378110) - [code](https://github.com/Leiay/looped_transformer/tree/main)