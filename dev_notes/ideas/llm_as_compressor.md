# Compressor LLM

the ideas is to use an LLM to compress knowleged. the aaproach will be using prompt that trigger generation of an expected text.
that prompt could be get via gradient descent on given sequence on token given as prompt.
the compression ratio = prompt token lenght / ouput token ratio, laten token are prefered to discrete.
if we observe that getting 100% accuracy is difficult (or required too much token),
we could fill the gap in the prediction by adding addtiontional information to fix prediction
eg. give order of the right token for the ones where the right token is not the first (the highest probability) in prediction list.
this additionnal information should consider in the compression ratio.
how determining the right number token can be done by just trying with different token number and see how it's perform,
or look for more efficient approach so that we can speed up experiments.
a challenged will find highest possible compression ration of each compressed data.

we could try with try with small, and medium model (avoid large due to lack of compute).

the goal is to see how transformer (llm) can act as cpu for data compression (case of text) but could be extend to more modalities, this could help improve our undersading for thinker

# experiment notes
1. we should add this properly into ./thinker/ project (take time to understand it's structure before), as part of thinker it will help to
   analyse how compute and data could be splitted in an LLM
2. we prepare and test all the code on this pc (with cpu and same conda env used for thinker)
3. tiny experiment can be run on this pc with cpu, prepare notebook to run futher experiment on google colab
4. how to choose properly data and model to experiment with

# research thinking
A comprehensive literature review has been conducted and is stored in [knowledge/llm_compressor/literature_review.md](../../knowledge/llm_compressor/literature_review.md).

**Key Findings:**
1. **Language Modeling Is Compression (arXiv:2309.10668)**: Foundational work by DeepMind proving LLMs are powerful general-purpose compressors across modalities (images, audio, etc.).
2. **GReaTer (arXiv:2410.03842)**: Recent work on using numerical gradients over the reasoning chain to optimize prompts, which aligns with the project's core idea.
3. **LLMZip & LM-GC**: Practical implementations of LLM-based lossless compression for text and neural network gradients.

# for more
we could analyse how compression perform out of distribution of model training, to see how that impact performance:
   from random sequence to more structured one to the one used on model traning data (eg. wikitext)
we could analyse how data lenght impact compression
we could also train a tiny model from scratch for that specific puporse and see how it peform compare to existing model
we could finetune existing model for this specific task and see how that impact thier performance, be carefull of memorisation in model weights
to mitigate memorization in model weigths, we could use lora, use different data for training and testing
link to thinker (full architecture) : 
- we could make latent in such a way that the first latent encode most important key informations and more latent make decoding more accurate by adding more nkowledge, I am still lookin for how to do it eg. masking gradient descent on latent, espicially the lastest latent. I'm open to better approach
- we could train a model to be a compressor and avoid using gradient descent
- we could use that compressing to ingest nkowdlege
