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
- [toy_model.py](/toy_model.py#L373)
- on going [experiment log](/experiment.log.md)

Based on the observed result we could re-use the same approach on Language Modeling Task following the [original ideas](https://www.figma.com/file/MNe376umkTm5iCpg9kSmcq/thinking-transformer?type=design&node-id=328-196&mode=design).

**About the model**  
The model is a cross-attention latent-based transformer (like Perceiver):
1. layer weight sharing to allow reuseable compute block
2. hidden latent vector as information passing
3. cross attention on input
4. cross attention on past latent (wider information passing)

[here's a visual](https://www.figma.com/file/MNe376umkTm5iCpg9kSmcq/thinking-transformer?type=design&node-id=328-196&mode=design)

[here's a draft of the initial idea](/ideas-draft.md)

Similar ideas:
1. Looped Transformers - [paper](https://arxiv.org/pdf/2311.12424) - [x_post](https://twitter.com/DimitrisPapail/status/1747302035077378110) - [code](https://github.com/Leiay/looped_transformer/tree/main)