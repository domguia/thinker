# Thinker
The trained computer

We want to train a model that does numeric computation such as 1 + 2 = 3,
what do we need for computation:
1. `input`
2. `computer`
3. `memory`
4. `algorithm`, that gives the desired output

in our case `computer` & `algorithm` will be merged in the `model`  
`memory` will be intermediates latent

the `algorithm` we decided to learn are (from easy to difficult) :
1. addition
2. multiplication
3. number factorisation

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