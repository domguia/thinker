# The trained computer

We want to train a model that does numeric computation such as 1 + 2 = 3,
what do we need for computation:
1. `input`
2. `computer`
3. `memory`
4. `algorithm`, that gives the desired output

in our case `computer` & `algorithm` will be merged in the `model`
the model will be a cross-attention latent-based transformer ie. Perceiver

`memory` will be intermediates latent

the `algorithm` we decided to learn are (from easy to difficult) :
1. addition
2. multiplication
3. number factorisation

Task 3 in particular will help to test how this method performs in variable complexity. Since the last task highly relies on memory to reduce computation we will observe how the model model will use the given memory.

Check ou : [/thinker_model.py](/thinker_model.py#L373)

Based on the observed result we could re-use the same approach on Language Modeling Task following the [original ideas](https://www.figma.com/file/MNe376umkTm5iCpg9kSmcq/thinking-transformer?type=design&node-id=328-196&mode=design).
