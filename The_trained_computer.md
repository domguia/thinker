""" # python comment out ;-)

# The trained computer

We want to train a model that do computation such as 1 + 2 = 3,
what do we need for computation:
1. `input`
2. `computer`
3. `memory`
4. `algorithm`, that give the desired output

in our case `computer` & `algorithm` will be merge in the `model`
the model will be a cross-attention latent based transformers ie. Perciever

`memory` will be intermediates latents

the `algorithm` we decided to learn are (from easy to difficult) :
1. addition
2. multiplication
3. number factorisation
    for variable complexity and highly rely on memmory to reduce computation (close to think process aka LM)

Below the code draft, should be implemented properly
```python
# """