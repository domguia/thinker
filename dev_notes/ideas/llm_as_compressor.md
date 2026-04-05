# Compressor LLM

the ideas is to use an LLM to compress knowleged. the aaproach will be using prompt that trigger generation of an expected text.
that prompt could be get via gradient descent on given sequence on token given as prompt.
the compression ratio = prompt token lenght / ouput token ratio.
if we observe that getting 100% accuracy is difficult (or requirequired too much token),
we could fill the gap in the prediction by adding addtiontional information to fix prediction
eg. give order of the right token for the ones where the right token is not the first (the highest probability) in prediction list.
this additionnal information should consider in the compression ratio.
how determining the right number token can be done by just trying with different token number and see how it's perform,
or look for more efficient approach so that we can speed up experiments.

we could try with try with small, and medium model (avoid large due to lack of compute).

# experiment notes
1. we should add this properly into ./thinker/ project (take time to understand it's structure before), as part of thinker it will help to analyse how compute and data could be splitted in an LLM
2. we prepare and test all the code on this pc (with cpu and same conda env used for thinker)
3. tiny experiment can be run on this pc with cpu, prepare notebook to run futher experiment on google colab
4. how to choose properly data and model to experiment with

# research thinking
what is the litterature review on this work (deep research required)

# for more
we could analyse how compression perform out of distribution of model training, to see how that impact performance:
   from random sequence to more structured one to the one used on model traning data (eg. wikitext)
we could also train a tiny model from scratch for that specific puporse and see how it peform compare to existing model
we could finetune existing model for this specific task and see how that impact thier performance, be carefull of memorisation in model weights
to mitigate memorization in model weigths, we could use lora, use different data for training and testing
