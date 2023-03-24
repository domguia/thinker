# Th1nker model

Why?
Because AI today is not smart enough, for lazy guys like me

What?
Train a model to learn a thinking process steps

How? by :
giving steps in the process
2. splitting knowledge than think process
3. by giving process mechanism to build/access knowledge

-- ideas : rewrite a text in the format "question? answer. question? answer. " and use that text to teach a model to follow instructions with just LLMM standard AR and avoid complex RL

## In details
the thinker can be a transformer base model, made with building blocks responsible for each task in the process
[detailed visual architecture](https://www.figma.com/file/MNe376umkTm5iCpg9kSmcq/thinking-transformer?node-id=328-196&t=fIFQ60I3hcz39A4t-0)

### Blocks
Here are the proposed building blocks:
1. the step process is given by calling during inference:
    1. ReadLayer : read any kind of information
    2. ProcessLayer : process information, always the same
    3. OutputLayer : get output from
    4. Probe (optional) : predicting the block that will be used in the next thinking step
2. processing memory (or short-term memory)
    1. Latent : is as input in all layers mentioned before, it also the initial input of the model
    2. Input : is the input data we want to process, we may have TextInput, ImageInput ... , InputText + Latent -> ReadTextInputLayer
    3. Ouput : is the expected output, optionally autoregressively (output probe)
3. medium-term memory component
    1. LatentStore : we store latent at each processing step, they will be read at inference with ReadLatentLayer
    2. 
4. long-term memory : is also built with latent but of the past run of the model
    1. it will be probably very large and need an indexer to get the useful one
    2. ReadLatentLayer here will be associated with a learned Indexer
    3. 

### Inference


### Training
1. learn how to do step:  
    just train in forward pass-over step with random initial latent,  
    re-read input at each step, to avoid model divergence
    get output at each step a give the loss training signal
    for autoregressive task  
    back-propagrade over many steps, use stop gradient after some step for the model to generalize how to handle/denoise any kind of latent in the middle of the thinking process, we can also simply add noise/mask/...

2. learn how to use memory
    **short memory** represented by the Latent will be learning with the training mentioned before since Latent is the communication mechanism between step
    **medium memory** for this one :
    1. we store latent at each step
    2. we use a ReadLatentLayer to read the previous latent step
    3. backprob signal passing through the read latent, to the layer that produced it. that will give the learning signal to backprop that latent can be re-use, so that backprop can optimize it
    4. keep all those gradients might use a lot of memory
        1. we can stop after a certain number of steps
        2. we will start with a small to study how the number of steps affects the inference 

3. learn how to use/build long-term memory
    for using long-term memory we can keep old latent in a store and read them during inference  
    but since those latent are built to be reused, the learning could be slow  
    to accelerate it we can backdrop the signal like in the medium term memory. that can be challenging to do since the long term it's way larger. An approach is hold it in different GPUs, the required backward state for the backward gradient step, that can only at scale.  

    but we will use a more efficient way to accelerate it. we assume that long term can be read by just reading a lot of input data and keeping them in compressed format in the memory. we can do that by running Input + Latent(compressed signal token) -> ReadInput -> Process -> Latent(compressed)  
    with this approach we can do backprop during training: eg. for text input we can run this on many text sections to get the compressed Latent knowledge and use during step with ReadLatentLayer. the training signal will be backprop thought process, reading and up to the compressed signal token (which a just one token, the other token can be random)  
    we can do this process hierarchically by compressing again the compressed token before giving them to the thinking steps. to avoid losing information, we can make the second step less compressive.  
    **important!** that will also teach the model how to synthesize his own knowledge, via extracting important information from what it already extracted. we might do many times during inference eg. 10 Levels of compression :-) to have contextual compression we can give the compression process the Latent in the current step. Also, that could slow down execution during training by making it more sequential and less parallel.  

4. learn how to index long-term memory  
    indexing because long-term memory, can be very wide - then pre-selected the tokens that will be read by the model will help significantly  
    can be done by learning how to predict the most useful block of memory  
    useful block can be estimated by how much attention is given to them, but might not be meaningful  
    RL can we using considering memory reading block as action, so that the model will how to choose the best one.  

    we can make it easy for the model by caching the most frequently used memory block, maybe they might be general knowledge that are not stored in the language model weight. Like coding skills, some English vocabulary, ... maybe they might differ for a given task such as coding python, or coding java, or writing a novel.  




### advance training with RL
how to make smart/dynamic steps?
here the model will is how to pick the process block to be efficient in reaching the final goal.  
eg. maybe readtext -> process -> process -> readlongterm_memory is more efficient given a task that standard step during training, even if we randomize the sequence during the training step  

RL focuses on making the process efficient.  
the reward could be : goal_reach * 1/computation_time

we can further than smart steps, we can take smart action in each step.  
eg. when reading latent where should we focus our attention  
    when reading input (split in part) which part of the input   should we focus on eg. For a large code base with many files  
    when reading long-term memory on what should we give attention to  
    when processing, is it necessary to check the previous output  

Lol : we can perform standard training + RL training, at the same time or alternate them. so that the model will accumulate knowledge in normal training, and learn how to use them efficiently during RL. we can do that for some cycle and then restart the process with a bigger model and use the smaller model as a guide (check deepmind Foundation RL Ada distillation that uses the signal from a smaller model to help bigger to jump-start learning)  

sometimes we also run the model from scratch (cleaned memory), so that it's also improves his/she/it learning abilities without background knowledge. Or targeted cleaning by removing, knowledge related to the task.  

Possible RL Task:
- learn how to code by, looking git commit history
- we can use code editing to learn that
- learn by talking with a bigger system like ChatGPT
- learn by searching on the internet
- learn with tools eg. interact with Linux terminal
- learn fact-checking

## Software Design
we can software design approach in MLOps for researchers:
1. Version Control
2. Automated Testing
3. CI/CD
4. Monitoring and Tracking
5. Reproducibility

Tools:
1. Pytorch
2. MLflow
3. DVC : Data Version Control

with the following particularities:
1. make module interoperable module/block : since we have many blocks, independently call during training/inference
2. use object orient programming to avoid rewriting the same code eg. inheritance InputLayer -> TextInputLayer
3. Focus on simplicity to edit, since we are doing research we try a lot of things using too much/rigid software design should be avoided

Why not ask existing LLM : "Now propose me a software design with pytorch"
