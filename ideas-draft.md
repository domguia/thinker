# Th1nker model
_Note: Sorry if reading this is difficult, this text is full of typos and fuzzy ideas, I will clean it while progressing on the work. Thanks!_

Why?  
Because AI today is not smart enough, for lazy guys like me

What?  
Train a model to learn a thinking process steps

How? by :  
1. Separate reasoning and knowledge in LLM.
2. Allow variable computation budget through reusable computation blocks.
3. Enable differentiable out-of-model memory/knowledge mechanism via long-term trained indexer/retrieval.

How to do that:
1. Have a latent process for main computation that uses re-usable blocks similar to those in Perceiver architecture.
2. Use cross-attention for input/output by re-attending to the input in the process and generating output progressively.
3. Use past latent as thinking memory by cross-attending on the past latent while ensuring that everything is differentiable.
4. Implement a long-term memory mechanism by cross-attending on all relevant content types (optionally, a rigid memory learned during training).

## In details
the thinker can be a transformer base model, made with building blocks responsible for each task in the process. Here the [visual explanation](visual-explanation.svg).  
[figma latest visual](https://www.figma.com/file/MNe376umkTm5iCpg9kSmcq/thinking-transformer?node-id=328-196&t=fIFQ60I3hcz39A4t-0)

### Blocks
Here are the proposed building blocks:
1. the step process is given by calling during inference:
    1. ReadLayer : read any kind of information [1](https://arxiv.org/abs/2202.05826) [2](https://arxiv.org/abs/2103.03206)
    2. ProcessLayer : process information, always the same
    3. OutputLayer : get output from
    4. Probe (optional) : predicting the block that will be used in the next thinking step
2. processing memory (or short-term memory) [3](https://arxiv.org/abs/2301.04589)
    1. Latent : is as input in all layers mentioned before, it also the initial input of the model
    2. Input : is the input data we want to process, we may have TextInput, ImageInput ... , InputText + Latent -> ReadTextInputLayer
    3. Ouput : is the expected output, optionally autoregressively (output probe)
3. medium-term memory component
    1. LatentStore : we store latent at each processing step, they will be read at inference with ReadLatentLayer
4. long-term memory : is also built with latent but of the past run of the model
    1. it will be probably very large and need an indexer to get the useful one
    2. ReadLatentLayer here will be associated with a learned Indexer

### Inference
Ability to vary computation budget and block call sequence depending on task. Consider using RL to find a better sequence of block calls.

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

**idea** : 
Why not ask existing LLM : "Now propose me a software design with pytorch"  
rewrite a text in the format "question? answer. question? answer. " and use that text to teach a model to follow instructions with just LLM standard AR and avoid complex RL  


# important update : 1 Layers to rules them all
with more general flexibility we can do more

### flex projection layer
adaptative projection layer, with random input/output dimensions
eg. `
x1 = tensor(16,2), x2 = tensor(16,10)
proj = nn.FlexLayer(in=6,out=4)

proj(x1) == x1 @ proj.matrix[0:2,:] # x1.shape[0] == 2
proj(x1).shape == 16,4

proj(x2) == x2[:6] @ proj.matrix # proj.matrix.shape[0] == 6
proj(x2).shape == 16,4

proj(x2, in=2) == x2[:2] @ proj.matrix[0:2,:]
proj(x2).shape == 16,4

proj(x2, out=3) == x2[:6] @ proj.matrix[:,0:3] # proj.matrix.shape[0] == 6
proj(x2).shape == 16,3
`

### flex attention
here we use flex project to get Q,K and V
so the inputs can contain elements with different dimensions (text embed, image embed, memory embed, ...)
Q,K shared dimension and V dimension can change during inference

the general rule of changing dimension is:
n current dim, m next dim
- if n>m: we use m first component of n dim eg. vec[:m]
- if n==m: no change
- if n<m: we need a projection layer for that, is m is too high for the defined projection layer. raise an error

we multi-head add some constrain, in the dim. to handle it we can:
1- we can fix head dimension, and vary number of head, by adding +head_dim is like adding 1head, or remove -head_dim is like removing 1head
2- we can fix the number of head but vary head dim, in this case for flex projection to work, we should chunk the input before flex projection
we want to avoid head to inconsistent while varying dimension.

we will use both during training for varying dim, and also after training to increase model size

### flex input/output
#### input:
call be a list of heterogeneous dimension attend via cross attention, on input by latent
when doing cross attention we keep still include latent so they can attend on them self
inputs -> K_input, V_input: K_input=K_1, K_2, .. K_n same for V 
    - input_1 -> K_1, V_1
    - input_2 -> K_2, V_2
    - input_n -> K_n, V_n
latent -> K_l,Q_l,V_l
K = K_input, K_l
V = V_input, V_l
after we just do standard scaled dot.

ideas: when doing cross-attention we keep the latent un-projected since they already have V dimension (multi-head brings down that dimension internally) 
thinking: I was worrying the input can be very large compared to the latent and crush the information in the latent which are the short memory of the model, a solution for that could be, just use an only subpart of latent query and value Q_l[:s], V_l[:s] while reading input, the other latent[s:] stay un-changed. So we can do standard self attn during the pure processing step. This might un-necessary complexity no need to that just keep it in mind.

kind of input:
we can use the embedding signal to differentiate inputs
signal.input.model_output # since we give model output as input
signal.input.token.fill_text_here

more vary input:
this allows us for the same input to work at different dimensions,
eg. token embedding, which can be sliced embed[:,:s] during training this will the first vector component more informative and the last less informative but they add add will add a more precision this will be very useful for :
- reduce dimension by still efficiently keeping information
- balance memory dimension
- during inference to balance compute
- when we scale up the model the child model can easily learn (or init param) from the parent by just considering the child's higher dimensions as additive information.


#### output:

to get ouput we use embeding signal:

signal.output.text_start # init text generation
signal.output.memory # extract short
signal.output.probe # predict thing about model
signal.output.index # predict thing about model

how: (is basically a transformer decoder on the latent)
output_signal cross attend on latent (and maybe inputs)
output_signal -> Q 
latent -> K,V
latent,inputs -> K,V # optionaly

after we got attention
output_signal -> output

we can also make it causal, in that case
output_signal -> Q,K_o,V_o 
latent -> K_l,V_l
Q=Q, K=K_l+K_o, V = V_l+V_o
with mask so that:
    output_signal[0], attends only on latent 
    output_signal[1], attends on latent and output_signal[0]
    output_signal[n], attends on latent and output_signal[0..n-1]
that allow to do autoregressive modeling

eg. for memory output  
memory_signal = signal.output.memory

we how effectively vary memory_signal: embed_dim and number_of_components
an approach could be : memory_signal = [signal_embed, rand_embed, ..., rand_embed]

memory_signal -> output = memory_value, index
Note: index would be trained by looking at the utility of each memory component (utility could be attention map)
signal.output.index -> index_to_find, the loss would be index_to_find[n] = index[n] for the same memory component, but index_to_find[n] != index[m] for different memory component. Problem: this is contrastive learning with only one positive example, which may not work. We can just train index_to_find,index to be close eg. mse_loss(index_to_find,index)



#### why not latent
we can make latent dimensions flexible

**General note about flexibility:** it's possible that adding only 1 component doesn't help the model that much to handle information, we can make flexibility by jumping off a fixed number of components, a jump of multiple of K (jump factor). At least that's probably sure we scale up the model.

### flex training

1. we can only have one layer, that we call for everything: read input, process, store in memory, read memory, generate output, and predict index, ...
2. we just change, how we run cross attention and the embedding we give to trigger behavior
3. we latent that we have in and out attention, they act like processing memory or short-term memory
4. the model can give output and reuse them (backprop should work) which act like medium-term memory
5. to build long-term memory, we run the other chunk of related input and give as output memory as input to the running mode
6. during training, we should after some layer do stop the gradient, so that model learns how to start with cold latent/memory so that at interference we can have a deeper run than during training [End-to-end Algorithm Synthesis with Recurrent Networks](1)
7. we can flex the latent, by adding/remove some during training or remove/adding dimension

### scale up model
we create a bigger model (student) that learn from smaller (teach)
every time when varying dimension if the dimension falls in the domain where the small model(teacher) has been trained on we use, the small model output latent as a signal with a coefficient and add it to the real output compute by the bigger parent. this will not add so much perturbation since both model run in that specific flex configuration.

coefficient could high at the beginning like 0.8 and reducing while the parent start reach performance of the student

coefficient = 0.8 * max(0, child_performance - parent_performance - 0.3 ) # for performance the higher the better. But we may use perplexity during experiment 
this mean : start with 0.8 coffecient and reduce it util the parent reach 70% of the children, in the same flex configuration. sothat we let the learn completly new mechanism. it just for boot starting the model

gradient check pointing for memory efficiency  
https://huggingface.co/docs/transformers/v4.18.0/en/performance#using-accelerate  
https://github.com/cybertronai/gradient-checkpointing  

# long vs short term memory

we can have specific embedding for short term and long term memory
so that when doing inference on normal task we can ask long term memory information.
but how do you when long term memory information will be usefull?

we can add a utility probe: that try to predict the utility of a information block, base on how frequent/intense the model give attention to it.
the utility probe should be share between short/long term memory, because it's easier to train it with compression and short term memory module. we expect the probe to generalize longterm extraction, so that it's also work during inference.


# Reference

[More details](https://typst.app/project/rhEBD144ScLMqpuimnY7vs)  

Some good ideas from:
1. [End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking, Arpit Bansal al.](https://arxiv.org/abs/2202.05826) 
2. [Perceiver: General Perception with Iterative Attention, Andrew Jaegle al.](https://arxiv.org/abs/2103.03206)
3. [Memory Augmented Large Language Models are Computationally Universal, Dale Schuurmans](https://arxiv.org/abs/2301.04589)
4. [Looped Transformers as Programmable Computers, Angeliki Giannou al.](https://arxiv.org/abs/2301.13196)
5. [Reflexion: an autonomous agent with dynamic memory and self-reflection](https://arxiv.org/abs/2303.11366)


6. Deep Implicit Layers
http://implicit-layers-tutorial.org/
https://youtu.be/MX1RJELWONc
