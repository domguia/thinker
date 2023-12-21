# Experiment Log

## 05 Dec 2023
We got the error below only on GPU, the code worked well on CPU!
```RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [27, 4, 606, 8]], which is output 0 of AsStridedBackward0, is at version 19; expected version 18 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!```

Because of:
`self.k_cache[:, begin_insert_idx:end_insert_idx] = k`

Fixed it in commit `f96161d`, which should be improve for code cleanes and performance

Should improve device assignement, now I have to use: `with torch.device(device):`

## 06 Dec 2023

Oups The model don't learn!

Make the problem simpler:
- task : copy input sequence of 1010 to output 
- run model for 1 step
- use only crossentropy loss

The model learned:
- to pic the right embedding 0, 1
- identify the end of the sequence
- for some sample make perfect copy

Observation:
- the task require more compute than expected, the model observed 1.5e samples = 30.000 batchs * 512 sample/batch * lr = 0.001 per batch 
- at the beginning the loss go down quite easily, after it's constant loss for long time `5e6 samples` then start reducing quickly during that phase the accuracy also improve. My interpretation is that the solution we're looking is rare in the solution space, and difficult to be found via the loss landscape. but ounce around the `exploring stop` and `improvement start` until it get stuck again.
- Now the model learn the task some sample a perfectly predicted but vaerage accuray is 0.5 works better but I didn't manage to 

## 07 Dec 2023

Looking for solution to make the model reach 100% accuracy on copy task, but the compute seem too long.

I move for 0,1 sequence of number to 0,1,2 .., 10 sequence wich with my with give stringe signale to model during trining since the're not just 2 options

![loss plot](/exp_logs/copy_task_w1layer_loss_curve.png)

## 08 Dec 2023

I suspect something wrong with my model implem 

## 18 Dec 2023
I spent a lot of time debugging finally the issue was the model the batch and the sequence lenght was permutted in transformer pytiorch installation that's why my model wasn't leaning anything.
After issue was fixed the model learn quite quickly copy task, with flip, rolled number
Addin 2 step incresea significaly the performance on learning rolled number

After I reagange the model to make it behave like thinker (memeory+step)
I manage to reach 97 accuracy on addition of number on base 16 (max number was 1000)

from a previous training I observe that changing didn't affect performance on this task so I used fix step=1 and latent=4
With batchsize of 4096, learning_rate=0.02, 

I had to implement restart with best model when loss drop and gradient clipping, especially gradient clipping is very effective

I was worrying about the data loading timing, but it seem to be good
```
0.000336 second for data loading
0.011631 second for forward+loss
0.081530 second for backward+remaining
```

I manage to have 95% GPU utilisation even with 2.6Gb usage on Colab T4

I manage to reach 97accuracy for addition of base 16 number between 0-1000 (base10) with bacth size1024 and lr 0.005
![loss accuracy plot](/exp_logs/loss_acc_1addition_base16_max_number1000_hdim32_2step_8latent_lr0.005-0.0002.png)

# 19 Dec
I manage to run 3000 iteration in 2min with 95 accuracy on addition task but didn't reach 100% accuracy  

I observe it took me a lot of time to find the hyperameter and training setup for fast training  

To go faster I can:  
1. run many experiment in the backgroud while doing other stuff -> move to pytorch lighting will help  
2. find simpler problem eg. make copy just by lookig at memory

Project Update:
1. Moved to Pytorch Lighting
2. Avoid un-necessary computation by making optionnal the self attention in nn.TransformerDecoder
3. Update the CopyDataset for larger batch and more task
4. Tried TPU without success du to error on sub libray `import _XLA` 

Some ideas:
For faster training improve the loss, by make it lower for likely output regarding the problem:
1. for LLM task make, compare output distribution unregarding token location eg:
    1. global embedding average, loss = avg(output) - avg(target)
    2. loss on token neigborh
    3. compare all output to all target embedding :
        1. cluster targets embedding with in n centroid with knn, n==output embedding
        2. push the closest each output to move closer to his closest embedding, step after step
2. for number base task, we can make the loss to be condition on the sequence (or just use causal decoding)

# 20 Dec

model with hdim=16 and 9k parameters seam to perform the addition as well, just need more iteration to converge, number of latent and step seam not affect much the performance even the training are the all same while changing latent and step
![heatmap](/exp_logs/heatmap_16hdim_batch1024_20dec_exp.png)
![training curve](/exp_logs/loss_acc_16hdim_latent8_step4_batch1024_20dec_exp.png)

to go faster I should test the model with simpler task, such as copy and thier variants

```python
model_cfg = CfgNode(
    vocab_size = 31+1, n_probe = 1,
    max_latent=64, max_input_len=48, output_len=40,
    d_model=16, nhead=8, d_hid=16*4, nlayers=1, dropout=0
)
data_cfg = CfgNode(low=0, high=16, seq_len=20, batch=1, task=None)
run_cfg = CfgNode(max_iter=4_000, learning_rate=0.01, batch = 1024)
exp_cfg = CfgNode(
    n_latent = range(4, 8+1), # hyper parameters grid search
    n_step = range(1, 4+1),
)
```
the result below
![overview map](/exp_logs/copy_task-seq_len_20-hdim16_20dec_exp.png)

Observation, the model with higher capacity step=4 perform consitenly bad, probably it need more iterations to converge

# 21 Dec

I design a basic currilum learning and it make the copy task much easier to learn, the model reach 100accuracy with that eproach but plateau 60accuracy without it.

How I made the currilum, I vary the sequence lenght in the dataset following an uniform distrubtion, so that the model can easily learn from the short sequence and progressively learn how the longer one's

![with uniform curriculum](/exp_logs/loss_uniform_copytask_acc_16hdim_latent8_step4_batch1024_20dec_exp.png)
![with non uniform](/exp_logs/loss_non-uniform_copytask_acc_16hdim_latent8_step4_batch1024_20dec_exp.png)

Regarding this significan improvement I made a better currilum: by make a dynamic sampling the dataset distrution. With a risk of making the problem non-stationnary, but I think even if is non stationary the loss landscape should be easier the navigate during optimisation. 

