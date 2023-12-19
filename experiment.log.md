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

