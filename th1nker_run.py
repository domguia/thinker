# # number of call step for the model should be evaluated considering task scheme and memory usage
# params = dict(
#     # data param
#     batch_size = (1, 4, 8, 32),
#     input_lenght = (16, 64, 128, 256, 512, 1024),
#     output_lenght = (16, 64, 128, 256, 512, 1024),

#     # model run param
#     steps = (1, 4, 8, 16, 32, 64, 128),
#     latent = (4, 8, 16, 32, 64, 128),
#     memory_context = (16, 32, 64, 128),

#     # model weight param
#     dim = (32, 64, 128, 256, 512, 1024)
#     n_layers = (1,2,3)
#     n_heads = 8
#     # head_dim = 8
#     # hidden_dim = ()
# )


import torch
import numpy as np
from torch.utils.data import DataLoader

from thinker_model import Th1nker, compute_loss
from numbers_data import NumbersComputeDataset


def train(r_cfg, dataset, model):

    batch_size = 1 # 1024
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    optimizer.zero_grad()
    model.train()

    from time import time
    start_time = time()
    for idx, (inputs,targets) in enumerate(dataloader):
    # for idx, (inputs,targets) in enumerate(dataset):
        end_dataloading_time = time()
        # targets = targets.sort(dim=1)[0][:,[0,0,-1,-1]] # predict min and max
        # inputs,targets = inputs.to(device), targets.to(device)
        inputs,targets = inputs[0].to(device, non_blocking=True), targets[0].to(device, non_blocking=True)
        batch_size = inputs.size(0)


        n_latent = 8 #np.random.randint(cfg.min_latent_size, cfg.max_latent_size+1)
        n_step = 2 #np.random.randint(cfg.min_step, cfg.max_step+1)
        # m_step = np.random.randint(1, n_step)

        # latent_size = np.random.randint(cfg.min_latent_size, cfg.max_latent_size+1)

        # with torch.device(device):
        #     model.init(batch_size, latent_size)
        #     model.load_input(inputs)
        # # logs('batch_size, n_step')

        # losses = []
        # for i in range(n_step-1):
        #     with torch.device(device):
        #         model.compute_step()
        #     # model.compute_step(with_output=targets.size(1))
        #     # # output = model.compute_step(with_output=y) #causal
        #     # output = model.get_output() #parallel
        #     # loss = compute_loss(output, targets, cfg.probe_m1024ode)
        #     # losses.append(loss)

        # with torch.device(device):
        #     model.compute_step(with_output=targets.size(1))
        #     output = model.get_output()
        #     loss = compute_loss(output, targets, cfg.probe_mode)

        #### Toy model
        # with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        # logits = model(torch.nn.functional.one_hot(inputs, num_classes=16).float())
        n_target = targets.size(1)
        logits, output_probe = model(inputs, n_latent, n_target, n_step)
        output_loss = torch.nn.functional.cross_entropy(logits.permute(0,2,1), targets.long())#, ignore_index=20)
        probe_loss = torch.nn.functional.mse_loss(output_probe.squeeze(-1), targets.float()/2)
        #output_loss = output_loss * (n_step*n_latent)/(cfg.max_latent_size*cfg.min_step)
        end_forward_loss_time = time()

        break_i = 20
        # for break_i in range(targets.size(1)-1,-1,-1):
        #     if targets[:,break_i].float().mean() < 20: break

        # probe, logits, outputs_probe = output
        accuracy = (targets == torch.argmax(logits, dim=2)).float().mean().item()
        match_counter.update(logits, targets)
        if idx%20==0:
            s = np.random.randint(targets.size(0))
            correct = (targets[s,:] == torch.argmax(logits[s,:], dim=1)) #[:break_i]
            print()
            print("acc: %.3f - sample: acc:%.2f _ %d/%d" % (accuracy, correct.float().mean(), correct.sum(), break_i))
            for i in range(targets.size(1)):
                val = targets[s,i].item()
                print(f"{val:2d}", end=', ')
                # if val == 20: break
            print()
            for j in range(i+1):
                val = torch.argmax(logits[s,j]).item()
                print(f"{val:2d}", end=', ')
            print()
            # for j in range(i+1):
            #     val = outputs_probe[s,j].item()*16
            #     print(f"{val:.2f}", end=', ')
            # print()

        # _, probe_loss, pred_loss, output_losses, outputs_probe_losses = loss

        (output_loss + probe_loss).backward()
        # output_loss.backward()

        if idx%1==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
        if idx%100==0:
            # print(f"{idx} :: loss: {output_loss.item():.4f}, n_step: {n_step}, latent_size: {latent_size}")
            print(f"{idx:3d}: loss: {output_loss.item():.3f} accuracy: {accuracy:.2f}")


        # loss_tracker.append(outputs_probe_loss.item())
        loss_tracker.append(output_loss.item())
        acc_tracker.append(accuracy)
        logs.append(dict(
            loss = output_loss.item(),
            probe_loss = probe_loss.item(),
            accuracy = accuracy,
            latent = n_latent,
            step = n_step
        ))

        if idx%100==0 and idx>=100:
            from IPython.display import clear_output
            clear_output()

            plot_loss_and_accuracy(logs)

            # HACK: for update learning rate during training
            # with open('./train_param.txt','r') as f:
            #     lr = float(f.read())*learing_rate

            # print('========== learing_rate:', lr)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr

            print('learning_rate', optimizer.param_groups[0]['lr'])
            print(match_counter,'\n')
            match_counter = MatchCount(20+1+20)

            mean = np.mean(loss_tracker[-50:])
            mean_prev = np.mean(loss_tracker[-100:-50])
            print(f'averaged loss -> mean_prev:{mean_prev:.4f} mean:{mean:.4f}')

        if idx%10==0 and idx>=10:
            mean_loss = np.mean(loss_tracker[-10:])
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), "model.pt")
                print(f'save best model mean_loss={mean_loss:.4f}')
            # scheduler.step()

        if idx%200==0 and idx>=200:
            mean_loss = np.mean(loss_tracker[-100:])
            if mean_loss > best_loss:
                print('loss increase, load best model')
                model.load_state_dict(torch.load("model.pt"))
                scheduler.step()

        if idx>=200 and (loss_tracker[-1]/loss_tracker[-10])>1.5:
            print('loss peak increase, load best model')
            model.load_state_dict(torch.load("model.pt"))
            scheduler.step()

        if idx%101==0:
            print(f"Timing\n"
                f"{end_dataloading_time-start_time:.6f} data loading\n"
                f"{end_forward_loss_time-end_dataloading_time:.6f} forward+loss\n"
                f"{time()-end_forward_loss_time:.6f} backward+remaining\n"
            )

        if idx>r_cfg.max_iter:
            return logs
        start_time = time()

if __name__ == '__main__':
    def train(r_cfg, dataset, model)
