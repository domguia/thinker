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

from thinker_model import Th1nker, compute_loss #, CfgNode
from numbers_data import NumbersComputeDataset, TASK_SCHEME

# should be defined here because of globals()
class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def merge_from_dict(self, d):
        self.__dict__.update(d)
    def __call__(self, *args, **kwargs):
        self.__dict__.update(**kwargs)
        args = [item.strip() for items in args for item in items.split(',')]
        self.__dict__.update(**{name: globals()[name] for name in args})
    def __str__(self):
        return self.__dict__.__str__()

cfg = CfgNode(
    hdim = 512,
    head_size = 16,
    number_of_head= 32,
    resid_pdrop = 0.1,
    attn_pdrop = 0,
    bias=False,

    vocab_size = 270,
    
    input_cache_size = 64,
    mem_cache_size = 512,

    min_latent_size = 8,
    max_latent_size = 32,
    max_output_len = 32,
    
    min_step=2,
    max_step=7,

    probe_mode="number_reg",
    good_pred_loss_treshold=0.5,
    decay_coef=4,
)


if __name__ == '__main__':


    dataset = NumbersComputeDataset(TASK_SCHEME)
    batch_size = 27
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch device :", device)
    # cfg(vocab_size=NumbersComputeDataset.get_vocabulary_size())
    model = Th1nker(cfg).to(device)
    
    # import torchinfo
    # torchinfo.summary(model)

    # Optimizers specified in the torch.optim package
    learing_rate=0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate) #, momentum=0.9)
    
    loss_tracker = []
    for idx, (inputs,targets) in enumerate(dataloader):
        inputs,targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        logs = CfgNode()
        logs('batch_size')

        n_step = np.random.randint(cfg.min_step, cfg.max_step)
        # m_step = np.random.randint(1, n_step)
        logs('n_step')

        # #### stop gradient run
        # with torch.no_grad():
        #     for _ in range(m_step):
        #         model.compute_step()
        # for _ in range(m_step, n_step-1):
        #     model.compute_step()

        #### full run with gradient

        optimizer.zero_grad()

        latent_size = np.random.randint(cfg.min_latent_size, cfg.max_latent_size+1)

        with torch.device(device):
            model.init(batch_size, latent_size)
            model.load_input(inputs)
        # logs('batch_size, n_step')

        losses = []
        for i in range(n_step-1):
            with torch.device(device):
                model.compute_step()
            # model.compute_step(with_output=targets.size(1))
            # # output = model.compute_step(with_output=y) #causal
            # output = model.get_output() #parallel
            # loss = compute_loss(output, targets, cfg.probe_mode)
            # losses.append(loss)

        with torch.device(device):
            model.compute_step(with_output=targets.size(1))
            output = model.get_output()
            loss = compute_loss(output, targets, cfg.probe_mode)

        for break_i in range(targets.size(1)-1,-1,-1):
            if targets[:,break_i].float().mean() < 20: break

        if idx%10==0:
            print()
            probe, logits, outputs_probe = output
            for i in range(targets.size(1)):
                val = targets[0,i].item()
                print(f"{val:4d}", end=', ')
                if val == 20: break
            print()
            for j in range(i+1):
                val = outputs_probe[0,j].item()*16
                print(f"{val:.2f}", end=', ')
            print('^')

        # losses.append(loss)

        # n = len(losses)
        # # losses = torch.Tensor(losses)
        # # losses = list(map(list, zip(*losses)))
        # # losses = [list(filter(lambda x: x, col)) for col in zip(*losses)]
        # losses = list(map(lambda x: torch.stack(list(x)).transpose(1,0), zip(*losses)))
        # _, probe_loss, pred_loss, _, outputs_probe_losses = losses
        
        # ## more weight to the good and llast loss
        # ## without neglecting the first lower quality
        # ## so that the model will value progress in early step
        # ## while give more importance to last/good one
        # good_ = pred_loss > cfg.good_pred_loss_treshold
        # coef_ = good_.clone()
        # good_pred_ratio = good_.sum(dim=1)/n
        # # coef_[good_] = 0.5/good_.sum(dim=1)
        # # coef_[~good_] = 0.5/(n-sum(good_))
        # coef_ = torch.where(good_,0.5/good_.sum(dim=1)[:,None],0.5/(n-good_.sum(dim=1)[:,None]))

        # ## decay coefficient followed steps
        # coef_decay = (cfg.decay_coef*torch.arange(n)/n).softmax(dim=0)
        # coef_ = coef_ * coef_decay
        
        # loss_1 = (probe_loss * coef_).mean()
        # loss_2 = (outputs_probe_losses * coef_ * coef_decay).mean()
        # loss_3 = (pred_loss * coef_).mean()
        # loss = loss_1 + loss_2 + loss_3
        
        # probe_loss, pred_loss, outputs_probe_losses = probe_loss[:,-1].mean().item(), outputs_probe_losses[:,-1].mean().item(), pred_loss[:,-1].mean().item()
        # logs('probe_loss, pred_loss, outputs_probe_losses')
        # probe_loss, pred_loss, outputs_probe_losses = probe_loss.mean().item(), outputs_probe_losses.mean().item(), pred_loss.mean().item()


        # print(f"loss {loss:.4f}, good pred : {good_pred_ratio:.4f} = {sum(good_)} / {n} preds over 0.5 treshold")

        _, probe_loss, pred_loss, output_losses, outputs_probe_losses = loss

        # loss = probe_loss + pred_loss[:,None] + outputs_probe_losses
        # targets < 20
        output_loss = output_losses[:,:break_i].mean()
        outputs_probe_loss = outputs_probe_losses[:,:break_i].mean()
        (output_loss + outputs_probe_loss*16*4).backward()

        optimizer.step()

        if idx%10==0:
            print(f"{idx} :: loss: {outputs_probe_loss.item():.4f} + {output_loss.item():.4f}, n_step: {n_step}, latent_size: {latent_size}")

        # logs('probe_loss, pred_loss, outputs_probe_losses')
        # logs('good_pred_ratio,loss')
        # print(logs)

        loss_tracker.append(outputs_probe_loss.item())

        if idx%50==0 and idx>=100:
            mean = np.mean(loss_tracker[-50:])
            mean_prev = np.mean(loss_tracker[-100:-50])

            print(f'averaged loss -> mean_prev:{mean_prev:.4f} mean:{mean:.4f}')
            # lr = learing_rate * max(np.abs(mean-mean_prev), 100/idx)
            
            import matplotlib.pyplot as plt
            plt.plot(loss_tracker[10:])
            plt.savefig("loss.png")

            with open('./train_param.txt','r') as f:
                lr = float(f.read())

            print('========== learing_rate:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # if idx%100==0:
            #     torch.save(model, "model.pck")

        # if idx == 1000:
        #     lr = learing_rate * 0.1
        #     print('learing_rate:', lr)
        #     # lr = float(input('Learning new rate: '))
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # break

