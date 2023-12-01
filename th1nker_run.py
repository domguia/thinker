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
    hdim = 32,
    head_size = 8,
    number_of_head= 4,
    resid_pdrop = 0.1,
    attn_pdrop = 0.1,
    bias=False,

    vocab_size = 270,
    
    input_cache_size = 256,
    mem_cache_size = 2048,

    min_latent_size = 16,
    max_latent_size = 128,
    max_output_len = 256,
    
    min_step=2,
    max_step=16,

    probe_mode="number_reg",
    good_pred_loss_treshold=0.5,
    decay_coef=4,
)


if __name__ == '__main__':


    dataset = NumbersComputeDataset(TASK_SCHEME)
    batch_size = 27
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # cfg(vocab_size=NumbersComputeDataset.get_vocabulary_size())
    model = Th1nker(cfg)

    for inputs,targets in dataloader:
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

        latent_size = np.random.randint(cfg.min_latent_size, cfg.max_latent_size+1)

        model.init(batch_size, latent_size)
        model.load_input(inputs)
        logs('batch_size, n_step')

        losses = []
        for i in range(n_step-1):
            model.compute_step()
            # model.compute_step(with_output=targets.size(1))
            # # output = model.compute_step(with_output=y) #causal
            # output = model.get_output() #parallel
            # loss = compute_loss(output, targets, cfg.probe_mode)
            # losses.append(loss)

        model.compute_step(with_output=targets.size(1))
        output = model.get_output()
        loss = compute_loss(output, targets, cfg.probe_mode)

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
        loss = output_losses.mean()
        loss.backward()

        print(f"loss: {loss.item():.4f}, n_step: {n_step}, latent_size: {latent_size}")

        # logs('probe_loss, pred_loss, outputs_probe_losses')
        # logs('good_pred_ratio,loss')
        # print(logs)

        # break
