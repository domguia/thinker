
import torch
import numpy as np
import lightning as L
from torch.utils.data import DataLoader

from thinker_model import Th1nker, compute_loss, CfgNode
from numbers_data import NumbersComputeDataset

model_cfg = CfgNode(
    # hdim = 32,
    # head_size = 4,
    # number_of_head= 8,
    # resid_pdrop = 0.1,
    # attn_pdrop = 0.1,
    # bias=False,

    # vocab_size = 31+1,

    # input_cache_size = 24,
    # mem_cache_size = 128,
    # max_output_len = 24,

    vocab_size = 31+1,
    max_latent=64,
    max_input_len=48,
    output_len=40,
    d_model=32,
    nhead=8,
    d_hid=32*4,
    nlayers=1,
    dropout=0
)

data_cfg = CfgNode(batch=2, step=2, max_number=1_000, operations='+-*/', operation_dist=[0,0,1.0,0],
    in_bases=[16], in_bases_dist=None, # [2,4,8,16] # list(range(2,8+1))
    out_bases=[16], out_bases_dist=None, # [.1,.2,.3,.4])
)

run_cfg = CfgNode(
    n_latent = 8, # range(4,8)
    n_step=4, # range(1,4)

    max_iter=5000,
    learning_rate=0.005,
    batch = 1*1024,
)

exp_cfg = CfgNode(
    n_latent = range(4,8+1),
    n_step = range(1,4+1),
)

# define the LightningModule
class LitTh1nker(L.LightningModule):
    def __init__(self, ):
        super().__init__(model_cfg, run_cfg)
        self.model = ToyThinker(**model_cfg.__dict__)
        self.cfg = run_cfg

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, targets = batch
        inputs, targets = inputs[0], targets[0] # implicit batch trick

        B, T = inputs.size()

        n_step = self.cfg.sample('n_step')
        n_latent = self.cfg.sample('n_latent')

        n_target = targets.size(1)
        logits, output_probe = model(inputs, n_latent, n_target, n_step)

        # compute loss
        output_loss = torch.nn.functional.cross_entropy(logits.permute(0,2,1), targets.long()) #, ignore_index=20)
        probe_loss = torch.nn.functional.mse_loss(output_probe.squeeze(-1), targets.float()/3)

        loss = output_loss + probe_loss

        # Logging to TensorBoard (if installed) by default
        # self.log("step", B*batch_idx) # for weight and bias
        self.log("sample", B*batch_idx*1.0)
        # self.log("batch_idx", batch_idx)
        self.log("output_loss", output_loss)
        self.log("outputs_probe_loss", probe_loss)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer


# init the autoencoder
model = LitTh1nker(model_cfg, run_cfg)

# init data loader
data_cfg.batch = run_cfg.batch # implicit batch trick
batch_size = 1

dataset = NumbersComputeDataset(data_cfg)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=run_cfg.max_iter, max_epochs=1)
trainer.fit(model=model, train_dataloaders=dataloader)


