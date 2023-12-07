
import torch
import numpy as np
import lightning as L
from torch.utils.data import DataLoader

from thinker_model import Th1nker, compute_loss, CfgNode
from numbers_data import NumbersComputeDataset, TASK_SCHEME

cfg = CfgNode(
    hdim = 32,
    head_size = 4,
    number_of_head= 8,
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

# define the LightningModule
class LitTh1nker(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = Th1nker(config)
        self.cfg = config

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, targets = batch

        B, T = inputs.size()

        n_step = np.random.randint(self.cfg.min_step, self.cfg.max_step)
        latent_size = np.random.randint(self.cfg.min_latent_size, self.cfg.max_latent_size+1)

        # with torch.device(device):
        self.model.init(batch_size, latent_size)
        self.model.load_input(inputs)

        for i in range(n_step-1):
            # with torch.device(device):
            self.model.compute_step()

        # with torch.device(device):
        self.model.compute_step(with_output=targets.size(1))
        output = self.model.get_output()
        
        loss = compute_loss(output, targets, cfg.probe_mode)
        _, probe_loss, pred_loss, output_losses, outputs_probe_losses = loss
        
        output_loss = output_losses.mean()
        outputs_probe_loss = outputs_probe_losses.mean()
        
        # loss fusion
        loss = (output_loss + outputs_probe_loss*16*4)

        # Logging to TensorBoard (if installed) by default
        # self.log("step", B*batch_idx) # for weight and bias
        self.log("sample", B*batch_idx*1.0)
        # self.log("batch_idx", batch_idx)
        self.log("output_loss", output_loss)
        self.log("outputs_probe_loss", outputs_probe_loss)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
model = LitTh1nker(cfg)


# init data loader
dataset = NumbersComputeDataset(TASK_SCHEME)
batch_size = 27
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)


# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=model, train_dataloaders=dataloader)


