import time
import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader

from core.toy_model import ToyThinker, all_losses_compute
from data.numbers import NumbersComputeDataset
from core.utils import CfgNode

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug', help='Name of the experiment')
    args = parser.parse_args()
    
    exp_dir = f"logs/{args.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # Model configuration
    model_cfg = CfgNode(
        vocab_size=17+30, max_latent=16, max_input_len=60, max_output_len=30,
        d_model=64, nhead=2, d_hid=128, nlayers=1, n_probe=1, dropout=0.0, rank=8
    )
    
    # Run configuration
    run_cfg = CfgNode(
        batch=256, learning_rate=1e-3, max_time_minutes=15, acc_gradient=1
    )
    
    # Data configuration for add_mul
    data_cfg = CfgNode(
        batch=run_cfg.batch, step=4, max_number=1000, operations='+-*', operation_dist=[0.4,0.4,0.2],
        in_bases=[10], in_bases_dist=None, out_bases=[10], out_bases_dist=None
    )

    # Initialize model
    model = ToyThinker(**model_cfg.__dict__).to(device)
    
    # Print Params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params / 1e3:.1f} K", flush=True)

    # Optimizer (AdamW) and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_cfg.learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)
    
    # Dataset
    dataset = NumbersComputeDataset(data_cfg)
    
    print(f"Starting training on {device} for a maximum of {run_cfg.max_time_minutes} minutes.", flush=True)
    
    start_time = time.time()
    max_time_seconds = run_cfg.max_time_minutes * 60
    
    loss_tracker = []
    acc_tracker = []
    best_loss = float('inf')
    
    model.train()
    optimizer.zero_grad()
    
    for idx, (inputs, targets) in enumerate(dataset):
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time_seconds:
            print(f"Time budget of {run_cfg.max_time_minutes} minutes reached. Stopping training.", flush=True)
            break
            
        inputs, targets = inputs.long().to(device, non_blocking=True), targets.long().to(device, non_blocking=True)
        
        # Sample hyperparameters for this step
        n_latent = np.random.randint(4, 9)
        n_step = np.random.randint(1, 4)
        read_step = n_step - 1
        
        # Forward pass
        outs = model(inputs, targets, n_latent, n_step, read_step,
                     is_full_ar=True, is_output_ar=True, output_step=1)
                     
        loss, logs_ = all_losses_compute(outs, targets, target_emb=None, last_step_only=False)
        
        # Accuracy computation
        logits = outs[1][:,-1,:,:].detach()
        # Shape is (batch_size, target_seq_len, vocab_size) if we take all seq length
        # Or outs[1] is (n_step, b, s2, vocab), taking last step: [-1] -> (b, s2, vocab)
        preds = torch.argmax(logits, dim=2)
        # Sequence exact match or token average accuracy? Let's use token average
        accuracy = (targets == preds).float().mean().item()
        
        # Backward pass
        loss.backward()
        
        if idx % run_cfg.acc_gradient == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        loss_tracker.append(loss.item())
        acc_tracker.append(accuracy)
        
        # Logging
        if idx % 100 == 0:
            mean_loss = np.mean(loss_tracker[-100:]) if len(loss_tracker) >= 100 else np.mean(loss_tracker)
            mean_acc = np.mean(acc_tracker[-100:]) if len(acc_tracker) >= 100 else np.mean(acc_tracker)
            print(f"Step {idx:5d} | Elapsed: {elapsed_time/60:.2f}m | Loss: {mean_loss:.4f} | Acc: {mean_acc:.4f} | n_latent: {n_latent} | n_step: {n_step}", flush=True)
            
            if mean_loss < best_loss and idx > 0:
                best_loss = mean_loss
                torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
                
                # Also save metrics
                with open(os.path.join(exp_dir, "metrics.csv"), "a") as f:
                    if os.path.getsize(os.path.join(exp_dir, "metrics.csv")) == 0:
                        f.write("step,elapsed_min,loss,accuracy,n_latent,n_step\n")
                    f.write(f"{idx},{elapsed_time/60:.2f},{mean_loss:.6f},{mean_acc:.6f},{n_latent},{n_step}\n")

    print("\n---")
    print(f"best_loss:        {best_loss:.6f}")
    print(f"training_seconds: {elapsed_time:.1f}")
    print(f"total_seconds:    {elapsed_time:.1f}")
    print(f"num_steps:        {idx + 1}")
    print(f"num_params_M:     {sum(p.numel() for p in model.parameters()) / 1e6:.2f}")

if __name__ == '__main__':
    train()
