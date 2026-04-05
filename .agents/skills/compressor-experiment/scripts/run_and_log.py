import torch
import os
import argparse
import datetime
import math
from core.compressor.model_wrapper import HFModelWrapper
from core.compressor.optimizer import SoftPromptOptimizer
from core.compressor.metrics import CompressionMetrics

def log_to_global(date, model, target_len, prompt_len, steps, accuracy, ratio, exp_name, run_id):
    line = f"| {date} | {model} | {target_len} | {prompt_len} | {steps} | {accuracy*100.0:.1f}% | {ratio:.2f}x | Exp: {exp_name}, Run: {run_id} |\n"
    log_file = "dev_notes/compressor_experiments.md"
    
    # Ensure header exists
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("# LLM Compressor Experiments\n\n")
            f.write("| Date | Model | Target Len | Prompt Len | Steps | Accuracy | Ratio | Notes |\n")
            f.write("|------|-------|------------|------------|-------|----------|-------|-------|\n")
            
    with open(log_file, "a") as f:
        f.write(line)

def main():
    parser = argparse.ArgumentParser(description="Run and log a compressor experiment")
    parser.add_argument("--text", type=str, required=True, help="Text to compress")
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name")
    parser.add_argument("--n_prompt", type=int, default=10, help="Number of prompt tokens")
    parser.add_argument("--n_steps", type=int, default=50, help="Optimization steps")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for logs/")
    parser.add_argument("--run_id", type=str, default="0", help="Run ID within experiment")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    wrapper = HFModelWrapper(args.model, device=device)
    optimizer = SoftPromptOptimizer(wrapper)
    
    # Setup logging directory
    exp_dir = os.path.join("logs", args.exp_name, f"run_{args.run_id}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Starting optimization for '{args.text[:30]}...'")
    target_ids = wrapper.encode(args.text)
    target_len = target_ids.shape[1]
    
    # Optimize
    soft_prompt, last_loss = optimizer.optimize(args.text, n_prompt_tokens=args.n_prompt, n_steps=args.n_steps)
    
    # Project and Evaluate
    discrete_prompt_ids = optimizer.get_discrete_prompt(soft_prompt)
    
    with torch.no_grad():
        prompt_tensor = torch.tensor([discrete_prompt_ids], device=device)
        prompt_embeddings = wrapper.get_embeddings(prompt_tensor)
        target_embeddings = wrapper.get_embeddings(target_ids)
        
        full_embeddings = torch.cat([prompt_embeddings, target_embeddings[:, :-1]], dim=1)
        logits = wrapper.forward_with_embeddings(full_embeddings)
        target_logits = logits[:, args.n_prompt-1:, :]
        
        ranks = wrapper.get_token_ranks(target_logits, target_ids)
        
    metrics = CompressionMetrics.summary(discrete_prompt_ids, ranks, wrapper.model.config.vocab_size, target_ids[0].tolist())
    
    # Save artifacts
    torch.save({
        'soft_prompt': soft_prompt,
        'discrete_prompt_ids': discrete_prompt_ids,
        'ranks': ranks,
        'metrics': metrics,
        'text': args.text
    }, os.path.join(exp_dir, "data.pt"))
    
    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"Text: {args.text}\n")
        f.write(f"Discrete Prompt IDs: {discrete_prompt_ids}\n")
        f.write(f"Discrete Prompt Text: {wrapper.decode(torch.tensor([discrete_prompt_ids]))}\n")

    # Global log
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    log_to_global(
        date_str, args.model, target_len, args.n_prompt, 
        args.n_steps, metrics['accuracy'], metrics['compression_ratio'],
        args.exp_name, args.run_id
    )
    
    print(f"\nExperiment Complete!")
    print(f"Directory: {exp_dir}")
    print(f"Ratio: {metrics['compression_ratio']:.2f}x")
    print(f"Accuracy: {metrics['accuracy']:.2%}")

if __name__ == "__main__":
    main()
