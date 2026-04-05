import torch
import json
import os
import argparse
import datetime
from core.compressor.model_wrapper import HFModelWrapper
from core.compressor.optimizer import SoftPromptOptimizer
from core.compressor.metrics import CompressionMetrics

def log_experiment(date, model, target_len, prompt_len, steps, accuracy, ratio, exp_name, run_id):
    line = f"| {date} | {model} | {target_len} | {prompt_len} | {steps} | {accuracy*100.0:.1f}% | {ratio:.2f}x | Exp: {exp_name}, Run: {run_id} |\n"
    log_file = "dev_notes/compressor_experiments.md"
    with open(log_file, "a") as f:
        f.write(line)

def main():
    parser = argparse.ArgumentParser(description="Study Scaling Curve of LLM Compressor")
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps per run")
    parser.add_argument("--exp_name", type=str, default="scaling_study", help="Experiment name")
    parser.add_argument("--quick", action="store_true", help="Run only a small subset for verification")
    parser.add_argument("--hierarchical", action="store_true", help="Use hierarchical search for optimal n_prompt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    wrapper = HFModelWrapper(args.model, device=device)
    optimizer = SoftPromptOptimizer(wrapper)
    
    with open("data/wiki_samples.json", "r") as f:
        samples = json.load(f)
    
    prompt_lengths = [5, 10, 20, 50, 100]
    if args.quick:
        samples = {"short": samples["short"]}
        prompt_lengths = [10, 20]
        args.steps = 30

    date_str = datetime.date.today().strftime("%Y-%m-%d")
    exp_dir = os.path.join("logs", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    for text_key, text in samples.items():
        if text_key.startswith("_"): continue
        target_ids = wrapper.encode(text)
        target_len = target_ids.shape[1]
        
        if args.hierarchical:
            print(f"\n>>> Hierarchical Search for: {text_key} (len={target_len})")
            soft_prompt, n_prompt = optimizer.hierarchical_search(text, max_prompt_tokens=100, loss_threshold=0.01, steps_per_increment=args.steps)
            lengths_to_test = [n_prompt]
        else:
            lengths_to_test = prompt_lengths

        for n_prompt in lengths_to_test:
            run_id = f"{text_key}_p{n_prompt}"
            run_dir = os.path.join(exp_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            print(f"\n>>> Running: {text_key} (len={target_len}), prompt={n_prompt}")
            
            # Optimize with early stopping
            soft_prompt, _ = optimizer.optimize(text, n_prompt_tokens=n_prompt, n_steps=args.steps, loss_threshold=0.001)
            
            # Evaluate
            discrete_prompt_ids = optimizer.get_discrete_prompt(soft_prompt)
            with torch.no_grad():
                prompt_tensor = torch.tensor([discrete_prompt_ids], device=device)
                prompt_embeddings = wrapper.get_embeddings(prompt_tensor)
                target_embeddings = wrapper.get_embeddings(target_ids)
                full_embeddings = torch.cat([prompt_embeddings, target_embeddings[:, :-1]], dim=1)
                logits = wrapper.forward_with_embeddings(full_embeddings)
                target_logits = logits[:, n_prompt-1:, :]
                ranks = wrapper.get_token_ranks(target_logits, target_ids)
            
            metrics = CompressionMetrics.summary(discrete_prompt_ids, ranks, wrapper.model.config.vocab_size, target_ids[0].tolist())
            
            # Log
            log_experiment(date_str, args.model, target_len, n_prompt, args.steps, metrics['accuracy'], metrics['compression_ratio'], args.exp_name, run_id)
            
            # Save artifacts with full config
            config = {
                "model": args.model,
                "steps": args.steps,
                "n_prompt": n_prompt,
                "text_key": text_key,
                "device": device,
                "date": date_str
            }
            torch.save({
                'config': config,
                'metrics': metrics, 
                'prompt_ids': discrete_prompt_ids, 
                'ranks': ranks,
                'text': text
            }, os.path.join(run_dir, "data.pt"))
            
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
            print(f"Result: Ratio {metrics['compression_ratio']:.2f}x, Accuracy {metrics['accuracy']:.2%}")

if __name__ == "__main__":
    main()
