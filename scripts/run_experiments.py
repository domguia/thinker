import torch
import torch.nn as nn
from core.compressor.model_wrapper import HFModelWrapper
from core.compressor.optimizer import SoftPromptOptimizer
from core.compressor.metrics import CompressionMetrics
import datetime
import os

def log_experiment(date, model, target_len, prompt_len, steps, accuracy, ratio, notes):
    line = f"| {date} | {model} | {target_len} | {prompt_len} | {steps} | {accuracy*100.0:.1f}% | {ratio:.2f}x | {notes} |\n"
    with open("dev_notes/compressor_experiments.md", "a") as f:
        f.write(line)

def run_experiment(wrapper, text, n_prompt, n_steps=100):
    optimizer = SoftPromptOptimizer(wrapper)
    target_ids = wrapper.encode(text)
    target_len = target_ids.shape[1]
    
    # 1. Optimize soft prompt
    soft_prompt, _ = optimizer.optimize(text, n_prompt_tokens=n_prompt, n_steps=n_steps)
    
    # 2. Project to discrete
    discrete_prompt_ids = optimizer.get_discrete_prompt(soft_prompt)
    
    # 3. Evaluate discrete prompt
    with torch.no_grad():
        prompt_tensor = torch.tensor([discrete_prompt_ids], device=wrapper.device)
        prompt_embeddings = wrapper.get_embeddings(prompt_tensor)
        target_embeddings = wrapper.get_embeddings(target_ids)
        
        full_embeddings = torch.cat([prompt_embeddings, target_embeddings[:, :-1]], dim=1)
        logits = wrapper.forward_with_embeddings(full_embeddings)
        target_logits = logits[:, n_prompt-1:, :]
        
        ranks = wrapper.get_token_ranks(target_logits, target_ids)
        
    metrics = CompressionMetrics.summary(discrete_prompt_ids, ranks, wrapper.model.config.vocab_size, target_ids[0].tolist())
    return metrics, target_len

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    wrapper = HFModelWrapper("gpt2", device=device)
    
    texts = [
        "The Sun is the star at the center of the Solar System.",
        "The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy mainly as visible light, ultraviolet light, and infrared radiation."
    ]
    
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    
    for text in texts:
        n_prompt = 10
        n_steps = 30
        print(f"\nRunning: Len {len(text)}, Prompt {n_prompt}")
        metrics, target_tokens = run_experiment(wrapper, text, n_prompt, n_steps)
        log_experiment(
            date_str, "GPT-2", target_tokens, n_prompt, 
            n_steps, metrics['accuracy'], metrics['compression_ratio'],
            f"Text: '{text[:20]}...'"
        )
        print(f"Result: Ratio {metrics['compression_ratio']:.2f}x, Acc {metrics['accuracy']:.2f}")

if __name__ == "__main__":
    main()
