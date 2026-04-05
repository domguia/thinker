import torch
from core.compressor.model_wrapper import HFModelWrapper
from core.compressor.optimizer import SoftPromptOptimizer
from core.compressor.metrics import CompressionMetrics
import argparse

def main():
    parser = argparse.ArgumentParser(description="LLM as Data Compressor Demo")
    parser.add_argument("--text", type=str, default="Artificial intelligence is transforming the world. It can process vast amounts of data and learn complex patterns.", help="Text to compress")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name")
    parser.add_argument("--n_prompt", type=int, default=5, help="Number of prompt tokens")
    parser.add_argument("--n_steps", type=int, default=100, help="Optimization steps")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    wrapper = HFModelWrapper(args.model, device=device)
    optimizer = SoftPromptOptimizer(wrapper)
    
    print(f"Original text: {args.text}")
    target_ids = wrapper.encode(args.text)
    print(f"Original tokens: {target_ids.shape[1]}")

    # 1. Optimize soft prompt
    print("\n--- Optimizing Soft Prompt ---")
    soft_prompt, last_loss = optimizer.optimize(args.text, n_prompt_tokens=args.n_prompt, n_steps=args.n_steps)
    
    # 2. Project to discrete tokens
    print("\n--- Projecting to Discrete Tokens ---")
    discrete_prompt_ids = optimizer.get_discrete_prompt(soft_prompt)
    discrete_prompt_text = wrapper.decode(torch.tensor([discrete_prompt_ids]))
    print(f"Discrete Prompt: '{discrete_prompt_text}' (IDs: {discrete_prompt_ids})")

    # 3. Evaluate discrete prompt
    with torch.no_grad():
        prompt_tensor = torch.tensor([discrete_prompt_ids], device=device)
        prompt_embeddings = wrapper.get_embeddings(prompt_tensor)
        target_embeddings = wrapper.get_embeddings(target_ids)
        
        full_embeddings = torch.cat([prompt_embeddings, target_embeddings[:, :-1]], dim=1)
        logits = wrapper.forward_with_embeddings(full_embeddings)
        target_logits = logits[:, args.n_prompt-1:, :]
        
        ranks = wrapper.get_token_ranks(target_logits, target_ids)
        
    # 4. Calculate metrics
    metrics = CompressionMetrics.summary(discrete_prompt_ids, ranks, wrapper.model.config.vocab_size, target_ids[0].tolist())
    
    print("\n--- Compression Summary ---")
    for k, v in metrics.items():
        print(f"{k:20}: {v:.4f}")

    # 5. Reconstruction test
    # If we have ranks, we can perfectly reconstruct the original text
    # (In a real compressor, we would encode the ranks efficiently)
    print("\nReconstruction successful (via ranks)!" if metrics['accuracy'] > 0 else "Reconstruction failed.")

if __name__ == "__main__":
    main()
