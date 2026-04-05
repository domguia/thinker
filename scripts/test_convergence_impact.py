import torch
import time
from core.compressor.model_wrapper import HFModelWrapper
from core.compressor.optimizer import SoftPromptOptimizer
from core.compressor.metrics import CompressionMetrics
import pandas as pd

def run_deep_convergence_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing model on {device}...")
    wrapper = HFModelWrapper("gpt2", device=device)
    optimizer = SoftPromptOptimizer(wrapper)
    
    text = "The Moon is Earth's only natural satellite."
    target_ids = wrapper.encode(text)
    n_prompt = 10
    
    thresholds = [0.1, 0.01, 0.001, 0.0001, 0.00005]
    results = []
    
    print(f"\n--- Deep Convergence Test ---")
    print(f"Text: '{text}'")
    print(f"Target Length: {target_ids.shape[1]} tokens")
    print(f"Prompt Length: {n_prompt} tokens")
    print(f"Estimated total time: ~5-10 minutes on CPU")
    
    start_all = time.time()
    
    for thresh in thresholds:
        print(f"\n>>> Target Soft Loss: {thresh}")
        start_stage = time.time()
        
        # Optimize
        soft_prompt, final_loss = optimizer.optimize(text, n_prompt_tokens=n_prompt, n_steps=1000, 
                                                     loss_threshold=thresh, verbose=True)
        
        # Eval Discrete
        discrete_ids = optimizer.get_discrete_prompt(soft_prompt)
        with torch.no_grad():
            prompt_tensor = torch.tensor([discrete_ids], device=device)
            p_emb = wrapper.get_embeddings(prompt_tensor)
            t_emb = wrapper.get_embeddings(target_ids)
            full = torch.cat([p_emb, t_emb[:, :-1]], dim=1)
            logits = wrapper.forward_with_embeddings(full)[:, n_prompt-1:, :]
            ranks = wrapper.get_token_ranks(logits, target_ids)
            acc = (ranks == 0).float().mean().item()
            
        stage_time = time.time() - start_stage
        results.append({
            "target_thresh": thresh,
            "final_soft_loss": f"{final_loss:.6f}",
            "discrete_accuracy": f"{acc*100.2:.2f}%",
            "time_sec": f"{stage_time:.1f}s"
        })
        print(f"RESULT: Loss {final_loss:.6f} -> Discrete Acc: {acc*100.0:.2f}% (Took {stage_time:.1f}s)")

    total_time = time.time() - start_all
    df = pd.DataFrame(results)
    print("\n--- Final Summary Table ---")
    print(df)
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    run_deep_convergence_test()
