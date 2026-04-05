import torch
import time
import json
import os
from core.compressor.model_wrapper import HFModelWrapper
from core.compressor.optimizer import SoftPromptOptimizer
from core.compressor.metrics import CompressionMetrics
import pandas as pd

def run_deep_convergence_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing model on {device}...")
    wrapper = HFModelWrapper("gpt2", device=device)
    optimizer = SoftPromptOptimizer(wrapper)
    
    # Load samples
    with open("data/wiki_samples.json", "r") as f:
        samples = json.load(f)
    
    # We select specific samples to see the length impact
    test_texts = {
        "short": samples["short"],
        "medium": samples["medium"],
        "long": samples["long"]
    }
    
    # Thresholds to test
    thresholds = [0.01, 0.001, 0.0001, 0.00001]
    n_prompt = 10
    
    all_results = []
    
    print(f"\n--- Deep Convergence & Length Impact Test ---")
    print(f"Prompt Length: {n_prompt} tokens")
    
    start_all = time.time()
    
    for text_key, text in test_texts.items():
        target_ids = wrapper.encode(text)
        target_len = target_ids.shape[1]
        print(f"\n========================================")
        print(f"TEXT: {text_key} ({target_len} tokens)")
        print(f"========================================")
        
        for thresh in thresholds:
            print(f"\n>>> Target Soft Loss: {thresh}")
            start_stage = time.time()
            
            # Optimize
            soft_prompt, final_loss = optimizer.optimize(text, n_prompt_tokens=n_prompt, n_steps=1500, 
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
            
            res = {
                "text_type": text_key,
                "target_len": target_len,
                "target_thresh": thresh,
                "final_soft_loss": f"{final_loss:.6f}",
                "discrete_acc": f"{acc*100.0:.2f}%",
                "time": f"{stage_time:.1f}s"
            }
            all_results.append(res)
            print(f"RESULT [{text_key}]: Loss {final_loss:.6f} -> Discrete Acc: {acc*100.0:.2f}%")

    total_time = time.time() - start_all
    df = pd.DataFrame(all_results)
    print("\n--- Final Multi-Length Summary ---")
    print(df.to_string(index=False))
    
    # Save results for later analysis
    df.to_csv("logs/convergence_length_impact.csv", index=False)
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    run_deep_convergence_test()
