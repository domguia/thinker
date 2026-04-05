import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_logs(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        if '| 20' in line and 'Exp:' in line:
            parts = [p.strip() for p in line.split('|')]
            try:
                data.append({
                    'model': parts[2],
                    'target_len': int(parts[3]),
                    'prompt_len': int(parts[4]),
                    'accuracy': float(parts[6].replace('%', '')),
                    'ratio': float(parts[7].replace('x', '')),
                    'run': parts[8]
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(data)

def get_rank_data(exp_name):
    """Load rank information from all data.pt files in an experiment."""
    exp_dir = f"logs/{exp_name}"
    if not os.path.exists(exp_dir):
        return None
    
    all_ranks = []
    for run_id in os.listdir(exp_dir):
        data_path = os.path.join(exp_dir, run_id, "data.pt")
        if os.path.exists(data_path):
            try:
                data = torch.load(data_path, map_location='cpu')
                ranks = data['ranks'].view(-1).numpy()
                all_ranks.extend(ranks)
            except Exception:
                continue
    return all_ranks

def visualize(exp_name=None):
    df = parse_logs('dev_notes/compressor_experiments.md')
    if df is None or df.empty:
        return

    sns.set_theme(style="whitegrid")
    
    # 1. Main scaling plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.lineplot(ax=axes[0], data=df, x="prompt_len", y="ratio", hue="target_len", marker="o", palette="viridis")
    axes[0].set_title("Compression Ratio vs Prompt Length")
    axes[0].set_ylabel("Ratio (x)")
    
    sns.lineplot(ax=axes[1], data=df, x="prompt_len", y="accuracy", hue="target_len", marker="s", palette="viridis")
    axes[1].set_title("Accuracy (Top-1) vs Prompt Length")
    axes[1].set_ylabel("Accuracy (%)")
    
    plt.tight_layout()
    plt.savefig("logs/scaling_curves.png")
    
    # 2. Rank Distribution (if exp_name provided)
    if exp_name:
        ranks = get_rank_data(exp_name)
        if ranks:
            plt.figure(figsize=(10, 6))
            # Focus on small ranks where most mass should be
            sns.histplot([r for r in ranks if r < 50], bins=50, kde=False, color='skyblue')
            plt.title(f"Distribution of Token Ranks (Top-50) - {exp_name}")
            plt.xlabel("Rank (0 = Correct Prediction)")
            plt.ylabel("Frequency")
            plt.savefig(f"logs/{exp_name}_rank_dist.png")
            print(f"Rank distribution saved to logs/{exp_name}_rank_dist.png")

    print("Scaling curves saved to logs/scaling_curves.png")
    plt.show()

if __name__ == "__main__":
    visualize()
