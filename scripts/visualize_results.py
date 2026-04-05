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

def visualize():
    df = parse_logs('dev_notes/compressor_experiments.md')
    if df is None or df.empty:
        return

    sns.set_theme(style="whitegrid")
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Ratio vs Prompt Len (Facetted by Model or colored by Target Len)
    sns.lineplot(ax=axes[0], data=df, x="prompt_len", y="ratio", hue="target_len", marker="o", palette="viridis")
    axes[0].set_title("Compression Ratio vs Prompt Length")
    axes[0].set_xlabel("Prompt Length (tokens)")
    axes[0].set_ylabel("Ratio (x)")
    axes[0].legend(title="Target Length")

    # 2. Accuracy vs Prompt Len
    sns.lineplot(ax=axes[1], data=df, x="prompt_len", y="accuracy", hue="target_len", marker="s", palette="viridis")
    axes[1].set_title("Accuracy vs Prompt Length")
    axes[1].set_xlabel("Prompt Length (tokens)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(title="Target Length")

    plt.tight_layout()
    plt.savefig("logs/scaling_visualization.png")
    print("Visualization saved to logs/scaling_visualization.png")
    plt.show()

if __name__ == "__main__":
    visualize()
