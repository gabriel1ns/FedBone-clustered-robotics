import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_comparison(baseline_path, clustered_path, save_dir):
    """Compare baseline vs clustered FL performance"""
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(clustered_path) as f:
        clustered = json.load(f)
    
    # Extract metrics
    baseline_acc = [m[1]['accuracy'] for m in baseline['metrics']['metrics_centralized']]
    clustered_acc = clustered['metrics']['accuracy']
    
    baseline_rounds = list(range(1, len(baseline_acc) + 1))
    clustered_rounds = clustered['metrics']['rounds']
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy comparison
    axes[0].plot(baseline_rounds, baseline_acc, 'o-', label='FedAvg', linewidth=2, markersize=6)
    axes[0].plot(clustered_rounds, clustered_acc, 's-', label='Clustered FL', linewidth=2, markersize=6)
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # F1-Score comparison
    if 'f1' in clustered['metrics']:
        clustered_f1 = clustered['metrics']['f1']
        axes[1].plot(clustered_rounds, clustered_f1, 's-', label='Clustered FL', linewidth=2, markersize=6, color='orange')
        axes[1].set_xlabel('Round', fontsize=12)
        axes[1].set_ylabel('F1-Score', fontsize=12)
        axes[1].set_title('F1-Score over Rounds', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'comparison_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {save_path}")
    plt.close()


def plot_cluster_distribution(clustered_path, save_dir):
    """Visualize cluster distribution over rounds"""
    
    with open(clustered_path) as f:
        data = json.load(f)
    
    distributions = data['metrics']['cluster_distribution']
    rounds = data['metrics']['rounds']
    
    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    distributions_array = np.array(distributions).T
    num_clusters = distributions_array.shape[0]
    
    colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))
    
    ax.stackplot(rounds, distributions_array, labels=[f'Cluster {i}' for i in range(num_clusters)], 
                 colors=colors, alpha=0.8)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Number of Clients', fontsize=12)
    ax.set_title('Cluster Distribution Over Training Rounds', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'cluster_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Cluster distribution plot saved to {save_path}")
    plt.close()


def plot_convergence(baseline_path, clustered_path, save_dir):
    """Plot loss convergence"""
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(clustered_path) as f:
        clustered = json.load(f)
    
    baseline_loss = [loss[1] for loss in baseline['metrics']['losses_centralized']]
    clustered_loss = clustered['metrics']['loss']
    
    baseline_rounds = list(range(1, len(baseline_loss) + 1))
    clustered_rounds = clustered['metrics']['rounds']
    
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_rounds, baseline_loss, 'o-', label='FedAvg', linewidth=2, markersize=6)
    plt.plot(clustered_rounds, clustered_loss, 's-', label='Clustered FL', linewidth=2, markersize=6)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Convergence Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'loss_convergence.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss convergence plot saved to {save_path}")
    plt.close()


def create_all_plots(results_dir):
    """Generate all visualization plots"""
    
    results_dir = Path(results_dir)
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_path = results_dir / 'baseline_fedavg_results.json'
    clustered_path = results_dir / 'clustered_fl_results.json'
    
    if baseline_path.exists() and clustered_path.exists():
        print("\nGenerating visualizations...")
        plot_comparison(baseline_path, clustered_path, plots_dir)
        plot_cluster_distribution(clustered_path, plots_dir)
        plot_convergence(baseline_path, clustered_path, plots_dir)
        print("\n✓ All plots generated successfully!")
    else:
        print("⚠ Results files not found. Run experiments first.")


if __name__ == "__main__":
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    create_all_plots(results_dir)