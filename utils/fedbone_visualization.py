import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


def plot_fedbone_metrics(results_path, save_dir):
    """Plot FedBone training metrics"""
    
    with open(results_path) as f:
        data = json.load(f)
    
    metrics = data['metrics']
    rounds = metrics['rounds']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(rounds, metrics['accuracy'], 'o-', linewidth=2, markersize=6, color='#2E86AB')
    axes[0, 0].set_xlabel('Round', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Test Accuracy over Rounds', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1-Score
    axes[0, 1].plot(rounds, metrics['f1'], 's-', linewidth=2, markersize=6, color='#A23B72')
    axes[0, 1].set_xlabel('Round', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('F1-Score over Rounds', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss
    axes[1, 0].plot(rounds, metrics['loss'], '^-', linewidth=2, markersize=6, color='#F18F01')
    axes[1, 0].set_xlabel('Round', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Training Loss over Rounds', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient Conflict Score
    axes[1, 1].plot(rounds, metrics['conflict_scores'], 'd-', linewidth=2, markersize=6, color='#C73E1D')
    axes[1, 1].set_xlabel('Round', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Gradient Conflict', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Gradient Conflict Score over Rounds', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'fedbone_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ FedBone metrics plot saved to {save_path}")
    plt.close()


def plot_gp_comparison(results_path, save_dir):
    """Compare GP Aggregation vs Baseline"""
    
    with open(results_path) as f:
        data = json.load(f)
    
    baseline = data['baseline']
    gp_agg = data['gp_aggregation']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy comparison
    axes[0].plot(baseline['rounds'], baseline['accuracy'], 'o-', 
                 label='Simple Averaging', linewidth=2, markersize=6, alpha=0.8)
    axes[0].plot(gp_agg['rounds'], gp_agg['accuracy'], 's-', 
                 label='GP Aggregation', linewidth=2, markersize=6, alpha=0.8)
    axes[0].set_xlabel('Round', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy: GP vs Baseline', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # F1-Score comparison
    axes[1].plot(baseline['rounds'], baseline['f1'], 'o-', 
                 label='Simple Averaging', linewidth=2, markersize=6, alpha=0.8)
    axes[1].plot(gp_agg['rounds'], gp_agg['f1'], 's-', 
                 label='GP Aggregation', linewidth=2, markersize=6, alpha=0.8)
    axes[1].set_xlabel('Round', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[1].set_title('F1-Score: GP vs Baseline', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    # Loss comparison
    axes[2].plot(baseline['rounds'], baseline['loss'], 'o-', 
                 label='Simple Averaging', linewidth=2, markersize=6, alpha=0.8)
    axes[2].plot(gp_agg['rounds'], gp_agg['loss'], 's-', 
                 label='GP Aggregation', linewidth=2, markersize=6, alpha=0.8)
    axes[2].set_xlabel('Round', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[2].set_title('Loss: GP vs Baseline', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11, loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'gp_vs_baseline.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ GP comparison plot saved to {save_path}")
    plt.close()


def plot_gradient_conflict_heatmap(results_path, save_dir):
    """Visualize gradient conflict evolution as heatmap"""
    
    with open(results_path) as f:
        data = json.load(f)
    
    if 'conflict_scores' not in data['metrics']:
        print("⚠ No conflict scores found in results")
        return
    
    conflict_scores = data['metrics']['conflict_scores']
    rounds = data['metrics']['rounds']
    
    # Create heatmap-style visualization
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Reshape for visualization (1 x num_rounds)
    conflict_matrix = np.array(conflict_scores).reshape(1, -1)
    
    im = ax.imshow(conflict_matrix, cmap='YlOrRd', aspect='auto', 
                   interpolation='nearest', vmin=0, vmax=max(conflict_scores))
    
    ax.set_xlabel('Training Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Conflict\nIntensity', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Conflict Evolution During Training', fontsize=14, fontweight='bold')
    
    # Set x-ticks
    tick_positions = np.linspace(0, len(rounds)-1, min(10, len(rounds)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([rounds[i] for i in tick_positions])
    ax.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label('Conflict Score', fontsize=11)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'gradient_conflict_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gradient conflict heatmap saved to {save_path}")
    plt.close()


def plot_architecture_diagram(save_dir):
    """Create FedBone architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'FedBone Architecture', 
            fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Client side (left)
    client_y = 0.5
    ax.add_patch(plt.Rectangle((0.05, client_y), 0.15, 0.3, 
                               facecolor='#E8F4F8', edgecolor='#2E86AB', linewidth=2))
    ax.text(0.125, client_y + 0.25, 'Client (Robot)', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.125, client_y + 0.2, 'Patch Embedding', fontsize=9, ha='center')
    ax.text(0.125, client_y + 0.15, '↓', fontsize=12, ha='center')
    ax.text(0.125, client_y + 0.1, 'Task Adaptation', fontsize=9, ha='center')
    ax.text(0.125, client_y + 0.05, '↓', fontsize=12, ha='center')
    ax.text(0.125, client_y, 'Task Head', fontsize=9, ha='center')
    
    # Server side (center)
    server_y = 0.6
    ax.add_patch(plt.Rectangle((0.35, server_y), 0.3, 0.25, 
                               facecolor='#FFF4E6', edgecolor='#F18F01', linewidth=2))
    ax.text(0.5, server_y + 0.2, 'Server (Cloud)', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.5, server_y + 0.15, 'General Model', fontsize=9, ha='center')
    ax.text(0.5, server_y + 0.1, '(Large-scale LSTM)', fontsize=8, ha='center', style='italic')
    ax.text(0.5, server_y + 0.05, 'GP Aggregation', fontsize=9, ha='center')
    
    # Multiple clients (right)
    for i, y_offset in enumerate([0.7, 0.5, 0.3]):
        ax.add_patch(plt.Rectangle((0.75, y_offset), 0.15, 0.15, 
                                   facecolor='#F4E8F8', edgecolor='#A23B72', linewidth=1.5))
        ax.text(0.825, y_offset + 0.075, f'Client {i+1}', fontsize=8, ha='center')
    
    # Arrows
    # Client to Server
    ax.annotate('', xy=(0.35, 0.7), xytext=(0.2, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'))
    ax.text(0.275, 0.68, 'embeddings', fontsize=8, ha='center')
    
    # Server to Client
    ax.annotate('', xy=(0.2, 0.6), xytext=(0.35, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2, color='#F18F01'))
    ax.text(0.275, 0.58, 'features', fontsize=8, ha='center')
    
    # Multi-client aggregation
    ax.annotate('', xy=(0.65, 0.725), xytext=(0.75, 0.775),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#A23B72'))
    ax.annotate('', xy=(0.65, 0.725), xytext=(0.75, 0.575),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#A23B72'))
    ax.annotate('', xy=(0.65, 0.725), xytext=(0.75, 0.375),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#A23B72'))
    
    # Legend
    ax.text(0.5, 0.15, 'Key Features:', fontsize=10, fontweight='bold', ha='center')
    ax.text(0.5, 0.1, '• Split Learning: Lightweight client, heavy server', fontsize=8, ha='center')
    ax.text(0.5, 0.07, '• GP Aggregation: Gradient projection for multi-task', fontsize=8, ha='center')
    ax.text(0.5, 0.04, '• Task Adaptation: Deformable Conv + Self-Attention', fontsize=8, ha='center')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'fedbone_architecture.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Architecture diagram saved to {save_path}")
    plt.close()


def create_all_fedbone_plots(results_dir):
    """Generate all FedBone visualization plots"""
    
    results_dir = Path(results_dir)
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fedbone_path = results_dir / 'fedbone_multitask_results.json'
    comparison_path = results_dir / 'fedbone_comparison_results.json'
    
    print("\nGenerating FedBone visualizations...")
    
    if fedbone_path.exists():
        plot_fedbone_metrics(fedbone_path, plots_dir)
        plot_gradient_conflict_heatmap(fedbone_path, plots_dir)
    else:
        print("⚠ FedBone results not found. Run experiments first.")
    
    if comparison_path.exists():
        plot_gp_comparison(comparison_path, plots_dir)
    else:
        print("⚠ Comparison results not found.")
    
    plot_architecture_diagram(plots_dir)
    
    print("\n✓ All FedBone plots generated successfully!")


if __name__ == "__main__":
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    create_all_fedbone_plots(results_dir)