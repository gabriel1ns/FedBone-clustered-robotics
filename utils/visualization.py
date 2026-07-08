import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_style("whitegrid")


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def plot_comparison(baseline_path, clustered_path, save_dir):
    fedavg = _load(baseline_path)["metrics"]
    clustered = _load(clustered_path)["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for metrics, label, marker in (
        (fedavg, "FedAvg", "o"),
        (clustered, "Clustered FL", "s"),
    ):
        axes[0].plot(metrics["rounds"], metrics["rmse"], marker=marker, label=label)
        axes[1].plot(
            metrics["rounds"], metrics["task_success_rate"], marker=marker, label=label
        )

    axes[0].set(title="RoboMimic action RMSE", xlabel="Round", ylabel="Macro RMSE")
    axes[1].set(title="Offline task success", xlabel="Round", ylabel="Macro TSR")
    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(save_dir) / "comparison_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_distribution(clustered_path, save_dir):
    metrics = _load(clustered_path)["metrics"]
    task_histories = metrics["per_task"]
    if not task_histories:
        return

    task_name, history = next(iter(task_histories.items()))
    cluster_ids = sorted({
        cluster_id
        for distribution in history["cluster_distribution"]
        for cluster_id in distribution
    })
    values = np.asarray([
        [distribution.get(cluster_id, 0) for distribution in history["cluster_distribution"]]
        for cluster_id in cluster_ids
    ])

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.stackplot(
        history["rounds"],
        values,
        labels=[f"Cluster {cluster_id}" for cluster_id in cluster_ids],
        alpha=0.8,
    )
    axis.set(
        title=f"Cluster distribution — {task_name}",
        xlabel="Round",
        ylabel="Robot clients",
    )
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "cluster_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(baseline_path, clustered_path, save_dir):
    fedavg = _load(baseline_path)["metrics"]
    clustered = _load(clustered_path)["metrics"]
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(fedavg["rounds"], fedavg["loss"], "o-", label="FedAvg")
    axis.plot(clustered["rounds"], clustered["loss"], "s-", label="Clustered FL")
    axis.set(
        title="Normalized action-loss convergence",
        xlabel="Round",
        ylabel="Macro MSE loss",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "loss_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_all_plots(results_dir):
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = results_dir / "baseline_fedavg_results.json"
    clustered_path = results_dir / "clustered_fl_results.json"

    if not baseline_path.exists() or not clustered_path.exists():
        print("Results files not found. Run runner/run_experiments.py first.")
        return

    plot_comparison(baseline_path, clustered_path, plots_dir)
    plot_cluster_distribution(clustered_path, plots_dir)
    plot_convergence(baseline_path, clustered_path, plots_dir)
    print(f"Baseline plots saved to {plots_dir}")


if __name__ == "__main__":
    create_all_plots(Path(__file__).parent.parent / "results")
