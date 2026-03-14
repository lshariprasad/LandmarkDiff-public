"""Plot training curves from experiment tracker JSONL metrics.

Generates publication-quality training curves showing loss, learning rate,
and validation metrics over training steps.

Usage:
    # Plot from experiment tracker directory
    python scripts/plot_training_curves.py --experiments_dir checkpoints/experiments

    # Plot from specific JSONL metrics file
    python scripts/plot_training_curves.py --metrics_file checkpoints/experiments/exp_001_metrics.jsonl

    # Compare multiple experiments
    python scripts/plot_training_curves.py --experiments_dir checkpoints/experiments --compare exp_001 exp_002
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_metrics(metrics_path: Path) -> list[dict]:
    """Load metrics from a JSONL file."""
    entries = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def smooth(values: list[float], window: int = 10) -> list[float]:
    """Exponential moving average smoothing."""
    if not values or window <= 1:
        return values
    smoothed = []
    ema = values[0]
    alpha = 2.0 / (window + 1)
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed


def plot_single_experiment(
    metrics: list[dict],
    output_path: str,
    title: str = "Training Curves",
    smooth_window: int = 20,
) -> None:
    """Plot training curves for a single experiment."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
        return

    steps = [m["step"] for m in metrics if m.get("step") is not None]
    losses = [m["loss"] for m in metrics if "loss" in m and m.get("step") is not None]
    lrs = [m["lr"] for m in metrics if "lr" in m and m.get("step") is not None]

    if not steps:
        print("No metrics with step data found.")
        return

    # Create figure with subplots
    n_plots = 2  # loss + lr
    has_val_ssim = any("val_ssim" in m or "ssim" in m for m in metrics)
    has_val_lpips = any("val_lpips" in m or "lpips" in m for m in metrics)
    if has_val_ssim:
        n_plots += 1
    if has_val_lpips:
        n_plots += 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    # Plot 1: Loss
    ax = axes[0]
    if losses:
        loss_steps = [m["step"] for m in metrics if "loss" in m and m.get("step") is not None]
        ax.plot(loss_steps, losses, alpha=0.3, color="steelblue", linewidth=0.5)
        ax.plot(
            loss_steps,
            smooth(losses, smooth_window),
            color="steelblue",
            linewidth=2,
            label="Loss (smoothed)",
        )
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    # Plot 2: Learning Rate
    ax = axes[1]
    if lrs:
        lr_steps = [m["step"] for m in metrics if "lr" in m and m.get("step") is not None]
        ax.plot(lr_steps, lrs, color="coral", linewidth=1.5)
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    # Plot 3: Validation SSIM (if available)
    plot_idx = 2
    if has_val_ssim:
        ax = axes[plot_idx]
        val_steps = []
        val_ssim = []
        for m in metrics:
            s = m.get("step")
            v = m.get("val_ssim") or m.get("ssim")
            if s is not None and v is not None:
                val_steps.append(s)
                val_ssim.append(v)
        if val_steps:
            ax.plot(val_steps, val_ssim, "o-", color="forestgreen", markersize=4, linewidth=1.5)
            ax.set_ylabel("SSIM")
            ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 4: Validation LPIPS (if available)
    if has_val_lpips:
        ax = axes[plot_idx]
        val_steps = []
        val_lpips = []
        for m in metrics:
            s = m.get("step")
            v = m.get("val_lpips") or m.get("lpips")
            if s is not None and v is not None:
                val_steps.append(s)
                val_lpips.append(v)
        if val_steps:
            ax.plot(val_steps, val_lpips, "s-", color="darkorange", markersize=4, linewidth=1.5)
            ax.set_ylabel("LPIPS")
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Training Step")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {output_path}")


def plot_comparison(
    experiments: dict[str, list[dict]],
    output_path: str,
    smooth_window: int = 20,
) -> None:
    """Plot training curves comparing multiple experiments."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed.")
        return

    colors = ["steelblue", "coral", "forestgreen", "darkorange", "purple", "brown"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Loss comparison
    ax = axes[0]
    for i, (exp_name, metrics) in enumerate(experiments.items()):
        steps = [m["step"] for m in metrics if "loss" in m and m.get("step") is not None]
        losses = [m["loss"] for m in metrics if "loss" in m and m.get("step") is not None]
        if steps:
            color = colors[i % len(colors)]
            ax.plot(steps, smooth(losses, smooth_window), color=color, linewidth=2, label=exp_name)
    ax.set_ylabel("Loss")
    ax.set_title("Experiment Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # LR comparison
    ax = axes[1]
    for i, (exp_name, metrics) in enumerate(experiments.items()):
        steps = [m["step"] for m in metrics if "lr" in m and m.get("step") is not None]
        lrs = [m["lr"] for m in metrics if "lr" in m and m.get("step") is not None]
        if steps:
            color = colors[i % len(colors)]
            ax.plot(steps, lrs, color=color, linewidth=1.5, label=exp_name)
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Training Step")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {output_path}")


def print_summary(metrics: list[dict], exp_name: str = "Experiment") -> None:
    """Print a text summary of training metrics."""
    if not metrics:
        print(f"  {exp_name}: No metrics found")
        return

    steps = [m.get("step", 0) for m in metrics]
    losses = [m["loss"] for m in metrics if "loss" in m]

    print(f"\n  {exp_name}:")
    print(f"    Steps: {min(steps)} -> {max(steps)} ({len(metrics)} entries)")
    if losses:
        print(
            f"    Loss: {losses[0]:.6f} -> {losses[-1]:.6f} "
            f"(min: {min(losses):.6f}, mean: {np.mean(losses):.6f})"
        )
    # Final validation metrics
    for key in ["val_ssim", "ssim", "val_lpips", "lpips"]:
        vals = [m[key] for m in metrics if key in m]
        if vals:
            print(f"    {key}: {vals[-1]:.4f} (best: {max(vals):.4f})")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument(
        "--experiments_dir", default=None, help="Path to experiment tracker directory"
    )
    parser.add_argument("--metrics_file", default=None, help="Path to a single JSONL metrics file")
    parser.add_argument(
        "--compare",
        nargs="*",
        default=None,
        help="Experiment IDs to compare (e.g., exp_001 exp_002)",
    )
    parser.add_argument("--output", default="training_curves.png", help="Output image path")
    parser.add_argument("--smooth", type=int, default=20, help="Smoothing window for loss curve")
    args = parser.parse_args()

    if args.metrics_file:
        metrics_path = Path(args.metrics_file)
        if not metrics_path.exists():
            print(f"ERROR: Metrics file not found: {metrics_path}")
            sys.exit(1)
        metrics = load_metrics(metrics_path)
        print_summary(metrics, metrics_path.stem)
        plot_single_experiment(metrics, args.output, smooth_window=args.smooth)
        return

    if args.experiments_dir:
        exp_dir = Path(args.experiments_dir)
        index_path = exp_dir / "index.json"
        if not index_path.exists():
            print(f"ERROR: No index.json found in {exp_dir}")
            sys.exit(1)

        with open(index_path) as f:
            index = json.load(f)

        if args.compare:
            # Compare specific experiments
            experiments = {}
            for exp_id in args.compare:
                exp = index["experiments"].get(exp_id)
                if exp:
                    mf = exp_dir / exp["metrics_file"]
                    if mf.exists():
                        experiments[f"{exp_id} ({exp['name']})"] = load_metrics(mf)
                        print_summary(experiments[f"{exp_id} ({exp['name']})"], exp_id)
                else:
                    print(f"WARNING: Experiment {exp_id} not found")
            if experiments:
                plot_comparison(experiments, args.output, smooth_window=args.smooth)
        else:
            # Plot the latest experiment
            exp_ids = sorted(index["experiments"].keys())
            if not exp_ids:
                print("No experiments found.")
                sys.exit(1)
            latest = exp_ids[-1]
            exp = index["experiments"][latest]
            mf = exp_dir / exp["metrics_file"]
            if mf.exists():
                metrics = load_metrics(mf)
                print_summary(metrics, latest)
                plot_single_experiment(
                    metrics,
                    args.output,
                    title=f"{latest}: {exp['name']}",
                    smooth_window=args.smooth,
                )
        return

    print("ERROR: Provide --experiments_dir or --metrics_file")
    sys.exit(1)


if __name__ == "__main__":
    main()
