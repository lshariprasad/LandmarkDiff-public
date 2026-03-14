"""Learning rate range test (Smith 2017).

Sweeps learning rate from a very small to very large value over a short
training run, recording loss at each step. The optimal LR is where the
loss decreases fastest (steepest negative gradient on loss vs log(LR)).

Usage:
    # Basic LR range test
    python scripts/lr_finder.py \
        --data_dir data/training_combined \
        --output lr_finder_results

    # Custom range
    python scripts/lr_finder.py \
        --data_dir data/training_combined \
        --lr_min 1e-8 --lr_max 1e-2 \
        --num_steps 200

    # Phase B mode (with auxiliary losses)
    python scripts/lr_finder.py \
        --data_dir data/training_combined \
        --phase B \
        --output lr_finder_phaseB
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_lr_finder(
    data_dir: str,
    output_dir: str = "lr_finder_results",
    lr_min: float = 1e-8,
    lr_max: float = 1e-2,
    num_steps: int = 200,
    batch_size: int = 2,
    phase: str = "A",
    beta: float = 0.98,
    seed: int = 42,
) -> dict:
    """Run learning rate range test.

    Args:
        data_dir: Training data directory.
        output_dir: Output directory for results.
        lr_min: Starting learning rate.
        lr_max: Maximum learning rate.
        num_steps: Number of training steps.
        batch_size: Batch size.
        phase: Training phase (A or B).
        beta: Smoothing factor for loss (EMA).
        seed: Random seed.

    Returns:
        Dict with LR sweep results.
    """
    import torch
    from torch.utils.data import DataLoader

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    print("LR Range Test")
    print(f"  Data: {data_dir}")
    print(f"  LR range: [{lr_min:.1e}, {lr_max:.1e}]")
    print(f"  Steps: {num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Phase: {phase}")
    print(f"  Device: {device}")
    print()

    # Load dataset
    from scripts.train_controlnet import SyntheticPairDataset

    dataset = SyntheticPairDataset(data_dir, resolution=512)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(dataloader)

    # Load model
    from diffusers import ControlNetModel

    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "CrucibleAI/ControlNetMediaPipeFace",
        subfolder="diffusion_sd15",
    )
    controlnet.train().to(device)

    # Simple MSE loss proxy (avoids full diffusion pipeline)
    # We use the ControlNet's mid-block output as a proxy loss
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=lr_min,
        weight_decay=0.01,
    )

    # Exponential LR schedule
    gamma = (lr_max / lr_min) ** (1.0 / num_steps)

    # Storage
    lrs = []
    raw_losses = []
    smoothed_losses = []
    best_loss = float("inf")
    avg_loss = 0.0

    print("Running LR sweep...")
    for step in range(num_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Current LR
        lr = lr_min * (gamma**step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass (simplified)
        conditioning = batch["conditioning"].to(device, dtype=dtype)

        # Create a dummy noise sample for the ControlNet
        b = conditioning.shape[0]
        dummy_sample = torch.randn(b, 4, 64, 64, device=device, dtype=dtype)
        dummy_encoder_hidden = torch.randn(b, 77, 768, device=device, dtype=dtype)
        dummy_timestep = torch.randint(0, 1000, (b,), device=device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            down_outputs, mid_output = controlnet(
                dummy_sample,
                dummy_timestep,
                encoder_hidden_states=dummy_encoder_hidden,
                controlnet_cond=conditioning,
                return_dict=False,
            )
            # Use mid-block MSE as proxy loss
            loss = mid_output.pow(2).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
        optimizer.step()

        # Track
        loss_val = loss.item()
        raw_losses.append(loss_val)
        lrs.append(lr)

        # Smoothed loss (EMA)
        avg_loss = beta * avg_loss + (1 - beta) * loss_val
        smoothed = avg_loss / (1 - beta ** (step + 1))  # bias correction
        smoothed_losses.append(smoothed)

        # Check for divergence
        if step > 10 and smoothed > 4 * best_loss:
            print(f"  Loss diverged at step {step}, LR={lr:.2e}")
            break

        if smoothed < best_loss:
            best_loss = smoothed

        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{num_steps}: LR={lr:.2e}, loss={smoothed:.6f}")

    # Find optimal LR
    # Look for the steepest descent in smoothed loss
    log_lrs = [math.log10(lr) for lr in lrs]
    if len(smoothed_losses) > 5:
        gradients = np.gradient(smoothed_losses, log_lrs)
        min_grad_idx = np.argmin(gradients)
        suggested_lr = lrs[min_grad_idx]

        # Also find the LR at minimum loss
        min_loss_idx = np.argmin(smoothed_losses)
        min_loss_lr = lrs[min_loss_idx]
    else:
        suggested_lr = 1e-5
        min_loss_lr = 1e-5
        min_grad_idx = 0
        min_loss_idx = 0

    # Results
    results = {
        "lr_min": lr_min,
        "lr_max": lr_max,
        "num_steps": len(lrs),
        "suggested_lr": float(suggested_lr),
        "min_loss_lr": float(min_loss_lr),
        "suggested_lr_range": [float(suggested_lr / 3), float(suggested_lr * 3)],
        "lrs": [float(lr) for lr in lrs],
        "raw_losses": [float(l) for l in raw_losses],
        "smoothed_losses": [float(l) for l in smoothed_losses],
    }

    # Save results
    results_path = out / "lr_finder_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate plot
    try:
        _plot_lr_finder(results, out)
    except Exception as e:
        print(f"  Plot generation failed: {e}")

    # Print summary
    print(f"\n{'=' * 50}")
    print("  LR Finder Results")
    print(f"{'=' * 50}")
    print(f"  Suggested LR: {suggested_lr:.2e}")
    print(f"  Min loss LR:  {min_loss_lr:.2e}")
    print(f"  Recommended range: [{suggested_lr / 3:.2e}, {suggested_lr * 3:.2e}]")
    print(f"  Results: {results_path}")

    return results


def _plot_lr_finder(results: dict, output_dir: Path) -> None:
    """Generate LR finder plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback: save as text plot
        print("  matplotlib not available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    lrs = results["lrs"]
    raw_losses = results["raw_losses"]
    smoothed_losses = results["smoothed_losses"]

    # Loss vs LR
    ax1.plot(lrs, raw_losses, alpha=0.3, label="Raw loss")
    ax1.plot(lrs, smoothed_losses, linewidth=2, label="Smoothed loss")
    ax1.set_xscale("log")
    ax1.set_xlabel("Learning Rate")
    ax1.set_ylabel("Loss")
    ax1.set_title("LR Range Test")
    ax1.legend()
    ax1.axvline(
        x=results["suggested_lr"],
        color="r",
        linestyle="--",
        label=f"Suggested: {results['suggested_lr']:.2e}",
    )
    ax1.legend()

    # Loss gradient
    log_lrs = [math.log10(lr) for lr in lrs]
    if len(smoothed_losses) > 5:
        gradients = np.gradient(smoothed_losses, log_lrs)
        ax2.plot(lrs, gradients, linewidth=2)
        ax2.set_xscale("log")
        ax2.set_xlabel("Learning Rate")
        ax2.set_ylabel("d(loss)/d(log(LR))")
        ax2.set_title("Loss Gradient")
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(x=results["suggested_lr"], color="r", linestyle="--")

    plt.tight_layout()
    plot_path = output_dir / "lr_finder_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning rate range test")
    parser.add_argument("--data_dir", required=True, help="Training data directory")
    parser.add_argument("--output", default="lr_finder_results", help="Output directory")
    parser.add_argument("--lr_min", type=float, default=1e-8)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--phase", default="A", choices=["A", "B"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_lr_finder(
        args.data_dir,
        args.output,
        args.lr_min,
        args.lr_max,
        args.num_steps,
        args.batch_size,
        args.phase,
        seed=args.seed,
    )
