"""plot_qualitative.py

Plot qualitative forecasts for 10 randomly selected test products at n_obs=0.

For each sample, three curves are drawn on the same panel:
  - Ground Truth  : the actual 12-week normalised sales trajectory
  - K-NN Mean     : average of the 3 pre-retrieved reference trajectories
  - Cold-RATD     : diffusion median (solid) + individual nsample draws (faint)

Usage
-----
    python plot_qualitative.py \\
        --modelfolder save/fashion_visuelle2_n0_20260414_111340 \\
        --n_samples 5 \\
        --n_plots 10 \\
        --seed 42 \\
        --out_dir plots/qualitative_n0

All paths are relative to the repo root
(Cold-Start-Demand-Forecasting-Using-Retrieval-Augmented-Diffusion-Models/).
"""

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

from dataset_visuelle2 import Dataset_Visuelle2
from main_model_fashion import RATD_Fashion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config_and_model(modelfolder: str, device: str) -> RATD_Fashion:
    """Load config.json and model_best.pth from a run folder."""
    config_path = os.path.join(modelfolder, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Strip CLI-only keys that RATD_Fashion does not expect
    config.pop("cli", None)

    model = RATD_Fashion(config, device)
    model = model.to(device)

    ckpt_path = os.path.join(modelfolder, "model_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(modelfolder, "model.pth")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from: {ckpt_path}")
    return model


def _knn_mean_pred(reference_tensor: torch.Tensor) -> np.ndarray:
    """Compute K-NN Mean prediction from the reference tensor.

    Args:
        reference_tensor: shape (36, 2) from Dataset_Visuelle2,
                          layout (k*pred_len, K) where k=3, pred_len=12.

    Returns:
        np.ndarray of shape (12,) – sales channel (col 0) KNN mean.
    """
    ref_np = reference_tensor.numpy()        # (36, 2)
    sales  = ref_np[:, 0]                    # (36,)  sales channel
    refs   = np.split(sales, 3)              # 3 x (12,)
    return np.mean(refs, axis=0)             # (12,)


def _run_diffusion(model, batch_single: dict, n_samples: int,
                   device: str) -> np.ndarray:
    """Run Cold-RATD on a single sample and return all n_samples draws.

    Returns:
        np.ndarray of shape (n_samples, 12) – normalised sales channel.
    """
    # Add batch dimension to every tensor value
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
             for k, v in batch_single.items()}

    # evaluate() returns (samples, observed_data, target_mask, observed_mask, observed_tp)
    samples, *_ = model.evaluate(batch, n_samples=n_samples)
    # samples shape: (B=1, n_samples, K=2, L=12)
    samples_np = samples[0, :, 0, :].cpu().numpy()   # (n_samples, 12)
    return samples_np


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

WEEKS = np.arange(1, 13)

COLOR_GT    = "#2c7bb6"    # blue   – ground truth
COLOR_KNN   = "#d7191c"    # red    – K-NN Mean
COLOR_RATD  = "#1a9641"    # green  – Cold-RATD median
COLOR_SHADE = "#a8ddb5"    # light green – individual RATD draws


def _plot_sample(ax, gt: np.ndarray, knn: np.ndarray,
                 ratd_samples: np.ndarray, sample_idx: int,
                 n_obs: int = 0) -> None:
    """Draw one panel: GT vs KNN Mean vs Cold-RATD samples."""

    ratd_median = np.median(ratd_samples, axis=0)  # (12,)

    # Individual diffusion samples (faint)
    for s in ratd_samples:
        ax.plot(WEEKS, s, color=COLOR_SHADE, alpha=0.35, linewidth=0.8,
                zorder=1)

    # K-NN Mean
    ax.plot(WEEKS, knn, color=COLOR_KNN, linewidth=1.6,
            linestyle="--", label="K-NN Mean", zorder=3)

    # Cold-RATD median
    ax.plot(WEEKS, ratd_median, color=COLOR_RATD, linewidth=1.8,
            label="Cold-RATD (median)", zorder=4)

    # Ground truth
    ax.plot(WEEKS, gt, color=COLOR_GT, linewidth=1.8,
            linestyle="-", label="Ground Truth", zorder=5)

    # Shade the observed region (n_obs weeks)
    if n_obs > 0:
        ax.axvspan(0.5, n_obs + 0.5, color="#f7f7f7", zorder=0, label="Observed")

    ax.set_title(f"Sample {sample_idx}", fontsize=9, pad=3)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Week", fontsize=8)
    ax.set_ylabel("Norm. Sales", fontsize=8)


def plot_grid(samples_data: list, n_obs: int, out_path: str,
              n_samples_label: int) -> None:
    """Arrange all panels in a 2-column grid and save.

    Args:
        samples_data : list of dicts, each with keys
                       'idx', 'gt', 'knn', 'ratd_samples'.
        n_obs        : observation weeks (0 for cold-start).
        out_path     : destination file path (.png or .pdf).
        n_samples_label: number of diffusion samples drawn (for title).
    """
    n = len(samples_data)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flat if n > 1 else [axes]

    for i, data in enumerate(samples_data):
        ax = axes_flat[i]
        _plot_sample(ax, data["gt"], data["knn"], data["ratd_samples"],
                     sample_idx=data["idx"], n_obs=n_obs)

    # Hide any unused subplots
    for j in range(i + 1, nrows * ncols):
        axes_flat[j].set_visible(False)

    # Shared legend (from first axis)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    # Deduplicate (individual sample lines share the same label pattern)
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=4, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.03))

    setting = "Zero-Shot (n_obs=0)" if n_obs == 0 else f"{n_obs}-Shot (n_obs={n_obs})"
    fig.suptitle(
        f"Qualitative Comparison — Visuelle 2.0 SS19 — {setting}\n"
        f"Cold-RATD ({n_samples_label} samples)  vs  K-NN Mean  vs  Ground Truth",
        fontsize=11, y=1.01
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot qualitative Cold-RATD vs K-NN Mean vs Ground Truth"
    )
    p.add_argument("--modelfolder", type=str,
                   default="save/fashion_visuelle2_n0_20260414_111340",
                   help="Run folder containing config.json and model_best.pth")
    p.add_argument("--data_root",     type=str,
                   default="/home/shu_sho_bhit/BTP_2")
    p.add_argument("--processed_dir", type=str,
                   default="/home/shu_sho_bhit/BTP_2/visuelle2_processed")
    p.add_argument("--n_obs",    type=int, default=0,
                   help="Observation weeks (0 = pure cold-start)")
    p.add_argument("--n_plots",  type=int, default=10,
                   help="Number of test samples to visualise")
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of diffusion samples per product")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",  type=str,
                   default="results/qualitative_n0")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Reproducibility ------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Model ----------------------------------------------------------
    os.chdir(_REPO)
    model = _load_config_and_model(args.modelfolder, args.device)

    # ---- Dataset --------------------------------------------------------
    test_ds = Dataset_Visuelle2(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        flag="test",
        n_obs=args.n_obs,
    )
    print(f"Test set size: {len(test_ds)} samples")

    # Pick n_plots random indices
    all_indices = list(range(len(test_ds)))
    chosen = random.sample(all_indices, min(args.n_plots, len(all_indices)))
    chosen.sort()
    print(f"Selected test indices: {chosen}")

    # ---- Collect predictions --------------------------------------------
    samples_data = []

    for idx in chosen:
        item = test_ds[idx]

        # Ground truth: sales channel (channel 0), shape (12, 2) -> (12,)
        gt_np = item["observed_data"][:, 0].numpy()   # (12,)

        # K-NN Mean: average of 3 retrieved reference sales trajectories
        knn_pred = _knn_mean_pred(item["reference"])  # (12,)

        # Cold-RATD: run diffusion model
        ratd_samples = _run_diffusion(model, item, args.n_samples, args.device)
        # shape: (n_samples, 12)

        samples_data.append({
            "idx":          idx,
            "gt":           gt_np,
            "knn":          knn_pred,
            "ratd_samples": ratd_samples,
        })

        print(f"  [{idx:5d}] GT mean={gt_np.mean():.4f}  "
              f"KNN mean={knn_pred.mean():.4f}  "
              f"RATD median mean={np.median(ratd_samples, axis=0).mean():.4f}")

    # ---- Plot -----------------------------------------------------------
    setting_tag = f"n{args.n_obs}"
    out_path = os.path.join(args.out_dir, f"qualitative_{setting_tag}.png")

    plot_grid(
        samples_data=samples_data,
        n_obs=args.n_obs,
        out_path=out_path,
        n_samples_label=args.n_samples,
    )

    print("Done.")


if __name__ == "__main__":
    main()
