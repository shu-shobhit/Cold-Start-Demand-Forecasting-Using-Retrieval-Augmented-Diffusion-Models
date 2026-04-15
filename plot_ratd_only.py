"""plot_ratd_only.py

Plot Ground Truth vs Cold-RATD predictions for N randomly selected test products.

Each sample is saved as a separate figure showing:
  - Ground Truth   : actual 12-week [0,1] normalised sales trajectory
  - Cold-RATD      : diffusion median (solid) + individual draws (faint)

n_obs controls how many weeks are revealed as conditioning context (0–4).
For n_obs > 0, the observed region is shaded and those weeks show the GT
value in the RATD curve (the diffusion output for observed positions is
noise-contaminated and is replaced by GT for display).

Usage
-----
    python plot_ratd_only.py \\
        --modelfolder save/fashion_visuelle2_n0_20260414_111340 \\
        --n_obs 0 \\
        --n_samples 5 \\
        --n_plots 10 \\
        --seed 42 \\
        --out_dir results/ratd_only_n0
"""

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

from dataset_visuelle2 import Dataset_Visuelle2
from main_model_fashion import RATD_Fashion


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(modelfolder: str, device: str) -> RATD_Fashion:
    config_path = os.path.join(modelfolder, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config.pop("cli", None)

    model = RATD_Fashion(config, device).to(device)

    ckpt = os.path.join(modelfolder, "model_best.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(modelfolder, "model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded model from: {ckpt}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_diffusion(model: RATD_Fashion, batch_single: dict,
                   n_samples: int, device: str) -> np.ndarray:
    """Run Cold-RATD on one sample.

    Returns:
        np.ndarray of shape (n_samples, 12) in StandardScaler space.
    """
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
             for k, v in batch_single.items()}
    samples, *_ = model.evaluate(batch, n_samples=n_samples)
    return samples[0, :, 0, :].cpu().numpy()   # (n_samples, 12)


def _build_full_curve(gt: np.ndarray, pred_samples: np.ndarray,
                      n_obs: int) -> np.ndarray:
    """Splice GT observed weeks with model predicted weeks into (n_samples, 12).

    For n_obs=0 all 12 weeks come from pred_samples.
    For n_obs>0 weeks 0..n_obs-1 are filled with GT (the diffusion output
    for observed positions is discarded because impute() never clamps them).
    """
    n_samples = pred_samples.shape[0]
    full = np.empty((n_samples, 12), dtype=np.float32)
    if n_obs > 0:
        full[:, :n_obs] = gt[:n_obs]
    full[:, n_obs:] = pred_samples
    return full


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

WEEKS = np.arange(1, 13)

COLOR_GT         = "#2c7bb6"   # blue
COLOR_RATD       = "#1a9641"   # green
COLOR_RATD_SHADE = "#a8ddb5"   # light green


def _plot_sample(ax, gt: np.ndarray, ratd_full: np.ndarray,
                 sample_idx: int, n_obs: int) -> None:
    ratd_median = np.median(ratd_full, axis=0)   # (12,)

    for s in ratd_full:
        ax.plot(WEEKS, s, color=COLOR_RATD_SHADE, alpha=0.35,
                linewidth=0.8, zorder=1)

    ax.plot(WEEKS, ratd_median, color=COLOR_RATD, linewidth=1.8,
            label="Cold-RATD (median)", zorder=3)
    ax.plot(WEEKS, gt, color=COLOR_GT, linewidth=1.8,
            linestyle="-", label="Ground Truth", zorder=4)

    if n_obs > 0:
        ax.axvspan(0.5, n_obs + 0.5, color="#f0f0f0", zorder=0,
                   label="Observed")

    ax.set_title(f"Sample {sample_idx}", fontsize=9, pad=3)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Week", fontsize=8)
    ax.set_ylabel("Sales [0–1]", fontsize=8)


def _save_figure(data: dict, n_obs: int, out_path: str,
                 n_samples_label: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    _plot_sample(ax, data["gt"], data["ratd_full"],
                 sample_idx=data["idx"], n_obs=n_obs)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="best", fontsize=8, frameon=True)

    setting = "Zero-Shot (n_obs=0)" if n_obs == 0 else f"{n_obs}-Shot (n_obs={n_obs})"
    fig.suptitle(
        f"Cold-RATD vs Ground Truth — Visuelle 2.0 SS19 — {setting}\n"
        f"({n_samples_label} diffusion samples)",
        fontsize=10,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Cold-RATD vs Ground Truth (variable n_obs)"
    )
    p.add_argument("--modelfolder", type=str,
                   default="save/fashion_visuelle2_n0_20260414_111340")
    p.add_argument("--data_root",     type=str,
                   default="/home/shu_sho_bhit/BTP_2")
    p.add_argument("--processed_dir", type=str,
                   default="/home/shu_sho_bhit/BTP_2/visuelle2_processed")
    p.add_argument("--n_obs",    type=int, default=0,
                   help="Observed weeks (0 = pure cold-start, 1-4 = few-shot)")
    p.add_argument("--n_plots",  type=int, default=10)
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of diffusion samples per product")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",  type=str, default="results/ratd_only_n0")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.chdir(_REPO)

    model = _load_model(args.modelfolder, args.device)

    test_ds = Dataset_Visuelle2(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        flag="test",
        n_obs=args.n_obs,
    )
    print(f"Test set size: {len(test_ds)} samples")

    chosen = sorted(random.sample(range(len(test_ds)),
                                  min(args.n_plots, len(test_ds))))
    print(f"Selected test indices: {chosen}")

    setting_tag = f"n{args.n_obs}"
    os.makedirs(args.out_dir, exist_ok=True)

    for idx in chosen:
        item = test_ds[idx]

        # Ground truth in [0,1] space
        gt_scaled = item["observed_data"][:, 0].numpy()         # (12,) StandardScaler
        gt_np     = test_ds.inverse_transform_sales(gt_scaled)  # (12,) [0,1]

        # RATD predictions in StandardScaler space, then inverse-transform
        ratd_scaled = _run_diffusion(model, item, args.n_samples, args.device)
        # Only the held-out weeks are valid; observed weeks are noise-contaminated
        ratd_pred = test_ds.inverse_transform_sales(
            ratd_scaled[:, args.n_obs:]
        )                                                        # (n_samples, pred_len) [0,1]

        # Splice GT into observed weeks for a clean full-length curve
        ratd_full = _build_full_curve(gt_np, ratd_pred, args.n_obs)

        print(
            f"  [{idx:5d}] GT mean={gt_np.mean():.4f}  "
            f"RATD median mean={np.median(ratd_full, axis=0).mean():.4f}"
        )

        out_path = os.path.join(args.out_dir,
                                f"sample_{idx:05d}_{setting_tag}.png")
        _save_figure(
            data={"idx": idx, "gt": gt_np, "ratd_full": ratd_full},
            n_obs=args.n_obs,
            out_path=out_path,
            n_samples_label=args.n_samples,
        )

    print("Done.")


if __name__ == "__main__":
    main()
