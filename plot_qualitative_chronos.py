"""plot_qualitative_chronos.py

Plot qualitative forecasts for N randomly selected test products.

For each sample, three curves are drawn on the same panel:
  - Ground Truth  : the actual 12-week normalised sales trajectory
  - Chronos       : median (solid) + individual nsample draws (faint)
  - Cold-RATD     : median (solid) + individual nsample draws (faint)

Usage
-----
    python plot_qualitative_chronos.py \\
        --modelfolder save/fashion_visuelle2_n0_20260414_111340 \\
        --n_obs 1 \\
        --n_samples 5 \\
        --n_plots 10 \\
        --seed 42 \\
        --out_dir plots/qualitative_chronos_n1

All paths are relative to the repo root
(Cold-Start-Demand-Forecasting-Using-Retrieval-Augmented-Diffusion-Models/).

Notes
-----
- Chronos requires n_obs >= 1 (it needs at least one observed value as context).
- For the plot, observed weeks are shaded. Both Chronos and Cold-RATD show
  predicted values only for the held-out weeks (n_obs..11). Observed weeks
  are filled in from the ground truth for all three curves.
"""

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

from dataset_visuelle2 import Dataset_Visuelle2
from main_model_fashion import RATD_Fashion


# ---------------------------------------------------------------------------
# Helpers: model loading
# ---------------------------------------------------------------------------

def _load_config_and_model(modelfolder: str, device: str) -> RATD_Fashion:
    """Load config.json and model_best.pth from a run folder."""
    config_path = os.path.join(modelfolder, "config.json")
    with open(config_path) as f:
        config = json.load(f)

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


# ---------------------------------------------------------------------------
# Helpers: Chronos inference
# ---------------------------------------------------------------------------

def _load_chronos(model_id: str, device: str):
    """Load a ChronosPipeline; raises ImportError if not installed."""
    try:
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "chronos-forecasting is not installed.\n"
            "Install with:  pip install chronos-forecasting"
        )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline = ChronosPipeline.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype
    )
    print(f"Loaded Chronos pipeline: {model_id}")
    return pipeline


def _run_chronos(
    pipeline,
    obs_scaled: np.ndarray,
    n_obs: int,
    dataset: Dataset_Visuelle2,
    n_samples: int,
    scaler_val: float,
    mean_scaler: float,
) -> np.ndarray:
    """Run Chronos on a single product and return samples in StandardScaler space.

    Args:
        pipeline:    ChronosPipeline.
        obs_scaled:  (12,) float32 StandardScaler-normalised sales values.
        n_obs:       Number of observed weeks (>= 1).
        dataset:     Dataset instance (for inverse_transform_sales).
        n_samples:   Number of Chronos samples to draw.
        scaler_val:  StandardScaler scale_ for sales channel.
        mean_scaler: StandardScaler mean_ for sales channel.

    Returns:
        np.ndarray of shape (n_samples, pred_len) in StandardScaler space.
        pred_len = 12 - n_obs.
    """
    pred_len = 12 - n_obs

    # Inverse-transform context to non-negative [0,1] range
    ctx_scaled = obs_scaled[:n_obs]                      # (n_obs,)
    ctx_01 = dataset.inverse_transform_sales(ctx_scaled) # (n_obs,)
    ctx_01 = np.clip(ctx_01, 0.0, None)
    ctx_t  = torch.from_numpy(ctx_01).float().unsqueeze(0)  # (1, n_obs)

    with torch.inference_mode():
        # forecast shape: (1, n_samples, pred_len) in [0,1] range
        forecast_01 = pipeline.predict(
            ctx_t,
            prediction_length=pred_len,
            num_samples=n_samples,
            limit_prediction_length=False,
        )

    forecast_01 = forecast_01[0].cpu().float().numpy()   # (n_samples, pred_len)
    # Re-normalise to StandardScaler space: x_norm = (x_01 - mean) / std
    return (forecast_01 - mean_scaler) / scaler_val


# ---------------------------------------------------------------------------
# Helpers: Cold-RATD inference
# ---------------------------------------------------------------------------

def _run_diffusion(
    model: RATD_Fashion,
    batch_single: dict,
    n_samples: int,
    device: str,
) -> np.ndarray:
    """Run Cold-RATD on one sample and return all n_samples draws.

    Returns:
        np.ndarray of shape (n_samples, 12) in StandardScaler space.
    """
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
             for k, v in batch_single.items()}
    samples, *_ = model.evaluate(batch, n_samples=n_samples)
    # samples: (B=1, n_samples, K=2, L=12)
    return samples[0, :, 0, :].cpu().numpy()   # (n_samples, 12)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

WEEKS = np.arange(1, 13)

COLOR_GT           = "#2c7bb6"    # blue
COLOR_CHRONOS      = "#d7191c"    # red
COLOR_CHRONOS_SHADE = "#f4a582"   # light red
COLOR_RATD         = "#1a9641"    # green
COLOR_RATD_SHADE   = "#a8ddb5"    # light green


def _build_full_curve(gt: np.ndarray, pred_samples: np.ndarray, n_obs: int):
    """Splice observed GT weeks with predicted weeks into full 12-week curves.

    For weeks 0..n_obs-1 (observed), all samples equal the GT value.
    For weeks n_obs..11 (held-out), samples come from pred_samples.

    Args:
        gt:           (12,) ground-truth sales.
        pred_samples: (n_samples, pred_len) model predictions (held-out only).
        n_obs:        Number of observed weeks.

    Returns:
        np.ndarray of shape (n_samples, 12).
    """
    n_samples = pred_samples.shape[0]
    full = np.empty((n_samples, 12), dtype=np.float32)
    full[:, :n_obs]  = gt[:n_obs]          # observed region: replicate GT
    full[:, n_obs:]  = pred_samples         # held-out region: model samples
    return full


def _plot_sample(
    ax,
    gt: np.ndarray,
    chronos_full: np.ndarray,
    ratd_full: np.ndarray,
    sample_idx: int,
    n_obs: int,
) -> None:
    """Draw one panel with GT, Chronos, and Cold-RATD."""

    chronos_median = np.median(chronos_full, axis=0)   # (12,)
    ratd_median    = np.median(ratd_full,    axis=0)   # (12,)

    # Chronos median
    ax.plot(WEEKS, chronos_median, color=COLOR_CHRONOS, linewidth=1.6,
            linestyle="--", label="Chronos (median)", zorder=3)

    # Cold-RATD median
    ax.plot(WEEKS, ratd_median, color=COLOR_RATD, linewidth=1.8,
            label="Cold-RATD (median)", zorder=4)

    # Ground truth (on top)
    ax.plot(WEEKS, gt, color=COLOR_GT, linewidth=1.8,
            linestyle="-", label="Ground Truth", zorder=6)

    # Shade the observed region
    if n_obs > 0:
        ax.axvspan(0.5, n_obs + 0.5, color="#f0f0f0", zorder=0,
                   label="Observed")

    ax.set_title(f"Sample {sample_idx}", fontsize=9, pad=3)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Week", fontsize=8)
    ax.set_ylabel("Sales [0–1]", fontsize=8)


def save_single(
    data: dict,
    n_obs: int,
    out_path: str,
    n_samples_label: int,
) -> None:
    """Save one sample as its own figure."""
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

    _plot_sample(
        ax,
        gt=data["gt"],
        chronos_full=data["chronos_full"],
        ratd_full=data["ratd_full"],
        sample_idx=data["idx"],
        n_obs=n_obs,
    )

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="best", fontsize=8, frameon=True)

    setting = f"{n_obs}-Shot (n_obs={n_obs})"
    fig.suptitle(
        f"Qualitative Comparison — Visuelle 2.0 SS19 — {setting}\n"
        f"Cold-RATD ({n_samples_label} samples)  vs  Chronos  vs  Ground Truth",
        fontsize=10,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot qualitative Cold-RATD vs Chronos vs Ground Truth"
    )
    p.add_argument("--modelfolder", type=str,
                   default="save/fashion_visuelle2_n0_20260414_111340",
                   help="Run folder containing config.json and model_best.pth")
    p.add_argument("--data_root",     type=str,
                   default="/home/shu_sho_bhit/BTP_2")
    p.add_argument("--processed_dir", type=str,
                   default="/home/shu_sho_bhit/BTP_2/visuelle2_processed")
    p.add_argument("--chronos_model", type=str,
                   default="amazon/chronos-t5-mini",
                   help="HuggingFace model ID for Chronos variant")
    p.add_argument("--n_obs",    type=int, default=1,
                   help="Observed weeks (>= 1 required for Chronos context)")
    p.add_argument("--n_plots",  type=int, default=10,
                   help="Number of test samples to visualise")
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of samples per model per product")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",  type=str,
                   default="results/qualitative_chronos_n1")
    return p.parse_args()


def main():
    args = parse_args()

    if args.n_obs < 1:
        raise ValueError("--n_obs must be >= 1 (Chronos needs at least one "
                         "observed week as context)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.chdir(_REPO)

    # ---- Models ---------------------------------------------------------
    model     = _load_config_and_model(args.modelfolder, args.device)
    chronos   = _load_chronos(args.chronos_model, args.device)

    # ---- Datasets -------------------------------------------------------
    train_ds = Dataset_Visuelle2(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        flag="train",
        n_obs=0,
    )
    test_ds = Dataset_Visuelle2(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        flag="test",
        n_obs=args.n_obs,
    )
    print(f"Test set size: {len(test_ds)} samples")

    # Scaler params from training split (for Chronos re-normalisation)
    scaler_val  = float(train_ds.scaler.scale_[0])
    mean_scaler = float(train_ds.scaler.mean_[0])
    print(f"Scaler: scale={scaler_val:.6f}  mean={mean_scaler:.6f}")

    # ---- Sample selection -----------------------------------------------
    all_indices = list(range(len(test_ds)))
    chosen = random.sample(all_indices, min(args.n_plots, len(all_indices)))
    chosen.sort()
    print(f"Selected test indices: {chosen}")

    # ---- Collect predictions and save one figure per sample ------------
    setting_tag = f"n{args.n_obs}"
    os.makedirs(args.out_dir, exist_ok=True)

    for idx in chosen:
        item = test_ds[idx]

        # Ground truth: full 12-week sales (StandardScaler space -> [0,1])
        gt_scaled = item["observed_data"][:, 0].numpy()          # (12,) StandardScaler
        gt_np     = test_ds.inverse_transform_sales(gt_scaled)   # (12,) [0,1]

        # Chronos: context must be passed in StandardScaler space (the function
        # handles the inverse-transform internally). Returns StandardScaler space.
        chronos_preds_scaled = _run_chronos(
            chronos,
            obs_scaled=gt_scaled,
            n_obs=args.n_obs,
            dataset=test_ds,
            n_samples=args.n_samples,
            scaler_val=scaler_val,
            mean_scaler=mean_scaler,
        )                                                          # (n_samples, pred_len)
        chronos_preds = test_ds.inverse_transform_sales(chronos_preds_scaled)  # [0,1]

        # Cold-RATD: returns (n_samples, 12) in StandardScaler space.
        # Observed positions (weeks 0..n_obs-1) in the output are noise-
        # contaminated because impute() never clamps them back to ground truth.
        ratd_preds_scaled = _run_diffusion(
            model, item, args.n_samples, args.device
        )                                                          # (n_samples, 12)
        ratd_preds = test_ds.inverse_transform_sales(
            ratd_preds_scaled[:, args.n_obs:]
        )                                                          # (n_samples, pred_len) [0,1]

        # Splice GT (already [0,1]) into observed weeks for both models
        chronos_full = _build_full_curve(gt_np, chronos_preds, args.n_obs)
        ratd_full    = _build_full_curve(gt_np, ratd_preds,    args.n_obs)

        data = {
            "idx":          idx,
            "gt":           gt_np,
            "chronos_full": chronos_full,
            "ratd_full":    ratd_full,
        }

        print(
            f"  [{idx:5d}] GT mean={gt_np.mean():.4f} [0,1]  "
            f"Chronos median mean={np.median(chronos_full, axis=0).mean():.4f}  "
            f"RATD median mean={np.median(ratd_full, axis=0).mean():.4f}"
        )

        out_path = os.path.join(
            args.out_dir,
            f"sample_{idx:05d}_{setting_tag}.png",
        )
        save_single(
            data=data,
            n_obs=args.n_obs,
            out_path=out_path,
            n_samples_label=args.n_samples,
        )

    print("Done.")


if __name__ == "__main__":
    main()
