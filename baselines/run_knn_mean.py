"""baselines/run_knn_mean.py

Tier-0 non-parametric baselines for cold-start (n_obs=0) forecasting:

  Global Mean  -- predict the per-channel mean trajectory averaged across
                  all training products.  No product-specific information.

  K-NN Mean    -- for each test product, average the k=3 retrieved training
                  reference trajectories stored in test_references.pt.
                  Same FAISS neighbours used by RATD; zero extra computation.

Both baselines operate entirely without training.  They set the lower bar
that any learned model must beat.

Usage:
  python baselines/run_knn_mean.py
  python baselines/run_knn_mean.py --processed_dir /path/to/visuelle2_processed
  python baselines/run_knn_mean.py --output results/baselines/knn_mean_visuelle2.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Allow importing from the repo root (utils.py lives there)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dataset_visuelle2 import Dataset_Visuelle2
from dataset_hnm import Dataset_HnM
from utils import calc_quantile_CRPS, calc_quantile_CRPS_sum, calc_wape, console


# ---------------------------------------------------------------------------
# Dataset-agnostic helpers
# ---------------------------------------------------------------------------

def _get_sample_key(dataset, i: int):
    """Return the reference-dict lookup key for sample i.

    Visuelle 2.0 uses integer ``external_code`` values; H&M uses string
    ``article_id`` values.  Both are stored as the same key type in their
    respective references dicts.
    """
    if hasattr(dataset, "codes"):
        return int(dataset.codes[i])
    return dataset.article_ids[i]


def _load_datasets(args):
    """Instantiate train and test datasets based on ``args.dataset``."""
    if args.dataset == "visuelle2":
        train_ds = Dataset_Visuelle2(
            data_root=args.data_root,
            processed_dir=args.processed_dir,
            flag="train",
            n_obs=args.n_obs,
        )
        test_ds = Dataset_Visuelle2(
            data_root=args.data_root,
            processed_dir=args.processed_dir,
            flag="test",
            n_obs=args.n_obs,
        )
    else:
        train_ds = Dataset_HnM(
            processed_dir=args.processed_dir,
            flag="train",
            n_obs=args.n_obs,
        )
        test_ds = Dataset_HnM(
            processed_dir=args.processed_dir,
            flag="test",
            n_obs=args.n_obs,
        )
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_eval_arrays(test_dataset, n_obs: int) -> tuple:
    """Iterate the test set once and collect ground-truth arrays.

    Works for both Dataset_Visuelle2 (K=2) and Dataset_HnM (K=1).
    The K dimension is inferred from the first sample.

    Returns:
        targets_np   : (N, 12, K) float32 -- normalised sales [+discount]
        eval_mask_np : (N, 12, K) float32 -- 0 for observed weeks, 1 for held-out
    """
    N = len(test_dataset)
    K = test_dataset.K_CHANNELS

    targets_np   = np.empty((N, 12, K), dtype=np.float32)
    eval_mask_np = np.zeros((N, 12, K), dtype=np.float32)

    for i in range(N):
        sample = test_dataset[i]
        targets_np[i]   = sample["observed_data"].numpy()   # (12, K)
        gt_mask         = sample["gt_mask"].numpy()          # (12, K)
        eval_mask_np[i] = 1.0 - gt_mask

    return targets_np, eval_mask_np


def _metrics(
    targets_np:   np.ndarray,
    preds_np:     np.ndarray,
    eval_mask_np: np.ndarray,
    scaler_val:   float,
    mean_scaler:  float,
    tag:          str,
) -> dict:
    """Compute RMSE, MAE, WAPE (sales channel only) and CRPS for one baseline.

    CRPS is estimated by repeating the single point prediction 20 times
    (degenerate distribution: all mass at the median).  This will equal the
    MAE, but is included for table completeness.

    Args:
        targets_np:   (N, 12, 2) ground-truth, normalised.
        preds_np:     (N, 12, 2) point predictions, normalised.
        eval_mask_np: (N, 12, 2) 1 = evaluate, 0 = ignore.
        scaler_val:   StandardScaler scale_ (for RMSE/MAE de-normalisation).
        mean_scaler:  StandardScaler mean_ (for RMSE/MAE de-normalisation).
        tag:          Label for log output.

    Returns:
        dict with keys RMSE, MAE, WAPE, CRPS, CRPS_sum.
    """
    target_t  = torch.from_numpy(targets_np)       # (N, 12, 2)
    pred_t    = torch.from_numpy(preds_np)          # (N, 12, 2)
    ep_t      = torch.from_numpy(eval_mask_np)      # (N, 12, 2)

    # Point metrics in normalised space (matching RATD's reporting)
    diff  = (pred_t - target_t) * ep_t
    mse   = float(((diff * scaler_val) ** 2).sum() / ep_t.sum())
    mae   = float((torch.abs(diff) * scaler_val).sum() / ep_t.sum())
    rmse  = float(np.sqrt(mse))

    # WAPE on sales channel (channel 0) in de-normalised space
    wape  = calc_wape(
        target_t[..., :1],     # (N, 12, 1)
        pred_t[..., :1],
        ep_t[..., :1],
        scaler=scaler_val,
        mean_scaler=mean_scaler,
    )

    # CRPS: replicate the point prediction nsample=20 times to form a
    # degenerate distribution.  Equals MAE for a point forecast.
    nsample   = 20
    samples_t = pred_t.unsqueeze(1).expand(-1, nsample, -1, -1)   # (N, 20, 12, 2)
    # calc_quantile_CRPS expects (N, nsample, 12, 2) with inner dim being K:
    # RATD convention: target (N, L, K), forecast (N, nsample, L, K)
    crps     = calc_quantile_CRPS(
        target_t, samples_t, ep_t, mean_scaler, scaler_val
    )
    crps_sum = calc_quantile_CRPS_sum(
        target_t, samples_t, ep_t, mean_scaler, scaler_val
    )

    metrics = {
        "RMSE":     rmse,
        "MAE":      mae,
        "WAPE":     wape,
        "CRPS":     crps,
        "CRPS_sum": crps_sum,
    }
    console.log(
        f"[bold cyan]{tag}[/]  "
        f"RMSE={rmse:.4f}  MAE={mae:.4f}  WAPE={wape*100:.1f}%  "
        f"CRPS={crps:.4f}  CRPS_sum={crps_sum:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Global Mean baseline
# ---------------------------------------------------------------------------

def run_global_mean(
    train_dataset: Dataset_Visuelle2,
    test_dataset:  Dataset_Visuelle2,
    n_obs:         int,
    scaler_val:    float,
    mean_scaler:   float,
) -> dict:
    """Predict the per-channel mean trajectory from all training samples.

    The mean is computed across all (store, product) training rows in the
    normalised space, then broadcast to every test sample.

    Args:
        train_dataset: Fitted Dataset_Visuelle2 for the training split.
        test_dataset:  Dataset_Visuelle2 for the test split.
        n_obs:         Number of observation weeks (controls eval mask).
        scaler_val:    StandardScaler scale_ value.
        mean_scaler:   StandardScaler mean_ value.

    Returns:
        Metric dict (RMSE, MAE, WAPE, CRPS, CRPS_sum).
    """
    console.log("Running [bold]Global Mean[/] baseline ...")

    # Mean over all training rows: (N_train, 12, 2) -> (12, 2)
    train_np  = train_dataset.data                          # (N_train, 12, 2)
    mean_traj = train_np.mean(axis=0)                       # (12, 2)

    targets_np, eval_mask_np = _build_eval_arrays(test_dataset, n_obs)
    N = len(targets_np)

    # Broadcast mean to all test samples
    preds_np = np.broadcast_to(mean_traj, (N, 12, 2)).copy()

    return _metrics(targets_np, preds_np, eval_mask_np,
                    scaler_val, mean_scaler, f"Global Mean (n_obs={n_obs})")


# ---------------------------------------------------------------------------
# K-NN Mean baseline
# ---------------------------------------------------------------------------

def run_knn_mean(
    test_dataset: Dataset_Visuelle2,
    n_obs:        int,
    scaler_val:   float,
    mean_scaler:  float,
) -> dict:
    """Average the k=3 pre-retrieved reference trajectories per test product.

    Reference tensors are already stored in test_references.pt (same file used
    by RATD).  Each reference is the mean trajectory of one retrieved neighbour
    averaged across all its stores.  Shape per product: (2, 36) = (K, k*L).

    The prediction for each test sample is the mean of the k=3 reference
    trajectories (each of length 12), averaged channel-wise.

    Args:
        test_dataset: Dataset_Visuelle2 for the test split.
        n_obs:        Number of observation weeks (controls eval mask).
        scaler_val:   StandardScaler scale_ value.
        mean_scaler:  StandardScaler mean_ value.

    Returns:
        Metric dict (RMSE, MAE, WAPE, CRPS, CRPS_sum).
    """
    console.log("Running [bold]K-NN Mean[/] baseline ...")

    N          = len(test_dataset)
    targets_np   = np.empty((N, 12, 2), dtype=np.float32)
    eval_mask_np = np.zeros((N, 12, 2), dtype=np.float32)
    preds_np     = np.empty((N, 12, 2), dtype=np.float32)

    references = test_dataset.references      # dict: code -> Tensor (2, 36)

    for i in range(N):
        sample        = test_dataset[i]
        code          = int(test_dataset.codes[i])
        targets_np[i] = sample["observed_data"].numpy()
        gt_mask       = sample["gt_mask"].numpy()
        eval_mask_np[i] = 1.0 - gt_mask

        # ref_kl shape: (2, 36) = (K_channels, k * pred_len)
        ref_kl = references[code].numpy()           # (2, 36)
        # Split into 3 reference trajectories of length 12 each, then average
        # Shape after split: 3 x (2, 12)
        refs = np.split(ref_kl, 3, axis=1)          # list of 3 arrays (2,12)
        knn_mean = np.mean(refs, axis=0)             # (2, 12)
        preds_np[i] = knn_mean.T                     # (12, 2)

    return _metrics(targets_np, preds_np, eval_mask_np,
                    scaler_val, mean_scaler, f"K-NN Mean (n_obs={n_obs})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Tier-0 non-parametric baselines: Global Mean and K-NN Mean"
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/home/shu_sho_bhit/BTP_2",
        help="Path to BTP_2/ directory",
    )
    p.add_argument(
        "--processed_dir",
        type=str,
        default="/home/shu_sho_bhit/BTP_2/visuelle2_processed",
        help="Path to visuelle2_processed/ directory",
    )
    p.add_argument(
        "--n_obs",
        type=int,
        default=0,
        help="Observation weeks for eval mask (0 = pure cold-start)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save JSON results (e.g. results/baselines/knn_mean_visuelle2.json)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    console.log(f"Loading Visuelle 2.0 datasets from [bold]{args.data_root}[/] ...")
    train_dataset = Dataset_Visuelle2(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        flag="train",
        n_obs=args.n_obs,
    )
    test_dataset = Dataset_Visuelle2(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        flag="test",
        n_obs=args.n_obs,
    )

    # Scaler params for metric de-normalisation
    if train_dataset.scale and train_dataset.scaler is not None:
        scaler_val  = float(train_dataset.scaler.scale_[0])   # sales channel
        mean_scaler = float(train_dataset.scaler.mean_[0])
    else:
        scaler_val, mean_scaler = 1.0, 0.0

    console.log(f"Scaler: scale={scaler_val:.6f}  mean={mean_scaler:.6f}")

    results = {}

    results["global_mean"] = run_global_mean(
        train_dataset, test_dataset,
        n_obs=args.n_obs,
        scaler_val=scaler_val,
        mean_scaler=mean_scaler,
    )

    results["knn_mean"] = run_knn_mean(
        test_dataset,
        n_obs=args.n_obs,
        scaler_val=scaler_val,
        mean_scaler=mean_scaler,
    )

    # Summary
    console.print("\n[bold cyan]Tier-0 Baseline Summary[/]")
    console.print(f"{'Model':<14} {'RMSE':>8} {'MAE':>8} {'WAPE%':>7} {'CRPS':>8} {'CRPS_sum':>10}")
    console.print("-" * 58)
    for name, m in results.items():
        console.print(
            f"{name:<14} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
            f"{m['WAPE']*100:>6.1f}% {m['CRPS']:>8.4f} {m['CRPS_sum']:>10.4f}"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        console.log(f"Results saved to [bold]{args.output}[/]")


if __name__ == "__main__":
    main()
