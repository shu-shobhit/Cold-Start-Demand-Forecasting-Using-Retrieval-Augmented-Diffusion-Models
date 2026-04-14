"""baselines/run_chronos.py

Tier-2 foundation model baseline for few-shot (n_obs >= 1) forecasting:

  Chronos-Mini  -- amazon/chronos-t5-mini (21M parameters), used zero-shot
                   with no fine-tuning on Visuelle 2.0 or H&M.

Key design choices:
  - Chronos is strictly univariate: only the sales channel (index 0) is
    passed as context.  The discount channel (Visuelle2) is ignored entirely.
  - Context values are inverse-transformed from StandardScaler space back
    to the original non-negative range before being handed to Chronos.
    Chronos does its own internal mean-scaling, which behaves better on
    non-negative inputs than on the zero-centred StandardScaler output.
    For Visuelle2 this gives the [0, 1] normalised range; for H&M it gives
    log1p-scaled sales (also >= 0).
  - Chronos predictions (returned in the same non-negative space as the
    input) are re-normalised to StandardScaler space so that RMSE / MAE
    are computed on the same scale as every other baseline and RATD_Fashion.
  - WAPE and CRPS are fully de-normalised (standard across all baselines).
  - Because Chronos produces a genuine sample distribution (nsample=50),
    CRPS is computed from the full sample set rather than a degenerate
    point-forecast repetition.  This makes Chronos's CRPS directly
    comparable to RATD_Fashion's.
  - All metrics are reported on the SALES channel only.  CRPS_sum is also
    over the sales channel only (K=1) because no discount predictions exist.
    This distinction is flagged in the JSON output under "note".

Usage:
  # Sweep all n_obs in {1, 2, 3, 4} on Visuelle 2.0 (default)
  python baselines/run_chronos.py

  # Single n_obs
  python baselines/run_chronos.py --mode single --n_obs 2

  # H&M dataset
  python baselines/run_chronos.py --dataset hnm \
      --hnm_processed_dir /path/to/hnm_processed

  # Save results
  python baselines/run_chronos.py --output results/baselines/chronos_visuelle2.json

  # Override processed data directory
  python baselines/run_chronos.py --processed_dir /path/to/visuelle2_processed

Dependencies:
  pip install chronos-forecasting
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dataset_visuelle2 import Dataset_Visuelle2
from dataset_hnm import Dataset_HnM
from utils import calc_quantile_CRPS, calc_quantile_CRPS_sum, calc_wape, console


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def _load_pipeline(model_id: str, device: str):
    """Load a ChronosPipeline from a HuggingFace model ID.

    Args:
        model_id: HuggingFace model identifier, e.g. ``"amazon/chronos-t5-mini"``.
        device:   PyTorch device string (``"cuda"`` or ``"cpu"``).

    Returns:
        ChronosPipeline: Ready-to-use pipeline for probabilistic forecasting.

    Raises:
        ImportError: If the ``chronos-forecasting`` package is not installed.
    """
    try:
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "chronos-forecasting is not installed.\n"
            "Install it with:  pip install chronos-forecasting"
        )

    # bfloat16 is fast on modern GPUs; fall back to float32 on CPU.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    console.log(
        f"Loading Chronos pipeline [bold]{model_id}[/] "
        f"(dtype={dtype}, device={device}) ..."
    )
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Dataset-agnostic helpers
# ---------------------------------------------------------------------------

def _inverse_transform_ctx(dataset, ctx_scaled: np.ndarray) -> np.ndarray:
    """Undo StandardScaler on the sales channel to get non-negative values.

    Chronos requires non-negative context values for its internal mean-scaling
    normalisation to behave well.  This function inverts only the StandardScaler
    step, leaving the result in the original non-negative space:

      - Dataset_Visuelle2: returns values in the [0, 1] normalised sales range.
      - Dataset_HnM: returns log1p-scaled sales values (always >= 0).

    Args:
        dataset:    Loaded Dataset_Visuelle2 or Dataset_HnM instance.
        ctx_scaled: 1-D ndarray of StandardScaler-normalised sales values.

    Returns:
        1-D ndarray in the non-negative original space, same shape as input.
    """
    if hasattr(dataset, "inverse_transform_sales"):
        # Dataset_Visuelle2 path: uses 2-channel scaler, handles dummy discount column
        return dataset.inverse_transform_sales(ctx_scaled)
    # Dataset_HnM path: scaler was fit on 1-D log1p(sales); undo it directly
    flat = ctx_scaled.reshape(-1, 1)
    return dataset.scaler.inverse_transform(flat).reshape(ctx_scaled.shape)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_contexts_and_targets(
    test_dataset,
    n_obs: int,
) -> tuple:
    """Collect per-sample data needed for Chronos inference and metric computation.

    The sales context is inverse-transformed from StandardScaler space to the
    original non-negative range so that Chronos receives values compatible with
    its internal mean-scaling normalisation.  Works for both Dataset_Visuelle2
    (K=2, returns [0,1] range) and Dataset_HnM (K=1, returns log1p space).

    Args:
        test_dataset: Loaded ``Dataset_Visuelle2`` or ``Dataset_HnM`` test split.
        n_obs:        Number of observed weeks (1-4).

    Returns:
        contexts_01:  list of N ``torch.Tensor`` objects, each shape ``(n_obs,)``,
                      sales values in the non-negative original space.
        targets_np:   ``(N, 12, K)`` float32 ndarray, StandardScaler-normalised.
        eval_mask_np: ``(N, 12, K)`` float32 ndarray, 1 = held-out week.
    """
    N = len(test_dataset)
    K = test_dataset.K_CHANNELS          # 2 for Visuelle2, 1 for H&M

    targets_np   = np.empty((N, 12, K), dtype=np.float32)
    eval_mask_np = np.zeros((N, 12, K), dtype=np.float32)
    contexts_01  = []

    for i in range(N):
        sample = test_dataset[i]
        obs    = sample["observed_data"].numpy()   # (12, K) StandardScaler space
        gt     = sample["gt_mask"].numpy()         # (12, K)

        targets_np[i]   = obs
        eval_mask_np[i] = 1.0 - gt

        # Sales context in StandardScaler space: first n_obs weeks, channel 0
        ctx_scaled = obs[:n_obs, 0]                # (n_obs,) float32 ndarray

        # Inverse-transform to original non-negative range for Chronos.
        ctx_01 = _inverse_transform_ctx(test_dataset, ctx_scaled)  # (n_obs,)
        ctx_01 = np.clip(ctx_01, 0.0, None)        # clip negatives to 0 (numerical noise)

        contexts_01.append(torch.from_numpy(ctx_01).float())

    return contexts_01, targets_np, eval_mask_np


# ---------------------------------------------------------------------------
# Batched Chronos inference
# ---------------------------------------------------------------------------

def _run_chronos_predict(
    pipeline,
    contexts_01:  list,
    pred_len:     int,
    nsample:      int,
    batch_size:   int,
    scaler_val:   float,
    mean_scaler:  float,
) -> np.ndarray:
    """Run batched Chronos inference and return samples in StandardScaler space.

    Chronos receives values in the original [0, 1] range and returns predictions
    in the same scale.  Those predictions are then re-normalised to
    StandardScaler space so metric computation is consistent with all other
    baselines: ``x_norm = (x_01 - mean_scaler) / scaler_val``.

    Args:
        pipeline:    Loaded ChronosPipeline.
        contexts_01: List of N ``Tensor(n_obs,)`` in original [0, 1] range.
        pred_len:    Number of future weeks to forecast (12 - n_obs).
        nsample:     Number of forecast samples to draw per test sample.
        batch_size:  Number of test samples processed in one pipeline call.
        scaler_val:  StandardScaler ``scale_[0]`` (sales channel std).
        mean_scaler: StandardScaler ``mean_[0]`` (sales channel mean).

    Returns:
        np.ndarray: ``(N, nsample, pred_len)`` float32 in StandardScaler space.
    """
    N        = len(contexts_01)
    n_batches = (N + batch_size - 1) // batch_size
    all_samples_norm = np.empty((N, nsample, pred_len), dtype=np.float32)

    with torch.inference_mode():
        for b in range(n_batches):
            start = b * batch_size
            end   = min(start + batch_size, N)

            # Stack into (B, n_obs) -- all contexts share the same length
            batch_ctx = torch.stack(contexts_01[start:end], dim=0)  # (B, n_obs)

            # forecast shape: (B, nsample, pred_len) -- values in [0,1] range
            forecast_01 = pipeline.predict(
                context=batch_ctx,
                prediction_length=pred_len,
                num_samples=nsample,
                limit_prediction_length=False,
            )
            forecast_01 = forecast_01.cpu().float().numpy()  # (B, nsample, pred_len)

            # Re-normalise to StandardScaler space for consistent metric computation
            # x_norm = (x_01 - mean_scaler) / scaler_val
            forecast_norm = (forecast_01 - mean_scaler) / scaler_val

            all_samples_norm[start:end] = forecast_norm

            if (b + 1) % 5 == 0 or end == N:
                console.log(
                    f"  Inference batch {b + 1}/{n_batches}  "
                    f"({end}/{N} test samples done)"
                )

    return all_samples_norm


# ---------------------------------------------------------------------------
# Metric computation (sales channel only)
# ---------------------------------------------------------------------------

def _compute_metrics(
    all_samples_norm: np.ndarray,
    targets_np:       np.ndarray,
    eval_mask_np:     np.ndarray,
    n_obs:            int,
    scaler_val:       float,
    mean_scaler:      float,
    tag:              str,
) -> dict:
    """Compute RMSE, MAE, WAPE, CRPS, CRPS_sum for Chronos predictions.

    All metrics are computed on the **sales channel (index 0) only**.
    CRPS uses the full nsample distribution produced by Chronos (not a
    degenerate point-forecast repetition).

    Tensor layout notes:
      - ``all_samples_norm`` covers only the pred_len held-out weeks.
        It is embedded into a full-length (12,) array with zeros at the
        observed positions; those positions are masked to zero by eval_mask.
      - eval_mask for the discount channel (index 1) is forced to zero so
        that the absent discount predictions do not inflate error metrics.

    Args:
        all_samples_norm: ``(N, nsample, pred_len)`` StandardScaler-normalised
                          sales predictions (weeks n_obs to 11).
        targets_np:       ``(N, 12, 2)`` ground-truth, StandardScaler-normalised.
        eval_mask_np:     ``(N, 12, 2)`` 1 = held-out, 0 = observed.
        n_obs:            Number of observed weeks.
        scaler_val:       StandardScaler ``scale_[0]``.
        mean_scaler:      StandardScaler ``mean_[0]``.
        tag:              Label string for console output.

    Returns:
        dict with keys: RMSE, MAE, WAPE, CRPS, CRPS_sum, note.
    """
    N, nsample, pred_len = all_samples_norm.shape

    # ------------------------------------------------------------------
    # Build full-length (12-week) sample and eval_mask tensors.
    # Observed weeks (0..n_obs-1) are left as zero; eval_mask is 0 there
    # so they never contribute to any metric.
    # ------------------------------------------------------------------
    samples_full = np.zeros((N, nsample, 12), dtype=np.float32)
    samples_full[:, :, n_obs:] = all_samples_norm        # (N, nsample, 12)

    # Median across samples for point metrics (RMSE, MAE)
    median_full  = np.median(samples_full, axis=1)       # (N, 12)

    # Sales-channel tensors
    target_sales = torch.from_numpy(targets_np[:, :, 0])    # (N, 12)
    pred_median  = torch.from_numpy(median_full)            # (N, 12)
    eval_sales   = torch.from_numpy(eval_mask_np[:, :, 0])  # (N, 12)

    # ------------------------------------------------------------------
    # RMSE and MAE  (StandardScaler-normalised space, sales channel)
    # ------------------------------------------------------------------
    diff = (pred_median - target_sales) * eval_sales
    mse  = float(((diff * scaler_val) ** 2).sum() / eval_sales.sum())
    mae  = float((torch.abs(diff) * scaler_val).sum() / eval_sales.sum())
    rmse = float(np.sqrt(mse))

    # ------------------------------------------------------------------
    # WAPE  (de-normalised, sales channel)
    # ------------------------------------------------------------------
    wape = calc_wape(
        target_sales.unsqueeze(-1),    # (N, 12, 1)
        pred_median.unsqueeze(-1),     # (N, 12, 1)
        eval_sales.unsqueeze(-1),      # (N, 12, 1)
        scaler=scaler_val,
        mean_scaler=mean_scaler,
    )

    # ------------------------------------------------------------------
    # CRPS and CRPS_sum  (de-normalised, sales channel, genuine samples)
    #
    # K=1 because Chronos is univariate.  CRPS_sum with K=1 equals CRPS
    # (sum of one channel is the channel itself), but is included so the
    # output dict has the same keys as all other baselines.
    # ------------------------------------------------------------------
    samples_t = torch.from_numpy(samples_full).unsqueeze(-1)  # (N, nsample, 12, 1)
    target_t  = target_sales.unsqueeze(-1)                    # (N, 12, 1)
    ev_t      = eval_sales.unsqueeze(-1)                      # (N, 12, 1)

    crps     = calc_quantile_CRPS(target_t, samples_t, ev_t, mean_scaler, scaler_val)
    crps_sum = calc_quantile_CRPS_sum(target_t, samples_t, ev_t, mean_scaler, scaler_val)

    metrics = {
        "RMSE":     rmse,
        "MAE":      mae,
        "WAPE":     wape,
        "CRPS":     crps,
        "CRPS_sum": crps_sum,
        "note":     "all metrics computed on sales channel only (Chronos is univariate)",
    }
    console.log(
        f"[bold cyan]{tag}[/]  "
        f"RMSE={rmse:.4f}  MAE={mae:.4f}  WAPE={wape * 100:.1f}%  "
        f"CRPS={crps:.4f}  CRPS_sum={crps_sum:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Per-n_obs runner
# ---------------------------------------------------------------------------

def run_chronos_n_obs(
    pipeline,
    test_dataset,
    n_obs:        int,
    scaler_val:   float,
    mean_scaler:  float,
    nsample:      int = 50,
    batch_size:   int = 256,
) -> dict:
    """Run Chronos-mini zero-shot inference for one fixed n_obs setting.

    Works with both Dataset_Visuelle2 and Dataset_HnM.  Metrics are always
    reported on the sales channel (index 0) only.

    Args:
        pipeline:     Loaded ChronosPipeline.
        test_dataset: ``Dataset_Visuelle2`` or ``Dataset_HnM`` test split
                      (``n_obs`` attribute is mutated per run).
        n_obs:        Number of observed sales weeks to use as context (1-4).
        scaler_val:   StandardScaler ``scale_[0]`` from the training split.
        mean_scaler:  StandardScaler ``mean_[0]`` from the training split.
        nsample:      Forecast samples to draw per test series.
        batch_size:   Test samples per Chronos call.

    Returns:
        dict with keys: RMSE, MAE, WAPE, CRPS, CRPS_sum, note.

    Raises:
        AssertionError: If n_obs < 1.
    """
    assert n_obs >= 1, (
        f"Chronos requires at least 1 observed week for context "
        f"(got n_obs={n_obs}). Use the Tier-0 baselines for n_obs=0."
    )

    pred_len = 12 - n_obs
    console.log(
        f"\nRunning [bold]Chronos-Mini[/] zero-shot  "
        f"n_obs={n_obs}  pred_len={pred_len}  nsample={nsample} ..."
    )

    # Mutate dataset so __getitem__ returns the correct gt_mask
    test_dataset.n_obs = n_obs

    contexts_01, targets_np, eval_mask_np = _collect_contexts_and_targets(
        test_dataset, n_obs
    )

    all_samples_norm = _run_chronos_predict(
        pipeline, contexts_01, pred_len, nsample, batch_size,
        scaler_val, mean_scaler,
    )

    tag = f"Chronos-Mini (n_obs={n_obs})"
    return _compute_metrics(
        all_samples_norm, targets_np, eval_mask_np,
        n_obs, scaler_val, mean_scaler, tag,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Tier-2 Chronos-Mini zero-shot baseline (Visuelle 2.0 and H&M)"
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["visuelle2", "hnm"],
        default="visuelle2",
        help="Which dataset to evaluate on: 'visuelle2' (default) or 'hnm'.",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/home/shu_sho_bhit/BTP_2",
        help="Path to the BTP_2/ directory containing the raw visuelle2/ folder. "
             "Ignored when --dataset hnm.",
    )
    p.add_argument(
        "--processed_dir",
        type=str,
        default="/home/shu_sho_bhit/BTP_2/visuelle2_processed",
        help="Directory with precomputed Visuelle 2.0 .pt files. "
             "Ignored when --dataset hnm.",
    )
    p.add_argument(
        "--hnm_processed_dir",
        type=str,
        default="/home/shu_sho_bhit/BTP_2/hnm_processed",
        help="Directory with precomputed H&M .pt files. "
             "Used only when --dataset hnm.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device string ('cuda' or 'cpu').",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["single", "sweep"],
        default="sweep",
        help=(
            "single: run for the --n_obs value only.  "
            "sweep: run all n_obs in {1, 2, 3, 4} and print a comparison table."
        ),
    )
    p.add_argument(
        "--n_obs",
        type=int,
        default=2,
        help="Observation weeks to use as context (1–4).  Ignored in sweep mode.",
    )
    p.add_argument(
        "--nsample",
        type=int,
        default=50,
        help="Number of forecast samples drawn per series.  "
             "Higher values give more stable CRPS estimates.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of test samples processed in one pipeline.predict() call.",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default="amazon/chronos-t5-mini",
        help="HuggingFace model ID for the Chronos variant to use.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save JSON results (e.g. results/baselines/chronos_visuelle2.json).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Load datasets -------------------------------------------------------
    if args.dataset == "visuelle2":
        console.log(
            f"Loading Visuelle 2.0 datasets from [bold]{args.data_root}[/] ..."
        )
        # Train dataset is needed only for scaler parameters; n_obs=0 is fine.
        train_dataset = Dataset_Visuelle2(
            data_root=args.data_root,
            processed_dir=args.processed_dir,
            flag="train",
            n_obs=0,
        )
        # Test dataset is shared across all n_obs runs; n_obs is mutated per run.
        test_dataset = Dataset_Visuelle2(
            data_root=args.data_root,
            processed_dir=args.processed_dir,
            flag="test",
            n_obs=args.n_obs,
        )
    else:
        console.log(
            f"Loading H&M datasets from [bold]{args.hnm_processed_dir}[/] ..."
        )
        train_dataset = Dataset_HnM(
            processed_dir=args.hnm_processed_dir,
            flag="train",
            n_obs=0,
        )
        test_dataset = Dataset_HnM(
            processed_dir=args.hnm_processed_dir,
            flag="test",
            n_obs=args.n_obs,
        )

    if train_dataset.scale and train_dataset.scaler is not None:
        scaler_val  = float(train_dataset.scaler.scale_[0])
        mean_scaler = float(train_dataset.scaler.mean_[0])
    else:
        scaler_val, mean_scaler = 1.0, 0.0
    console.log(f"Scaler: scale={scaler_val:.6f}  mean={mean_scaler:.6f}")

    # ---- Load Chronos pipeline -----------------------------------------------
    pipeline = _load_pipeline(args.model_id, args.device)

    # ---- Run inference -------------------------------------------------------
    results      = {}
    n_obs_list   = list(range(1, 5)) if args.mode == "sweep" else [args.n_obs]

    for n_obs in n_obs_list:
        results[n_obs] = run_chronos_n_obs(
            pipeline,
            test_dataset,
            n_obs,
            scaler_val,
            mean_scaler,
            nsample=args.nsample,
            batch_size=args.batch_size,
        )

    # ---- Summary table -------------------------------------------------------
    console.print("\n[bold cyan]Chronos-Mini Baseline Summary[/]")
    console.print(
        f"  dataset={args.dataset}  model={args.model_id}  nsample={args.nsample}"
    )
    console.print("  Note: all metrics on SALES channel only")
    console.print(
        f"{'n_obs':<8} {'RMSE':>8} {'MAE':>8} {'WAPE%':>7} "
        f"{'CRPS':>8} {'CRPS_sum':>10}"
    )
    console.print("-" * 58)
    for n_obs, m in results.items():
        label = f"n_obs={n_obs}"
        console.print(
            f"{label:<8} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
            f"{m['WAPE'] * 100:>6.1f}% {m['CRPS']:>8.4f} "
            f"{m['CRPS_sum']:>10.4f}"
        )

    # ---- Save JSON -----------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # Use string keys for JSON serialisation
        json_results = {str(k): v for k, v in results.items()}
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        console.log(f"Results saved to [bold]{args.output}[/]")


if __name__ == "__main__":
    main()
