"""exe_hnm.py

Training and evaluation entrypoint for the H&M cold-start forecasting model.

Mirrors exe_fashion.py but targets the H&M dataset:
  - Uses Dataset_HnM (K=1, article-level) and RATD_HnM
  - Has a dedicated val split for best-model selection (no test leakage)
  - Sweep evaluates n_obs in {0, 1, 2, 3, 4}
  - Scaler uses log1p + StandardScaler; WAPE is reported in raw unit space

Typical usage:

  # Train from scratch
  python exe_hnm.py --device cuda --seed 42

  # Resume interrupted run
  python exe_hnm.py --resume save/hnm_n2_20260413_120000

  # Evaluate only (sweep all n_obs)
  python exe_hnm.py --modelfolder save/hnm_n2_20260413_120000 --sweep --nsample 50
"""

import argparse
import datetime
import glob
import json
import os
import pickle

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset_hnm import Dataset_HnM, get_dataloader
from main_model_hnm import RATD_HnM
from utils import (
    calc_quantile_CRPS,
    calc_quantile_CRPS_sum,
    calc_wape,
    console,
    train,
)

# ---------------------------------------------------------------------------
# Repository root (one level up from this file)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train / evaluate RATD_HnM on H&M dataset")

    p.add_argument("--config",     type=str,
                   default=os.path.join(_REPO_ROOT, "config", "hnm.yaml"))
    p.add_argument("--processed_dir", type=str,
                   default=os.path.join(os.path.dirname(_REPO_ROOT), "hnm_processed"),
                   help="Path to hnm_processed/ directory")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int, default=1)

    # Observation settings
    p.add_argument("--n_obs_eval", type=int, default=2,
                   help="Fixed n_obs for single evaluation (ignored with --sweep)")
    p.add_argument("--n_obs_min",  type=int, default=0,
                   help="Minimum n_obs sampled during training")
    p.add_argument("--n_obs_max",  type=int, default=4,
                   help="Maximum n_obs sampled during training")

    # Evaluation
    p.add_argument("--nsample",    type=int, default=50,
                   help="Number of reverse-diffusion samples per batch")
    p.add_argument("--sweep",      action="store_true",
                   help="Evaluate all n_obs in {0,1,2,3,4} after training")
    p.add_argument("--modelfolder", type=str, default="",
                   help="Skip training; load model from this folder and evaluate")

    # Training checkpoints
    p.add_argument("--save_every",  type=int, default=10,
                   help="Save full checkpoint every N epochs (0 = only final+best)")
    p.add_argument("--val_interval", type=int, default=10,
                   help="Validate every N epochs for best-model selection")
    p.add_argument("--resume",      type=str, default="",
                   help="Resume training from checkpoint file or folder")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_resume_checkpoint(path: str) -> str:
    """Return the path to a resumable checkpoint given a file or folder."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        latest = os.path.join(path, "checkpoint_latest.pth")
        if os.path.exists(latest):
            return latest
        candidates = glob.glob(os.path.join(path, "checkpoint_epoch*.pth"))
        if candidates:
            def _epoch_num(p):
                try:
                    return int(
                        os.path.basename(p)
                        .replace("checkpoint_epoch", "")
                        .replace(".pth", "")
                    )
                except ValueError:
                    return -1
            return max(candidates, key=_epoch_num)
    raise FileNotFoundError(f"No resumable checkpoint found at: {path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_hnm(
    model,
    test_loader,
    dataset,
    nsample:      int   = 50,
    scaler_val:   float = 1.0,
    mean_scaler:  float = 0.0,
    foldername:   str   = "",
    tag:          str   = "",
) -> dict:
    """Evaluate RATD_HnM on a test/val split and return metric dict.

    Metrics in normalised (log1p-scaled) space: RMSE, MAE, CRPS, CRPS_sum.
    WAPE is computed in raw sales space via the dataset's inverse_transform.

    Args:
        model:       Trained RATD_HnM.
        test_loader: DataLoader for the evaluation split.
        dataset:     The underlying Dataset_HnM (for inverse_transform).
        nsample:     Number of reverse-diffusion samples per batch.
        scaler_val:  StandardScaler scale_ (used for RMSE/MAE de-norm).
        mean_scaler: StandardScaler mean_ (used for RMSE/MAE de-norm).
        foldername:  If non-empty, save pickles here.
        tag:         Suffix added to output filenames (e.g. "n2").

    Returns:
        dict with keys RMSE, MAE, WAPE, CRPS, CRPS_sum.
    """
    model.eval()

    all_target            = []
    all_evalpoint         = []
    all_observed_point    = []
    all_observed_time     = []
    all_generated_samples = []

    mse_total        = 0.0
    mae_total        = 0.0
    evalpoints_total = 0.0

    with torch.no_grad():
        for test_batch in test_loader:
            output = model.evaluate(test_batch, nsample)
            samples, c_target, eval_points, obs_points, obs_time = output

            samples     = samples.permute(0, 1, 3, 2)   # (B, nsample, L, K)
            c_target    = c_target.permute(0, 2, 1)     # (B, L, K)
            eval_points = eval_points.permute(0, 2, 1)
            obs_points  = obs_points.permute(0, 2, 1)

            samples_median = samples.median(dim=1).values

            all_target.append(c_target)
            all_evalpoint.append(eval_points)
            all_observed_point.append(obs_points)
            all_observed_time.append(obs_time)
            all_generated_samples.append(samples)

            mse_current = (
                ((samples_median - c_target) * eval_points) ** 2
            ) * (scaler_val ** 2)
            mae_current = (
                torch.abs((samples_median - c_target) * eval_points)
            ) * scaler_val

            mse_total        += mse_current.sum().item()
            mae_total        += mae_current.sum().item()
            evalpoints_total += eval_points.sum().item()

    all_target            = torch.cat(all_target, dim=0)
    all_evalpoint         = torch.cat(all_evalpoint, dim=0)
    all_observed_point    = torch.cat(all_observed_point, dim=0)
    all_observed_time     = torch.cat(all_observed_time, dim=0)
    all_generated_samples = torch.cat(all_generated_samples, dim=0)

    rmse = float(np.sqrt(mse_total / evalpoints_total))
    mae  = float(mae_total / evalpoints_total)

    # CRPS in normalised space
    CRPS     = calc_quantile_CRPS(
        all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler_val
    )
    CRPS_sum = calc_quantile_CRPS_sum(
        all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler_val
    )

    # WAPE in raw sales space using dataset.inverse_transform
    target_np  = all_target.numpy()
    median_np  = all_generated_samples.median(dim=1).values.numpy()
    ep_np      = all_evalpoint.numpy()

    # Reverse StandardScaler → undo log1p
    def _to_raw(arr_np):
        flat = arr_np.reshape(-1, 1)
        unstd = flat * scaler_val + mean_scaler    # reverse StandardScaler step
        return np.expm1(unstd).reshape(arr_np.shape)

    target_raw = _to_raw(target_np)
    median_raw = _to_raw(median_np)

    target_t = torch.from_numpy(target_raw)
    median_t = torch.from_numpy(median_raw)
    ep_t     = torch.from_numpy(ep_np)
    wape     = calc_wape(target_t, median_t, ep_t)

    metrics = {
        "RMSE": rmse, "MAE": mae, "WAPE": wape,
        "CRPS": CRPS, "CRPS_sum": CRPS_sum,
    }

    suffix = f"_{tag}" if tag else ""
    if foldername:
        pk_path = os.path.join(foldername, f"generated_outputs{suffix}_nsample{nsample}.pk")
        with open(pk_path, "wb") as f:
            pickle.dump(
                [all_generated_samples, all_target, all_evalpoint,
                 all_observed_point, all_observed_time, scaler_val, mean_scaler],
                f,
            )
        metric_path = os.path.join(foldername, f"metrics{suffix}.json")
        with open(metric_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def sweep_evaluate(
    model,
    test_dataset: Dataset_HnM,
    batch_size:   int,
    num_workers:  int,
    nsample:      int,
    scaler_val:   float,
    mean_scaler:  float,
    foldername:   str,
) -> None:
    """Evaluate at all n_obs in {0,1,2,3,4} and print a comparison table."""

    sweep_results = {}
    for n_obs in range(5):
        test_dataset.n_obs = n_obs
        loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        console.log(f"[bold]Sweep n_obs={n_obs}[/]")
        metrics = evaluate_hnm(
            model, loader, test_dataset, nsample=nsample,
            scaler_val=scaler_val, mean_scaler=mean_scaler,
            foldername=foldername, tag=f"n{n_obs}",
        )
        sweep_results[n_obs] = metrics
        console.log(
            f"  n_obs={n_obs}: "
            f"RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
            f"WAPE={metrics['WAPE']*100:.1f}%  CRPS={metrics['CRPS']:.4f}  "
            f"CRPS_sum={metrics['CRPS_sum']:.4f}"
        )

    # Summary table
    console.print("\n[bold cyan]Sweep results summary (H&M)[/]")
    console.print(f"{'n_obs':>5} {'RMSE':>8} {'MAE':>8} {'WAPE%':>7} {'CRPS':>8} {'CRPS_sum':>10}")
    console.print("-" * 52)
    for n_obs, m in sweep_results.items():
        console.print(
            f"{n_obs:>5d} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
            f"{m['WAPE']*100:>6.1f}% {m['CRPS']:>8.4f} {m['CRPS_sum']:>10.4f}"
        )

    if foldername:
        with open(os.path.join(foldername, "metrics_sweep.json"), "w") as f:
            json.dump({str(k): v for k, v in sweep_results.items()}, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Config ----------------------------------------------------------
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- Foldername / resume --------------------------------------------
    resume_ckpt = None
    if args.resume:
        resume_ckpt = _resolve_resume_checkpoint(args.resume)

    if args.modelfolder:
        foldername = args.modelfolder
    elif resume_ckpt is not None:
        foldername = os.path.dirname(resume_ckpt)
    else:
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag    = f"n{args.n_obs_eval}" if not args.sweep else "sweep"
        foldername = os.path.join(_REPO_ROOT, "save", f"hnm_{run_tag}_{timestamp}")
        os.makedirs(foldername, exist_ok=True)

    # ---- Dataset / DataLoaders ------------------------------------------
    console.log(f"Loading H&M dataset from [bold]{args.processed_dir}[/] ...")
    (
        train_loader, val_loader, test_loader,
        train_dataset, val_dataset, test_dataset,
    ) = get_dataloader(
        processed_dir=args.processed_dir,
        batch_size=config["train"]["batch_size"],
        n_obs_train=(args.n_obs_min, args.n_obs_max),
        n_obs_eval=args.n_obs_eval,
        num_workers=4,
    )

    # ---- Scaler params for metrics --------------------------------------
    if train_dataset.scale and train_dataset.scaler is not None:
        scaler_val  = float(train_dataset.scaler.scale_[0])
        mean_scaler = float(train_dataset.scaler.mean_[0])
    else:
        scaler_val  = 1.0
        mean_scaler = 0.0

    # ---- Save config snapshot -------------------------------------------
    if not args.modelfolder:
        with open(os.path.join(foldername, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # ---- Build model ----------------------------------------------------
    console.log("Building [bold]RATD_HnM[/] ...")
    model = RATD_HnM(config, args.device).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.log(f"  Parameters: {n_params:,}")

    # ---- Train or load --------------------------------------------------
    if args.modelfolder:
        best_ckpt = os.path.join(args.modelfolder, "model_best.pth")
        last_ckpt = os.path.join(args.modelfolder, "model.pth")
        ckpt_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
        model.load_state_dict(
            torch.load(ckpt_path, map_location=args.device, weights_only=True)
        )
        console.log(f"Loaded weights from [bold]{ckpt_path}[/]")
    else:
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=val_loader,       # use explicit val split (no test leakage)
            valid_epoch_interval=args.val_interval,
            foldername=foldername,
            save_every=args.save_every,
            resume_checkpoint=resume_ckpt,
        )

    # ---- Evaluate -------------------------------------------------------
    if args.sweep:
        sweep_evaluate(
            model, test_dataset,
            batch_size=config["train"]["batch_size"],
            num_workers=4,
            nsample=args.nsample,
            scaler_val=scaler_val,
            mean_scaler=mean_scaler,
            foldername=foldername,
        )
    else:
        test_dataset.n_obs = args.n_obs_eval
        metrics = evaluate_hnm(
            model, test_loader, test_dataset,
            nsample=args.nsample,
            scaler_val=scaler_val,
            mean_scaler=mean_scaler,
            foldername=foldername,
            tag=f"n{args.n_obs_eval}",
        )
        console.log(f"\n[bold cyan]Evaluation results (n_obs={args.n_obs_eval})[/]")
        for k, v in metrics.items():
            val_str = f"{v*100:.2f}%" if k == "WAPE" else f"{v:.6f}"
            console.log(f"  {k:<10}: {val_str}")


if __name__ == "__main__":
    main()
