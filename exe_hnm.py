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

  # Resume interrupted run (LR restored from checkpoint automatically)
  python exe_hnm.py --resume save/hnm_n2_20260413_120000

  # Resume and override LR (e.g. after a loss spike)
  python exe_hnm.py --resume save/hnm_n2_20260413_120000 --lr 3e-4

  # Evaluate a specific checkpoint across all n_obs settings
  python exe_hnm.py --eval_checkpoint save/hnm_n2_.../model_best.pth --sweep

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
    make_progress,
    train,
)

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train / evaluate RATD_HnM on H&M dataset")

    p.add_argument("--config",        type=str,
                   default=os.path.join(_REPO_ROOT, "config", "hnm.yaml"))
    p.add_argument("--processed_dir", type=str,
                   default=os.path.join(os.path.dirname(_REPO_ROOT), "hnm_processed"),
                   help="Path to hnm_processed/ directory")
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int, default=1)

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
                   help="Skip training; load model_best.pth from this folder and evaluate")
    p.add_argument("--eval_checkpoint", type=str, default="",
                   help="Path to any specific .pth file to evaluate "
                        "(model_best.pth, checkpoint_epochN.pth, etc.). "
                        "Skips training entirely. Accepts both weights-only and "
                        "full resumable checkpoints. Output written to the same "
                        "directory as the checkpoint.")

    # Training checkpoints
    p.add_argument("--save_every",   type=int, default=10,
                   help="Save full checkpoint every N epochs (0 = only final+best)")
    p.add_argument("--val_interval", type=int, default=10,
                   help="Validate every N epochs for best-model selection")
    p.add_argument("--resume",       type=str, default="",
                   help="Resume training from checkpoint file or folder "
                        "(auto-selects checkpoint_latest.pth if a folder is given)")
    p.add_argument("--lr",           type=float, default=None,
                   help="Override the learning rate from the config YAML. "
                        "Useful when resuming after a loss spike (e.g. --lr 3e-4). "
                        "Without this flag, the scheduler-reduced LR is restored "
                        "from the checkpoint automatically.")
    p.add_argument("--num_workers",  type=int, default=0,
                   help="DataLoader worker processes (0 = main process, required on Kaggle)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_resume_checkpoint(path: str) -> str:
    """Return the path to a resumable checkpoint given a file or folder.

    Preference order when path is a directory:
      1. checkpoint_latest.pth
      2. Highest-epoch checkpoint_epochN.pth found in the folder
    """
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
    raise FileNotFoundError(
        f"No resumable checkpoint found at: {path}\n"
        "Make sure the path points to a checkpoint_*.pth file or a run folder."
    )


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
    WAPE is computed in raw sales space by reversing log1p + StandardScaler.

    Args:
        model:       Trained RATD_HnM.
        test_loader: DataLoader for the evaluation split.
        dataset:     The underlying Dataset_HnM (unused directly; kept for
                     API symmetry with evaluate_fashion).
        nsample:     Number of reverse-diffusion samples per batch.
        scaler_val:  StandardScaler scale_ (used for RMSE/MAE de-norm).
        mean_scaler: StandardScaler mean_ (used for RMSE/MAE de-norm).
        foldername:  If non-empty, save pickles and JSON metrics here.
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

    try:
        total_batches = len(test_loader)
    except TypeError:
        total_batches = None

    desc = f"Evaluating{(' ' + tag) if tag else ''}"

    with torch.no_grad():
        with make_progress() as progress:
            task = progress.add_task(desc, total=total_batches, metrics="RMSE=---  MAE=---")
            for test_batch in test_loader:
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, obs_points, obs_time = output

                samples     = samples.permute(0, 1, 3, 2)   # (B, nsample, L, K)
                c_target    = c_target.permute(0, 2, 1)     # (B, L, K)
                eval_points = eval_points.permute(0, 2, 1)
                obs_points  = obs_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1).values

                all_target.append(c_target.cpu())
                all_evalpoint.append(eval_points.cpu())
                all_observed_point.append(obs_points.cpu())
                all_observed_time.append(obs_time.cpu())
                all_generated_samples.append(samples.cpu())

                mse_current = (
                    ((samples_median - c_target) * eval_points) ** 2
                ) * (scaler_val ** 2)
                mae_current = (
                    torch.abs((samples_median - c_target) * eval_points)
                ) * scaler_val

                mse_total        += mse_current.sum().item()
                mae_total        += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                running_rmse = np.sqrt(mse_total / max(evalpoints_total, 1))
                running_mae  = mae_total / max(evalpoints_total, 1)
                progress.update(
                    task,
                    advance=1,
                    metrics=f"RMSE={running_rmse:.4f}  MAE={running_mae:.4f}",
                )

    all_target            = torch.cat(all_target, dim=0)
    all_evalpoint         = torch.cat(all_evalpoint, dim=0)
    all_observed_point    = torch.cat(all_observed_point, dim=0)
    all_observed_time     = torch.cat(all_observed_time, dim=0)
    all_generated_samples = torch.cat(all_generated_samples, dim=0)

    rmse = float(np.sqrt(mse_total / max(evalpoints_total, 1)))
    mae  = float(mae_total / max(evalpoints_total, 1))

    CRPS     = calc_quantile_CRPS(
        all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler_val
    )
    CRPS_sum = calc_quantile_CRPS_sum(
        all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler_val
    )

    # WAPE in raw sales space: reverse StandardScaler then undo log1p
    def _to_raw(arr_np):
        flat  = arr_np.reshape(-1, 1)
        unstd = flat * scaler_val + mean_scaler
        return np.expm1(unstd).reshape(arr_np.shape)

    target_raw = _to_raw(all_target.numpy())
    median_raw = _to_raw(all_generated_samples.median(dim=1).values.numpy())
    ep_np      = all_evalpoint.numpy()

    wape = calc_wape(
        torch.from_numpy(target_raw),
        torch.from_numpy(median_raw),
        torch.from_numpy(ep_np),
    )

    metrics = {
        "RMSE": rmse, "MAE": mae, "WAPE": wape,
        "CRPS": CRPS, "CRPS_sum": CRPS_sum,
    }

    label = f" [{tag}]" if tag else ""
    console.print(f"\n[bold cyan]Metrics{label}[/]")
    console.print(f"  RMSE     : [yellow]{rmse:.6f}[/]  (normalised scale)")
    console.print(f"  MAE      : [yellow]{mae:.6f}[/]  (normalised scale)")
    console.print(f"  WAPE     : [yellow]{wape * 100:.2f}%[/]")
    console.print(f"  CRPS     : [yellow]{CRPS:.6f}[/]")
    console.print(f"  CRPS_sum : [yellow]{CRPS_sum:.6f}[/]")

    if foldername:
        suffix   = f"_{tag}" if tag else ""
        pk_path  = os.path.join(foldername, f"generated_outputs{suffix}_nsample{nsample}.pk")
        with open(pk_path, "wb") as f:
            pickle.dump(
                [all_generated_samples, all_target, all_evalpoint,
                 all_observed_point, all_observed_time, scaler_val, mean_scaler],
                f,
            )
        json_path = os.path.join(foldername, f"metrics{suffix}.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


# ---------------------------------------------------------------------------
# Sweep: evaluate one checkpoint at n_obs in {0,1,2,3,4}
# ---------------------------------------------------------------------------

def sweep_evaluate(
    model,
    test_dataset: Dataset_HnM,
    batch_size:   int,
    num_workers:  int,
    nsample:      int,
    scaler_val:   float,
    mean_scaler:  float,
    foldername:   str,
) -> dict:
    """Evaluate at all n_obs in {0,1,2,3,4} and print a comparison table."""

    sweep_results = {}
    for n_obs in range(5):
        print(f"\n{'=' * 52}")
        print(f"  SWEEP  n_obs = {n_obs}"
              f"{'  (pure cold-start)' if n_obs == 0 else f'  ({n_obs}-period few-shot)'}")
        print(f"{'=' * 52}")

        test_dataset.n_obs = n_obs
        loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        metrics = evaluate_hnm(
            model, loader, test_dataset, nsample=nsample,
            scaler_val=scaler_val, mean_scaler=mean_scaler,
            foldername=foldername, tag=f"n{n_obs}",
        )
        sweep_results[n_obs] = metrics

    # Comparison table
    console.print(f"\n[bold cyan]Sweep Summary (H&M)[/]")
    console.print(f"  {'n_obs':<8}{'RMSE':<14}{'MAE':<14}{'WAPE %':<12}{'CRPS':<14}{'CRPS_sum'}")
    console.print("-" * 70)
    for n_obs, m in sweep_results.items():
        label = "cold" if n_obs == 0 else f"{n_obs}-shot"
        console.print(
            f"  {label:<8}"
            f"{m['RMSE']:<14.6f}"
            f"{m['MAE']:<14.6f}"
            f"{m['WAPE'] * 100:<12.2f}"
            f"{m['CRPS']:<14.6f}"
            f"{m['CRPS_sum']:.6f}"
        )

    if foldername:
        sweep_path = os.path.join(foldername, "metrics_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump({str(k): v for k, v in sweep_results.items()}, f, indent=4)
        console.log(f"Sweep results saved to [bold]{sweep_path}[/]")

    return sweep_results


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

    if args.lr is not None:
        config["train"]["lr"] = args.lr
        print(f"LR overridden to {args.lr:.2e} via --lr flag")

    print(json.dumps(config, indent=4))

    # ---- Resolve resume checkpoint ---------------------------------------
    resume_ckpt = None
    if args.resume:
        resume_ckpt = _resolve_resume_checkpoint(args.resume)
        print(f"Resuming from checkpoint: {resume_ckpt}")

    # ---- Output folder ---------------------------------------------------
    if args.modelfolder:
        foldername = args.modelfolder
    elif args.eval_checkpoint:
        foldername = os.path.dirname(os.path.abspath(args.eval_checkpoint))
    elif resume_ckpt is not None:
        foldername = os.path.dirname(resume_ckpt)
        print(f"Continuing run folder  : {foldername}")
    else:
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag    = f"n{args.n_obs_eval}" if not args.sweep else "sweep"
        foldername = os.path.join(_REPO_ROOT, "save", f"hnm_{run_tag}_{timestamp}")
        os.makedirs(foldername, exist_ok=True)

    print(f"\nRun folder : {foldername}")
    print(f"Device     : {args.device}\n")

    # ---- Dataset / DataLoaders ------------------------------------------
    (
        train_loader, val_loader, test_loader,
        train_dataset, val_dataset, test_dataset,
    ) = get_dataloader(
        processed_dir=args.processed_dir,
        batch_size=config["train"]["batch_size"],
        n_obs_train=(args.n_obs_min, args.n_obs_max),
        n_obs_eval=args.n_obs_eval,
        num_workers=args.num_workers,
    )

    # ---- Scaler params for metrics --------------------------------------
    if train_dataset.scale and train_dataset.scaler is not None:
        scaler_val  = float(train_dataset.scaler.scale_[0])
        mean_scaler = float(train_dataset.scaler.mean_[0])
    else:
        scaler_val  = 1.0
        mean_scaler = 0.0

    print(f"Sales scaler: scale={scaler_val:.6f}  mean={mean_scaler:.6f}\n")

    # ---- Save config snapshot (with CLI args) ---------------------------
    if not args.modelfolder:
        with open(os.path.join(foldername, "config.json"), "w") as f:
            json.dump({**config, "cli": vars(args)}, f, indent=4)

    # ---- Build model ----------------------------------------------------
    model = RATD_HnM(config, args.device).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ---- Train or load --------------------------------------------------
    if args.eval_checkpoint:
        # Evaluate a specific checkpoint file directly (weights-only or full).
        ckpt_path = args.eval_checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(_REPO_ROOT, ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--eval_checkpoint not found: {ckpt_path}")
        print(f"Loading checkpoint for evaluation: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        if isinstance(raw, dict) and "model_state" in raw:
            model.load_state_dict(raw["model_state"])
            epoch    = raw.get("epoch", "?")
            best_val = raw.get("best_valid_loss", None)
            val_str  = f"{best_val:.6f}" if isinstance(best_val, float) else "?"
            print(f"  Full checkpoint  epoch={epoch}  best_val_loss={val_str}")
        else:
            model.load_state_dict(raw)
            print("  Weights-only checkpoint loaded.")
    elif args.modelfolder:
        best_ckpt = os.path.join(args.modelfolder, "model_best.pth")
        last_ckpt = os.path.join(args.modelfolder, "model.pth")
        ckpt_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=args.device, weights_only=False)
        )
    else:
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=val_loader,        # dedicated val split, no test leakage
            valid_epoch_interval=args.val_interval,
            foldername=foldername,
            save_every=args.save_every,
            resume_checkpoint=resume_ckpt,  # None = fresh run
            lr_override=args.lr,            # None = keep scheduler-restored LR
        )

    # ---- Evaluate -------------------------------------------------------
    if args.sweep:
        sweep_evaluate(
            model, test_dataset,
            batch_size=config["train"]["batch_size"],
            num_workers=args.num_workers,
            nsample=args.nsample,
            scaler_val=scaler_val,
            mean_scaler=mean_scaler,
            foldername=foldername,
        )
    else:
        test_dataset.n_obs = args.n_obs_eval
        evaluate_hnm(
            model, test_loader, test_dataset,
            nsample=args.nsample,
            scaler_val=scaler_val,
            mean_scaler=mean_scaler,
            foldername=foldername,
            tag=f"n{args.n_obs_eval}",
        )


if __name__ == "__main__":
    main()
