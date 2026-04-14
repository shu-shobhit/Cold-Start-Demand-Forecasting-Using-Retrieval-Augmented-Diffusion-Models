"""exe_fashion.py

Training and evaluation entrypoint for the cold-start fashion forecasting
experiment (Visuelle 2.0) with RATD_Fashion.

Usage examples
--------------
# Train (n_obs ∈ {0,1,2,3,4} sampled randomly during training):
  conda run -n ML python exe_fashion.py --device cuda

# Evaluate at a single n_obs setting from a saved checkpoint:
  conda run -n ML python exe_fashion.py \\
      --modelfolder save/fashion_visuelle2_n0_YYYYMMDD_HHMMSS \\
      --n_obs_eval 0 --nsample 50

# Evaluate a checkpoint across ALL n_obs settings in one pass:
  conda run -n ML python exe_fashion.py \\
      --modelfolder save/fashion_visuelle2_n0_YYYYMMDD_HHMMSS \\
      --sweep --nsample 50
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

from dataset_visuelle2 import Dataset_Visuelle2, get_dataloader
from main_model_fashion import RATD_Fashion
from utils import (
    calc_quantile_CRPS,
    calc_quantile_CRPS_sum,
    calc_wape,
    console,
    make_progress,
    train,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="RATD_Fashion – Visuelle 2.0 cold-start forecasting"
    )
    p.add_argument("--config",        type=str,
                   default="config/visuelle2.yaml",
                   help="Path to the YAML config file (relative to repo root)")
    p.add_argument("--data_root",     type=str,
                   default="/home/shu_sho_bhit/BTP_2",
                   help="Root directory containing the raw visuelle2/ folder")
    p.add_argument("--processed_dir", type=str,
                   default="/home/shu_sho_bhit/BTP_2/visuelle2_processed",
                   help="Directory with precomputed .pt files from scripts/")
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--n_obs_eval",    type=int, default=0,
                   help="Fixed n_obs used for single-n_obs evaluation "
                        "(ignored in --sweep mode)")
    p.add_argument("--n_obs_min",     type=int, default=0,
                   help="Min n_obs sampled during training (inclusive)")
    p.add_argument("--n_obs_max",     type=int, default=4,
                   help="Max n_obs sampled during training (inclusive)")
    p.add_argument("--nsample",       type=int, default=50,
                   help="Number of reverse-diffusion samples per batch "
                        "at evaluation")
    p.add_argument("--modelfolder",   type=str, default="",
                   help="If non-empty, skip training and load model.pth "
                        "from this folder")
    p.add_argument("--num_workers",   type=int, default=0)
    p.add_argument("--save_every",    type=int, default=1,
                   help="Save an intermediate checkpoint every N epochs "
                        "(0 = only save the final model and best model)")
    p.add_argument("--val_interval",  type=int, default=1,
                   help="Run validation every N epochs during training "
                        "(uses test set for best-model selection)")
    p.add_argument("--resume",        type=str, default="",
                   help="Resume training: path to a checkpoint_*.pth file, "
                        "or a run folder (auto-selects checkpoint_latest.pth "
                        "or the highest-epoch checkpoint found there)")
    p.add_argument("--lr",           type=float, default=None,
                   help="Override the learning rate from the config YAML. "
                        "Useful when resuming after a loss spike to restart "
                        "with a smaller LR (e.g. --lr 3e-4).")
    p.add_argument("--sweep",         action="store_true",
                   help="Evaluate at n_obs ∈ {0,1,2,3,4} and print "
                        "a comparison table")
    p.add_argument("--eval_checkpoint", type=str, default="",
                   help="Path to any specific .pth file to evaluate "
                        "(e.g. model_best.pth, checkpoint_epoch40.pth). "
                        "Skips training entirely. Accepts both weights-only "
                        "files (model_best / model.pth) and full resumable "
                        "checkpoints (checkpoint_*.pth). Output files are "
                        "written to the same directory as the checkpoint.")
    return p.parse_args()


def _resolve_resume_checkpoint(path: str) -> str:
    """Return the path to a full resumable checkpoint given a file or folder.

    Preference order when ``path`` is a directory:
      1. ``checkpoint_latest.pth``  — most convenient single target
      2. Highest-epoch ``checkpoint_epochN.pth`` found in the folder

    Args:
        path: Path to a ``.pth`` file or a run folder.

    Returns:
        str: Resolved path to a checkpoint file.

    Raises:
        FileNotFoundError: If no suitable checkpoint can be found.
    """
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        latest = os.path.join(path, "checkpoint_latest.pth")
        if os.path.exists(latest):
            return latest
        # Fall back to highest epoch checkpoint
        pattern = os.path.join(path, "checkpoint_epoch*.pth")
        candidates = glob.glob(pattern)
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
# Fashion-specific evaluation (sales channel only, RMSE/MAE/WAPE/CRPS)
# ---------------------------------------------------------------------------

def evaluate_fashion(
    model,
    test_loader,
    nsample:      int   = 50,
    scaler:       float = 1.0,
    mean_scaler:  float = 0.0,
    foldername:   str   = "",
    tag:          str   = "",
):
    """Evaluate RATD_Fashion on the Visuelle 2.0 test split.

    Metrics are computed on the **sales channel only** (channel index 0).
    Both RMSE / MAE (in the scaler-normalised space) and WAPE / CRPS /
    CRPS_sum (properly de-normalised via ``scaler`` and ``mean_scaler``) are
    reported.

    Args:
        model:       Trained RATD_Fashion model.
        test_loader: DataLoader for the test split.
        nsample:     Number of forecast samples per batch.
        scaler:      Multiplicative factor for de-normalisation
                     (``StandardScaler.scale_[0]``).
        mean_scaler: Additive offset for de-normalisation
                     (``StandardScaler.mean_[0]``).
        foldername:  Output directory for generated samples and metric files.
        tag:         Optional suffix appended to output file names (e.g.
                     ``"n0"`` for pure cold-start).

    Returns:
        dict: ``{"RMSE": ..., "MAE": ..., "WAPE": ..., "CRPS": ...,
                 "CRPS_sum": ...}``
    """

    model.eval()

    all_target   = []
    all_evalpt   = []
    all_obspt    = []
    all_obstime  = []
    all_samples  = []

    mse_total = mae_total = evalpts_total = 0.0

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
                # output: (samples, observed_data, target_mask, observed_mask, tp)
                # shapes: (B,nsample,K,L), (B,K,L), (B,K,L), (B,K,L), (B,L)
                samples, c_target, eval_points, obs_points, obs_time = output

                # Permute to (B, nsample/1, L, K) for utils compatibility
                samples     = samples.permute(0, 1, 3, 2)      # (B,nsample,L,K)
                c_target    = c_target.permute(0, 2, 1)         # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)      # (B,L,K)
                obs_points  = obs_points.permute(0, 2, 1)       # (B,L,K)

                # --- point metrics on sales channel (index 0) ---
                sales_target = c_target[..., 0]                 # (B,L)
                sales_eval   = eval_points[..., 0]              # (B,L)
                sales_median = samples[..., 0].median(dim=1).values  # (B,L)

                mse_total     += ((sales_median - sales_target) * sales_eval).pow(2).sum().item()
                mae_total     += torch.abs((sales_median - sales_target) * sales_eval).sum().item()
                evalpts_total += sales_eval.sum().item()

                all_target.append(c_target.cpu())
                all_evalpt.append(eval_points.cpu())
                all_obspt.append(obs_points.cpu())
                all_obstime.append(obs_time.cpu())
                all_samples.append(samples.cpu())

                running_rmse = np.sqrt(mse_total / max(evalpts_total, 1))
                running_mae  = mae_total / max(evalpts_total, 1)
                progress.update(
                    task,
                    advance=1,
                    metrics=f"RMSE={running_rmse:.4f}  MAE={running_mae:.4f}",
                )

    # --- Aggregate across all batches ---
    all_target  = torch.cat(all_target,  dim=0)   # (N, L, K)
    all_evalpt  = torch.cat(all_evalpt,  dim=0)   # (N, L, K)
    all_obspt   = torch.cat(all_obspt,   dim=0)   # (N, L, K)
    all_obstime = torch.cat(all_obstime, dim=0)   # (N, L)
    all_samples = torch.cat(all_samples, dim=0)   # (N, nsample, L, K)

    # Sales channel tensors for WAPE / CRPS (keep trailing K=1 dim for utils)
    target_sales  = all_target[..., 0].unsqueeze(-1)    # (N, L, 1)
    evalpt_sales  = all_evalpt[..., 0].unsqueeze(-1)    # (N, L, 1)
    samples_sales = all_samples[..., 0].unsqueeze(-1)   # (N, nsample, L, 1)

    rmse = np.sqrt(mse_total / max(evalpts_total, 1))
    mae  = mae_total / max(evalpts_total, 1)
    wape = calc_wape(
        target_sales,
        samples_sales.median(dim=1).values,
        evalpt_sales,
        scaler=scaler,
        mean_scaler=mean_scaler,
    )
    crps = calc_quantile_CRPS(
        target_sales, samples_sales, evalpt_sales, mean_scaler, scaler,
    )
    crps_sum = calc_quantile_CRPS_sum(
        target_sales, samples_sales, evalpt_sales, mean_scaler, scaler,
    )

    metrics = {
        "RMSE":     rmse,
        "MAE":      mae,
        "WAPE":     wape,
        "CRPS":     crps,
        "CRPS_sum": crps_sum,
    }

    label = f" [{tag}]" if tag else ""
    console.print(f"\n[bold cyan]Metrics{label}[/]")
    console.print(f"  RMSE     : [yellow]{rmse:.6f}[/]  (normalised scale)")
    console.print(f"  MAE      : [yellow]{mae:.6f}[/]  (normalised scale)")
    console.print(f"  WAPE     : [yellow]{wape * 100:.2f}%[/]")
    console.print(f"  CRPS     : [yellow]{crps:.6f}[/]")
    console.print(f"  CRPS_sum : [yellow]{crps_sum:.6f}[/]")

    if foldername:
        suffix = f"_{tag}" if tag else ""
        pk_path = os.path.join(
            foldername, f"generated_outputs_nsample{nsample}{suffix}.pk"
        )
        with open(pk_path, "wb") as f:
            pickle.dump(
                [all_samples, all_target, all_evalpt, all_obspt, all_obstime,
                 scaler, mean_scaler],
                f,
            )
        metric_path = os.path.join(
            foldername, f"result_nsample{nsample}{suffix}.pk"
        )
        with open(metric_path, "wb") as f:
            pickle.dump(metrics, f)
        json_path = os.path.join(foldername, f"metrics{suffix}.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


# ---------------------------------------------------------------------------
# Sweep: evaluate one checkpoint at n_obs ∈ {0,1,2,3,4}
# ---------------------------------------------------------------------------

def sweep_evaluate(
    model,
    test_dataset: Dataset_Visuelle2,
    batch_size:   int,
    num_workers:  int,
    nsample:      int,
    scaler:       float,
    mean_scaler:  float,
    foldername:   str,
):
    """Evaluate model at all n_obs settings and print a comparison table.

    Reuses the same ``test_dataset`` object — only ``n_obs`` is mutated
    between runs so the dataset files are loaded only once.

    Args:
        model:        Trained RATD_Fashion model.
        test_dataset: ``Dataset_Visuelle2`` test split (already loaded).
        batch_size:   Batch size for evaluation DataLoaders.
        num_workers:  DataLoader worker count.
        nsample:      Number of forecast samples per batch.
        scaler:       Multiplicative de-normalisation factor.
        mean_scaler:  Additive de-normalisation offset.
        foldername:   Output directory for per-n_obs metric files.

    Returns:
        dict: ``{n_obs_value: metrics_dict}`` for all n_obs settings.
    """

    sweep_results = {}

    for n_obs in range(5):   # 0, 1, 2, 3, 4
        print(f"\n{'=' * 52}")
        print(f"  SWEEP  n_obs = {n_obs}"
              f"{'  (pure cold-start)' if n_obs == 0 else f'  ({n_obs}-week few-shot)'}")
        print(f"{'=' * 52}")

        # Mutate the fixed n_obs in the shared dataset object
        test_dataset.n_obs = n_obs
        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        metrics = evaluate_fashion(
            model, loader,
            nsample=nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            tag=f"n{n_obs}",
        )
        sweep_results[n_obs] = metrics

    # --- Comparison table ---
    console.print(f"\n[bold cyan]Sweep Summary[/]")
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

    # Save combined sweep JSON
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

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Config ----------------------------------------------------------
    repo_root   = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(repo_root, args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.lr is not None:
        config["train"]["lr"] = args.lr
        print(f"LR overridden to {args.lr:.2e} via --lr flag")

    print(json.dumps(config, indent=4))

    # ---- Resolve resume checkpoint (if requested) ------------------------
    resume_ckpt = None
    if args.resume:
        resume_ckpt = _resolve_resume_checkpoint(args.resume)
        print(f"Resuming from checkpoint: {resume_ckpt}")

    # ---- Output folder ---------------------------------------------------
    if resume_ckpt is not None:
        # Continue saving into the *same* folder as the checkpoint
        foldername = os.path.dirname(resume_ckpt)
        print(f"Continuing run folder  : {foldername}")
    elif args.eval_checkpoint:
        # Write eval outputs next to the checkpoint being evaluated
        foldername = os.path.dirname(os.path.abspath(args.eval_checkpoint))
        print(f"Eval output folder     : {foldername}")
    else:
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag    = f"n{args.n_obs_eval}" if not args.sweep else "sweep"
        foldername = os.path.join(
            repo_root, "save", f"fashion_visuelle2_{run_tag}_{timestamp}"
        )
        os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump({**config, "cli": vars(args)}, f, indent=4)

    print(f"\nRun folder : {foldername}")
    print(f"Device     : {args.device}\n")

    # ---- Data ------------------------------------------------------------
    train_loader, test_loader, train_dataset, test_dataset = get_dataloader(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        batch_size=config["train"]["batch_size"],
        n_obs_train=(args.n_obs_min, args.n_obs_max),
        n_obs_eval=args.n_obs_eval,
        num_workers=args.num_workers,
    )

    # Extract the sales-channel (ch 0) scaler parameters for proper
    # de-normalisation in WAPE / CRPS metrics.
    if train_dataset.scale:
        scaler_val      = float(train_dataset.scaler.scale_[0])
        mean_scaler_val = float(train_dataset.scaler.mean_[0])
    else:
        scaler_val      = 1.0
        mean_scaler_val = 0.0

    print(f"Sales scaler: scale={scaler_val:.6f}  mean={mean_scaler_val:.6f}\n")

    # ---- Model -----------------------------------------------------------
    model = RATD_Fashion(config, args.device).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ---- Train or load ---------------------------------------------------
    if args.eval_checkpoint:
        # Evaluate a specific checkpoint file directly (weights-only or full).
        ckpt_path = args.eval_checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(repo_root, ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--eval_checkpoint not found: {ckpt_path}")
        print(f"Loading checkpoint for evaluation: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        if isinstance(raw, dict) and "model_state" in raw:
            # Full resumable checkpoint (checkpoint_epochN.pth / checkpoint_latest.pth)
            model.load_state_dict(raw["model_state"])
            epoch      = raw.get("epoch", "?")
            best_val   = raw.get("best_valid_loss", None)
            val_str    = f"{best_val:.6f}" if isinstance(best_val, float) else "?"
            print(f"  Full checkpoint  epoch={epoch}  best_val_loss={val_str}")
        else:
            # Weights-only checkpoint (model_best.pth / model.pth)
            model.load_state_dict(raw)
            print("  Weights-only checkpoint loaded.")
    elif args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=test_loader,           # use test set for best-model selection
            valid_epoch_interval=args.val_interval,
            foldername=foldername,
            save_every=args.save_every,
            resume_checkpoint=resume_ckpt,      # None = fresh run
        )
    else:
        # Prefer model_best.pth (lowest val loss) over model.pth (last epoch)
        best_ckpt = os.path.join(args.modelfolder, "model_best.pth")
        last_ckpt = os.path.join(args.modelfolder, "model.pth")
        if not os.path.isabs(best_ckpt):
            best_ckpt = os.path.join(repo_root, best_ckpt)
            last_ckpt = os.path.join(repo_root, last_ckpt)
        ckpt_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device,
                                         weights_only=False))

    # ---- Evaluate --------------------------------------------------------
    if args.sweep:
        sweep_evaluate(
            model,
            test_dataset,
            batch_size=config["train"]["batch_size"],
            num_workers=args.num_workers,
            nsample=args.nsample,
            scaler=scaler_val,
            mean_scaler=mean_scaler_val,
            foldername=foldername,
        )
    else:
        evaluate_fashion(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler_val,
            mean_scaler=mean_scaler_val,
            foldername=foldername,
            tag=f"n{args.n_obs_eval}",
        )


if __name__ == "__main__":
    main()
