"""baselines/lstm_baseline.py

Tier-1 RNN-LSTM baselines for few-shot (n_obs >= 1) forecasting:

  LSTM-NoAttr  -- 2-layer encoder LSTM on the observed sales/discount history;
                  linear decoder to the remaining (12 - n_obs) weeks.
                  No product attribute information.

  LSTM-Attr    -- Same LSTM, but each encoder input step is augmented with
                  a 32-dim projection of the 513-dim product embedding.
                  Shows whether attribute information helps a recurrent model.

One checkpoint is trained per (model variant, n_obs) combination, since the
decoder output dimension changes with n_obs.  Four n_obs values are supported:
1, 2, 3, 4.

Usage:

  # Train LSTM-Attr at n_obs=2
  python baselines/lstm_baseline.py --n_obs 2 --use_attr --mode train

  # Evaluate a saved checkpoint
  python baselines/lstm_baseline.py --n_obs 2 --use_attr --mode eval \\
      --modelfolder baselines/save/lstm_attr_n2

  # Sweep all n_obs (train + eval, no attributes)
  python baselines/lstm_baseline.py --mode sweep

  # Sweep all n_obs (train + eval, with attributes)
  python baselines/lstm_baseline.py --mode sweep --use_attr
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dataset_visuelle2 import Dataset_Visuelle2
from utils import calc_quantile_CRPS, calc_quantile_CRPS_sum, calc_wape, console


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Seq2seq LSTM forecaster for few-shot sales prediction.

    The encoder runs a 2-layer LSTM over the n_obs observed weeks.  Each
    input step is the K-channel observation (sales + discount), optionally
    concatenated with a projected product attribute vector.

    The decoder is a single linear layer that maps the final encoder hidden
    state directly to the (12 - n_obs) * K future values.

    Default hidden_dim=256 and attr_proj_dim=64 are chosen so that the
    parameter count is comparable to RATD_Fashion (~985K):
      - LSTM-NoAttr (hidden=256, 2 layers): ~796K parameters
      - LSTM-Attr   (hidden=256, 2 layers, attr_proj=64): ~894K parameters

    Args:
        n_obs:        Number of observed weeks given as context (1-4).
        pred_len:     Total season length (12). Decoder predicts pred_len - n_obs.
        k_channels:   Number of input channels (2: sales + discount).
        hidden_dim:   LSTM hidden size. Default 256.
        n_layers:     Number of stacked LSTM layers. Default 2.
        attr_dim:     Input dimension of product embedding (513). 0 = no attribute.
        attr_proj_dim: Dimension of the attribute projection layer. Default 64.
    """

    def __init__(
        self,
        n_obs:        int,
        pred_len:     int   = 12,
        k_channels:   int   = 2,
        hidden_dim:   int   = 256,
        n_layers:     int   = 2,
        attr_dim:     int   = 0,
        attr_proj_dim: int  = 64,
    ):
        super().__init__()
        self.n_obs     = n_obs
        self.pred_len  = pred_len
        self.k_channels = k_channels
        self.use_attr  = attr_dim > 0
        self.future_len = pred_len - n_obs

        # Optional attribute projection
        if self.use_attr:
            self.attr_proj = nn.Linear(attr_dim, attr_proj_dim)
            lstm_input_dim = k_channels + attr_proj_dim
        else:
            lstm_input_dim = k_channels

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0.0,
        )

        # Direct regression: hidden_dim -> future_len * k_channels
        self.decoder = nn.Linear(hidden_dim, self.future_len * k_channels)

    def forward(self, obs: torch.Tensor, product_emb: torch.Tensor = None) -> torch.Tensor:
        """Run encoder LSTM + decoder.

        Args:
            obs:         (B, n_obs, K) observed weeks (normalised).
            product_emb: (B, attr_dim) product embedding. Required when
                         use_attr=True, ignored otherwise.

        Returns:
            pred: (B, future_len, K) predicted future weeks (normalised).
        """
        B = obs.size(0)

        if self.use_attr and product_emb is not None:
            attr_vec = self.attr_proj(product_emb)                        # (B, attr_proj_dim)
            attr_exp = attr_vec.unsqueeze(1).expand(-1, self.n_obs, -1)   # (B, n_obs, attr_proj_dim)
            enc_input = torch.cat([obs, attr_exp], dim=2)                 # (B, n_obs, K+attr_proj_dim)
        else:
            enc_input = obs                                                # (B, n_obs, K)

        _, (h_n, _) = self.lstm(enc_input)   # h_n: (n_layers, B, hidden_dim)
        h_last = h_n[-1]                     # (B, hidden_dim) -- last layer

        out_flat = self.decoder(h_last)                   # (B, future_len * K)
        pred = out_flat.view(B, self.future_len, self.k_channels)
        return pred


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_lstm(
    model:       LSTMForecaster,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    foldername:   str,
    epochs:       int = 50,
    lr:           float = 1e-3,
    device:       str = "cpu",
) -> None:
    """Train LSTMForecaster and save the best checkpoint.

    Best model is selected on validation MSE (normalised space).

    Args:
        model:        Untrained LSTMForecaster.
        train_loader: DataLoader for training set (n_obs fixed).
        val_loader:   DataLoader for test set (n_obs fixed).
        foldername:   Directory to save model_best.pth and model.pth.
        epochs:       Number of training epochs.
        lr:           Adam learning rate.
        device:       'cuda' or 'cpu'.
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_val  = float("inf")
    os.makedirs(foldername, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            obs_data    = batch["observed_data"].to(device)       # (B, 12, 2)
            product_emb = batch["product_emb"].to(device)         # (B, 513)
            gt_mask     = batch["gt_mask"].to(device)             # (B, 12, 2)

            n_obs = model.n_obs
            obs        = obs_data[:, :n_obs, :]                    # (B, n_obs, 2)
            target     = obs_data[:, n_obs:, :]                    # (B, future_len, 2)
            eval_mask  = (1.0 - gt_mask)[:, n_obs:, :]            # (B, future_len, 2)

            emb_arg = product_emb if model.use_attr else None
            pred    = model(obs, emb_arg)                          # (B, future_len, 2)

            # MSE only on held-out positions (eval_mask=1)
            loss = criterion(pred * eval_mask, target * eval_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        train_loss /= n_batches

        # Validation
        model.eval()
        val_loss  = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                obs_data    = batch["observed_data"].to(device)
                product_emb = batch["product_emb"].to(device)
                gt_mask     = batch["gt_mask"].to(device)

                obs       = obs_data[:, :n_obs, :]
                target    = obs_data[:, n_obs:, :]
                eval_mask = (1.0 - gt_mask)[:, n_obs:, :]

                emb_arg = product_emb if model.use_attr else None
                pred    = model(obs, emb_arg)
                loss    = criterion(pred * eval_mask, target * eval_mask)
                val_loss    += loss.item()
                val_batches += 1

        val_loss /= val_batches
        tag = "[green]NEW BEST[/]" if val_loss < best_val else ""
        console.log(
            f"  Epoch {epoch:3d}/{epochs-1}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  {tag}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(foldername, "model_best.pth"))

    torch.save(model.state_dict(), os.path.join(foldername, "model.pth"))
    console.log(f"[bold green]Training complete.[/]  Best val={best_val:.6f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_lstm(
    model:        LSTMForecaster,
    test_loader:  DataLoader,
    scaler_val:   float,
    mean_scaler:  float,
    foldername:   str = "",
    tag:          str = "",
    device:       str = "cpu",
) -> dict:
    """Evaluate a trained LSTMForecaster and return metric dict.

    Args:
        model:        Trained LSTMForecaster (loaded best checkpoint).
        test_loader:  DataLoader for test set with fixed n_obs.
        scaler_val:   StandardScaler scale_ (sales channel).
        mean_scaler:  StandardScaler mean_ (sales channel).
        foldername:   If non-empty, save results JSON here.
        tag:          Filename suffix (e.g. 'lstm_attr_n2').
        device:       'cuda' or 'cpu'.

    Returns:
        dict with keys RMSE, MAE, WAPE, CRPS, CRPS_sum.
    """
    model.eval()
    model.to(device)

    all_targets  = []
    all_preds    = []
    all_ev_masks = []
    n_obs        = model.n_obs

    with torch.no_grad():
        for batch in test_loader:
            obs_data    = batch["observed_data"].to(device)    # (B, 12, 2)
            product_emb = batch["product_emb"].to(device)      # (B, 513)
            gt_mask     = batch["gt_mask"].to(device)          # (B, 12, 2)

            obs       = obs_data[:, :n_obs, :]
            target    = obs_data[:, n_obs:, :]                 # (B, future_len, 2)
            eval_mask = (1.0 - gt_mask)[:, n_obs:, :]

            emb_arg = product_emb if model.use_attr else None
            pred    = model(obs, emb_arg)                      # (B, future_len, 2)

            all_targets.append(target.cpu())
            all_preds.append(pred.cpu())
            all_ev_masks.append(eval_mask.cpu())

    targets_t  = torch.cat(all_targets,  dim=0)    # (N, future_len, 2)
    preds_t    = torch.cat(all_preds,    dim=0)
    ev_masks_t = torch.cat(all_ev_masks, dim=0)

    # Point metrics in normalised space
    diff      = (preds_t - targets_t) * ev_masks_t
    mse       = float(((diff * scaler_val) ** 2).sum() / ev_masks_t.sum())
    mae       = float((torch.abs(diff) * scaler_val).sum() / ev_masks_t.sum())
    rmse      = float(np.sqrt(mse))

    # WAPE on sales channel in de-normalised space
    wape = calc_wape(
        targets_t[..., :1],
        preds_t[..., :1],
        ev_masks_t[..., :1],
        scaler=scaler_val,
        mean_scaler=mean_scaler,
    )

    # CRPS: replicate the point prediction to form a degenerate distribution
    nsample   = 20
    samples_t = preds_t.unsqueeze(1).expand(-1, nsample, -1, -1)  # (N, 20, future_len, 2)
    crps      = calc_quantile_CRPS(
        targets_t, samples_t, ev_masks_t, mean_scaler, scaler_val
    )
    crps_sum  = calc_quantile_CRPS_sum(
        targets_t, samples_t, ev_masks_t, mean_scaler, scaler_val
    )

    metrics = {
        "RMSE": rmse, "MAE": mae, "WAPE": wape,
        "CRPS": crps, "CRPS_sum": crps_sum,
    }

    model_name = f"LSTM-{'Attr' if model.use_attr else 'NoAttr'} n_obs={n_obs}"
    console.log(
        f"[bold cyan]{model_name}[/]  "
        f"RMSE={rmse:.4f}  MAE={mae:.4f}  WAPE={wape*100:.1f}%  "
        f"CRPS={crps:.4f}  CRPS_sum={crps_sum:.4f}"
    )

    if foldername and tag:
        out_path = os.path.join(foldername, f"metrics_{tag}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Tier-1 LSTM baselines: LSTM-NoAttr and LSTM-Attr"
    )
    p.add_argument("--data_root",     type=str,
                   default="/home/shu_sho_bhit/BTP_2")
    p.add_argument("--processed_dir", type=str,
                   default="/home/shu_sho_bhit/BTP_2/visuelle2_processed")
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_obs",         type=int, default=2,
                   help="Observation weeks (1-4). Must be >= 1. Ignored when --mode sweep.")
    p.add_argument("--use_attr",      action="store_true",
                   help="Use product attribute embedding (LSTM-Attr variant).")
    p.add_argument("--mode",          type=str,
                   choices=["train", "eval", "sweep"],
                   default="train",
                   help="train: fit one model. eval: load and evaluate. "
                        "sweep: train+eval all n_obs in {1,2,3,4}.")
    p.add_argument("--modelfolder",   type=str, default="",
                   help="Folder containing model_best.pth (eval mode only).")
    p.add_argument("--save_dir",      type=str,
                   default=os.path.join(_REPO_ROOT, "baselines", "save"),
                   help="Root directory for saving checkpoints.")
    p.add_argument("--output",        type=str, default="",
                   help="JSON file for sweep results.")
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--hidden_dim",    type=int, default=256,
                   help="LSTM hidden size. Default 256 gives ~796K params (NoAttr) "
                        "or ~894K params (Attr), comparable to RATD_Fashion (~985K).")
    p.add_argument("--n_layers",      type=int, default=2)
    p.add_argument("--attr_proj_dim", type=int, default=64,
                   help="Attribute projection dim before LSTM input. Default 64.")
    p.add_argument("--num_workers",   type=int, default=0,
                   help="DataLoader worker count (0 = main process, required on Kaggle).")
    return p.parse_args()


def _get_folder(args, n_obs: int) -> str:
    variant = "attr" if args.use_attr else "noattr"
    return os.path.join(args.save_dir, f"lstm_{variant}_n{n_obs}")


def _build_datasets(args, n_obs: int):
    """Build train/test Dataset_Visuelle2 with the given n_obs."""
    train_ds = Dataset_Visuelle2(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        flag="train",
        n_obs=n_obs,
    )
    test_ds = Dataset_Visuelle2(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        flag="test",
        n_obs=n_obs,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_ds, test_ds, train_loader, test_loader


def _build_model(args, n_obs: int) -> LSTMForecaster:
    attr_dim = 513 if args.use_attr else 0
    return LSTMForecaster(
        n_obs=n_obs,
        pred_len=12,
        k_channels=2,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        attr_dim=attr_dim,
        attr_proj_dim=args.attr_proj_dim,
    )


def _run_single(args, n_obs: int) -> dict:
    """Train (if needed) and evaluate one (variant, n_obs) combination."""
    variant = "LSTM-Attr" if args.use_attr else "LSTM-NoAttr"
    console.rule(f"[bold]{variant}  n_obs={n_obs}[/]")

    train_ds, test_ds, train_loader, test_loader = _build_datasets(args, n_obs)

    if train_ds.scale and train_ds.scaler is not None:
        scaler_val  = float(train_ds.scaler.scale_[0])
        mean_scaler = float(train_ds.scaler.mean_[0])
    else:
        scaler_val, mean_scaler = 1.0, 0.0

    foldername = args.modelfolder if args.modelfolder else _get_folder(args, n_obs)
    model      = _build_model(args, n_obs)

    if args.mode in ("train", "sweep"):
        console.log(f"Training [bold]{variant}[/]  n_obs={n_obs}  epochs={args.epochs}")
        train_lstm(
            model, train_loader, test_loader,
            foldername=foldername,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
        )

    # Load best checkpoint for evaluation
    best_ckpt = os.path.join(foldername, "model_best.pth")
    last_ckpt = os.path.join(foldername, "model.pth")
    ckpt_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
    model.load_state_dict(
        torch.load(ckpt_path, map_location=args.device, weights_only=True)
    )
    console.log(f"Loaded weights from [bold]{ckpt_path}[/]")

    tag = f"lstm_{'attr' if args.use_attr else 'noattr'}_n{n_obs}"
    return evaluate_lstm(
        model, test_loader,
        scaler_val=scaler_val,
        mean_scaler=mean_scaler,
        foldername=foldername,
        tag=tag,
        device=args.device,
    )


def main():
    args = parse_args()

    if args.mode == "sweep":
        sweep_results = {}
        for n_obs in [1, 2, 3, 4]:
            sweep_results[n_obs] = _run_single(args, n_obs)

        variant = "LSTM-Attr" if args.use_attr else "LSTM-NoAttr"
        console.print(f"\n[bold cyan]Sweep results ({variant})[/]")
        console.print(
            f"{'n_obs':>5} {'RMSE':>8} {'MAE':>8} {'WAPE%':>7} {'CRPS':>8} {'CRPS_sum':>10}"
        )
        console.print("-" * 52)
        for n_obs, m in sweep_results.items():
            console.print(
                f"{n_obs:>5d} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
                f"{m['WAPE']*100:>6.1f}% {m['CRPS']:>8.4f} {m['CRPS_sum']:>10.4f}"
            )

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({str(k): v for k, v in sweep_results.items()}, f, indent=2)
            console.log(f"Sweep results saved to [bold]{args.output}[/]")

    else:
        _run_single(args, args.n_obs)


if __name__ == "__main__":
    main()
