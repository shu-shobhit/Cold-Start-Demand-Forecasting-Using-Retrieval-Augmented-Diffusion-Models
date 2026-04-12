"""Training and evaluation utilities for RATD forecasting experiments.

The helpers in this module implement the training loop, checkpoint saving,
forecast metric computation, and the batch-wise evaluation pipeline used by
the forecasting entrypoint.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

def _save_full_checkpoint(path, model, optimizer, lr_scheduler, epoch, best_valid_loss):
    """Save a resumable training checkpoint (model + optimiser + scheduler state).

    The file produced here is a dict with all state needed to resume training
    exactly.  It is separate from ``model_best.pth`` / ``model.pth``, which
    contain only ``model.state_dict()`` for lightweight evaluation loading.

    Args:
        path:             Destination file path.
        model:            Model being trained.
        optimizer:        Adam optimiser.
        lr_scheduler:     MultiStepLR scheduler.
        epoch:            The epoch that just completed (0-based).
        best_valid_loss:  Best validation loss seen so far.
    """
    torch.save(
        {
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  lr_scheduler.state_dict(),
            "best_valid_loss":  best_valid_loss,
        },
        path,
    )


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername="",
    save_every=10,
    resume_checkpoint=None,
):
    """Train the forecasting model, validate periodically, and save checkpoints.

    Checkpoint files written to ``foldername``:
      - ``checkpoint_epochN.pth`` — full resumable checkpoint every ``save_every``
        epochs (model + optimiser + scheduler + epoch counter).
      - ``checkpoint_latest.pth`` — always overwritten with the most recent full
        checkpoint (convenient target for ``--resume``).
      - ``model_best.pth`` — model weights only, saved whenever val loss improves.
      - ``model.pth``      — model weights only, saved at the end of training.

    Validation uses ``is_train=1`` (random diffusion timestep per batch) so it
    runs at the same speed as one training epoch.

    Args:
        model:                RATD forecasting model to optimize.
        config:               Training configuration dict (keys: lr, epochs,
                              itr_per_epoch).
        train_loader:         DataLoader for training batches.
        valid_loader:         Optional DataLoader for validation / best-model
                              selection (e.g. the test loader).
        valid_epoch_interval: Run validation every this many epochs (default 10).
        foldername:           Directory for checkpoint output.  Empty string
                              disables saving.
        save_every:           Save a full resumable checkpoint every this many
                              epochs (0 = only save at end and on improvement).
        resume_checkpoint:    Path to a ``checkpoint_*.pth`` file produced by a
                              previous run.  When supplied, model / optimiser /
                              scheduler states are restored and training resumes
                              from the next epoch.

    Returns:
        None: The function trains the model in-place.
    """

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    # MultiStepLR: drop LR by 10× at 75 % and 90 % of total epochs.
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    start_epoch     = 0
    best_valid_loss = float("inf")

    # ---- Resume from checkpoint -----------------------------------------
    if resume_checkpoint is not None:
        device = next(model.parameters()).device
        ckpt   = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch     = ckpt["epoch"] + 1          # resume from the *next* epoch
        best_valid_loss = ckpt.get("best_valid_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}  "
              f"(best_val={best_valid_loss:.6f}  →  continuing from epoch {start_epoch})")

    # ---- Main training loop ---------------------------------------------
    for epoch_no in range(start_epoch, config["epochs"]):
        # ---- Training ------------------------------------------------
        avg_loss = 0.0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0,
                  desc=f"Epoch {epoch_no}") as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_loss": avg_loss / batch_no,
                        "epoch":    epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

        lr_scheduler.step()
        print(f"Epoch {epoch_no:4d} | train_loss={avg_loss / batch_no:.6f} "
              f"| lr={lr_scheduler.get_last_lr()[0]:.2e}")

        # ---- Full resumable checkpoint (periodic + latest) ----------
        if foldername and save_every > 0 and (epoch_no + 1) % save_every == 0:
            periodic_path = os.path.join(foldername, f"checkpoint_epoch{epoch_no + 1}.pth")
            _save_full_checkpoint(periodic_path, model, optimizer, lr_scheduler,
                                  epoch_no, best_valid_loss)
            latest_path = os.path.join(foldername, "checkpoint_latest.pth")
            _save_full_checkpoint(latest_path, model, optimizer, lr_scheduler,
                                  epoch_no, best_valid_loss)
            print(f"  -> Checkpoint saved: {periodic_path}")

        # ---- Validation (best-model selection) ----------------------
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0.0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0,
                          desc="  Val") as vit:
                    for vbatch_no, valid_batch in enumerate(vit, start=1):
                        vloss = model(valid_batch, is_train=1)
                        avg_loss_valid += vloss.item()
                        vit.set_postfix(
                            ordered_dict={"val_loss": avg_loss_valid / vbatch_no},
                            refresh=False,
                        )
            model.train()
            avg_loss_valid /= vbatch_no
            print(f"  -> val_loss={avg_loss_valid:.6f}")

            if avg_loss_valid < best_valid_loss:
                best_valid_loss = avg_loss_valid
                if foldername:
                    best_path = os.path.join(foldername, "model_best.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"  -> New best model (val_loss={best_valid_loss:.6f}): {best_path}")

    # ---- Final model (always saved) ---------------------------------
    if foldername:
        final_path = os.path.join(foldername, "model.pth")
        torch.save(model.state_dict(), final_path)
        # Also update latest checkpoint so it reflects the finished state
        latest_path = os.path.join(foldername, "checkpoint_latest.pth")
        _save_full_checkpoint(latest_path, model, optimizer, lr_scheduler,
                              config["epochs"] - 1, best_valid_loss)
        print(f"Training complete.  Final model : {final_path}")
        if valid_loader is not None:
            print(f"                    Best model  : "
                  f"{os.path.join(foldername, 'model_best.pth')}"
                  f"  (val_loss={best_valid_loss:.6f})")


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    """Compute quantile loss for one forecast quantile.

    Args:
        target: Ground-truth targets.
        forecast: Forecast values at quantile ``q``.
        q: Quantile level.
        eval_points: Mask indicating which entries are evaluated.

    Returns:
        float: Aggregated quantile loss value.
    """

    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    """Compute the normalization denominator for CRPS-style metrics.

    Args:
        target: Ground-truth target tensor.
        eval_points: Evaluation mask tensor.

    Returns:
        torch.Tensor: Denominator used to normalize quantile losses.
    """

    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    """Compute CRPS from sampled forecasts over all features.

    Args:
        target: Ground-truth target tensor.
        forecast: Sampled forecast tensor.
        eval_points: Evaluation mask tensor.
        mean_scaler: Mean used to undo normalization.
        scaler: Scale used to undo normalization.

    Returns:
        float: Mean CRPS across the configured quantile grid.
    """

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):
    """Compute CRPS after summing forecasts across features.

    Args:
        target: Ground-truth target tensor.
        forecast: Sampled forecast tensor.
        eval_points: Evaluation mask tensor.
        mean_scaler: Mean used to undo normalization.
        scaler: Scale used to undo normalization.

    Returns:
        float: CRPS of the feature-summed forecast distribution.
    """

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_wape(target, forecast_median, eval_points, scaler=1, mean_scaler=0):
    """Compute Weighted Absolute Percentage Error (WAPE) from a median forecast.

    WAPE = sum(|pred - actual|) / sum(|actual|).  De-normalises using
    ``scaler`` and ``mean_scaler`` before computing, so the result is in the
    original unit space.

    Args:
        target: Ground-truth tensor ``(B, L, K)`` or ``(B, K, L)``.
        forecast_median: Point-forecast tensor, same shape as ``target``.
        eval_points: Binary mask matching ``target`` shape.
        scaler: Multiplicative de-normalization factor.
        mean_scaler: Additive de-normalization offset.

    Returns:
        float: WAPE as a fraction (multiply by 100 for percentage).
    """

    target_denorm   = target * scaler + mean_scaler
    forecast_denorm = forecast_median * scaler + mean_scaler
    numerator   = torch.abs((forecast_denorm - target_denorm) * eval_points).sum()
    denominator = torch.abs(target_denorm * eval_points).sum()
    return (numerator / denominator).item() if denominator > 0 else float("nan")


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    """Evaluate the model on the test split using sampled forecasts.

    Args:
        model: Trained forecasting model.
        test_loader: Dataloader for test batches.
        nsample: Number of stochastic forecast samples to generate per batch.
        scaler: Scale used to undo normalization.
        mean_scaler: Mean used to undo normalization.
        foldername: Directory used for output artifacts.

    Returns:
        None: Metrics and generated samples are written to disk.
    """

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0
        x=np.linspace(0,100,100)
        mse_list=np.zeros(100)
        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                # ``model.evaluate`` returns sampled trajectories together with
                # the exact masks and targets needed for downstream metrics.
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                # Point metrics use the per-timestep sample median as a robust
                # summary of the predictive distribution.
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                
                
                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler
                if batch_no<100:
                    mse_list[batch_no-1]=mse_current.sum().item()/eval_points.sum().item()
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
            # The script also stores a simple diagnostic plot for the first 100
            # batch-level MSE values, matching the original research code.
            fig,ax = plt.subplots()
            ax.plot(x,mse_list,color = '#1D2B53')
            plt.savefig('moti1.pdf')
            plt.show()
            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                # Concatenate all batch outputs so the entire predictive
                # distribution can be analyzed later without rerunning inference.
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)
