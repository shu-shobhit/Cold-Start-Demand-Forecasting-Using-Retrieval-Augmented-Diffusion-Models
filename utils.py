"""Training and evaluation utilities for RATD forecasting experiments.

The helpers in this module implement the training loop, checkpoint saving,
forecast metric computation, and the batch-wise evaluation pipeline used by
the forecasting entrypoint.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    """Train the forecasting model and save checkpoints.

    Args:
        model: RATD forecasting model to optimize.
        config: Training configuration dictionary.
        train_loader: Dataloader for training batches.
        valid_loader: Optional dataloader for validation batches.
        valid_epoch_interval: Validation frequency in epochs.
        foldername: Directory used for checkpoint output.

    Returns:
        None: The function trains the model and writes checkpoints.
    """

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                # The model wrapper returns the scalar diffusion noise-prediction
                # loss for the current batch.
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                # Some experiments cap the number of iterations per epoch, so
                # the loop exits early once the configured limit is reached.
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            #model.eval()
            #avg_loss_valid = 0
            #with torch.no_grad():
                #with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    #for batch_no, valid_batch in enumerate(it, start=1):
                        #loss = model(valid_batch, is_train=0)
                        #avg_loss_valid += loss.item()
                        #it.set_postfix(
                            #ordered_dict={
                                #"valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                #"epoch": epoch_no,
                            #},
                            #refresh=False,
                        #)
            #if best_valid_loss > avg_loss_valid:
                #if foldername != "":
            # The current research snapshot saves the latest model each
            # validation interval even though the best-loss logic is commented.
            torch.save(model.state_dict(), output_path)
                #best_valid_loss = avg_loss_valid
                #print(
                    #"\n best loss is updated to ",
                    #avg_loss_valid / batch_no,
                    #"at",
                    #epoch_no,
                #)

    if foldername != "":
        torch.save(model.state_dict(), output_path)


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
