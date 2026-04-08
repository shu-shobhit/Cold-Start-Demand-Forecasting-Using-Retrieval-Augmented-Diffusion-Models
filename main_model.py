"""High-level RATD model wrappers for forecasting and diffusion training.

This module contains the task-agnostic diffusion wrapper together with the
forecasting-specific subclass that prepares batches, applies masks, and calls
the denoising network defined in ``diff_models.py``.
"""

import numpy as np
import torch
import torch.nn as nn

from diff_models import diff_RATD

class RATD_base(nn.Module):
    """Base class that manages masking, conditioning, and diffusion loss.

    Args:
        target_dim: Number of target variables in the multivariate series.
        config: Nested configuration dictionary loaded from YAML.
        device: Torch device string used for tensors and modules.

    Returns:
        None: The constructor initializes model components in-place.
    """

    def __init__(self, target_dim, config, device):
        """Initialize the shared RATD diffusion wrapper.

        Args:
            target_dim: Number of target variables in the multivariate series.
            config: Nested configuration dictionary loaded from YAML.
            device: Torch device string used for tensors and modules.

        Returns:
            None: The constructor initializes model components in-place.
        """

        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.use_reference = config["model"]["use_reference"]
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        # Side information combines explicit time encoding with a learned
        # feature identity embedding for each variable.
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_RATD(config_diff, input_dim)

        self.pred_length=config_diff["ref_size"]
        self.his_length=config_diff["h_size"]
        # Precompute the diffusion schedule once so both training and sampling
        # can index the same alpha products efficiently.
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        """Create sinusoidal time embeddings for each position.

        Args:
            pos: Tensor of observed time indices with shape ``(B, L)``.
            d_model: Embedding size for the sinusoidal encoding.

        Returns:
            torch.Tensor: Time embeddings with shape ``(B, L, d_model)``.
        """

        # This mirrors transformer-style positional encodings so the denoiser
        # can reason about relative temporal location inside the forecast window.
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        """Sample a random conditional mask from observed entries.

        Args:
            observed_mask: Binary tensor marking observed values.

        Returns:
            torch.Tensor: Randomly thinned conditioning mask.
        """

        # Only observed entries are eligible to become conditioning inputs.
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            # Each sample receives its own keep/drop ratio so training sees a
            # wide range of forecasting difficulties.
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        """Construct a conditioning mask based on historical patterns.

        Args:
            observed_mask: Binary mask for available values.
            for_pattern_mask: Optional historical pattern used for reuse.

        Returns:
            torch.Tensor: Conditioning mask derived from prior masks.
        """

        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        """Apply a fixed test-time masking pattern.

        Args:
            observed_mask: Mask marking all valid observations in the window.
            test_pattern_mask: Mask defining which positions stay visible.

        Returns:
            torch.Tensor: Conditioning mask used for forecasting evaluation.
        """

        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_tp, cond_mask):
        """Build auxiliary conditioning features for the denoiser.

        Args:
            observed_tp: Time index tensor with shape ``(B, L)``.
            cond_mask: Conditioning mask with shape ``(B, K, L)``.

        Returns:
            torch.Tensor: Side information tensor with shape ``(B, C, K, L)``.
        """

        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            # Conditional models receive the visible/missing pattern as an
            # extra channel so the denoiser knows which values are grounded.
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, reference=None
    ):
        """Average validation loss across all diffusion timesteps.

        Args:
            observed_data: Clean target tensor.
            cond_mask: Conditioning mask.
            observed_mask: Mask of valid values.
            side_info: Auxiliary conditioning tensor.
            is_train: Training flag passed through to the loss routine.
            reference: Optional retrieval tensor.

        Returns:
            torch.Tensor: Mean validation loss across all diffusion steps.
        """

        if self.use_reference == False:
            reference=None
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t, reference=reference
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, reference, set_t=-1
    ):
        """Compute DDPM-style noise prediction loss for one timestep.

        Args:
            observed_data: Clean target tensor with shape ``(B, K, L)``.
            cond_mask: Conditioning mask indicating visible values.
            observed_mask: Mask of valid values in the batch.
            side_info: Auxiliary conditioning features.
            is_train: ``1`` during training and ``0`` during validation.
            reference: Optional retrieved reference tensor.
            set_t: Fixed timestep used during validation.

        Returns:
            torch.Tensor: Mean squared noise-prediction loss on target entries.
        """

        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)

        # Diffusion training corrupts the clean target with Gaussian noise and
        # asks the model to predict the injected noise on masked positions.
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t, reference=reference)  # (B,K,L)

        # Loss is only measured on target positions, never on values already
        # revealed through the conditioning mask.
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        """Package model inputs for conditional or unconditional denoising.

        Args:
            noisy_data: Noised version of the current sample.
            observed_data: Original clean observations.
            cond_mask: Mask defining visible conditioning entries.

        Returns:
            torch.Tensor: Input tensor shaped for the diffusion backbone.
        """

        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            # Conditional mode splits the input into observed context and
            # corrupted target regions, following the CSDI design.
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        """Generate multiple reverse-diffusion samples for forecasting.

        Args:
            observed_data: Clean observations with shape ``(B, K, L)``.
            cond_mask: Conditioning mask defining visible values.
            side_info: Side-information tensor for the denoiser.
            n_samples: Number of stochastic forecast samples to draw.

        Returns:
            torch.Tensor: Sampled forecasts with shape ``(B, n_samples, K, L)``.
        """

        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # Unconditional mode needs a separately noised conditioning history
            # at every timestep because the observed part is diffused too.
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            # Reverse diffusion starts from pure Gaussian noise.
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                # The denoiser predicts the current noise component so the
                # sample can be stepped toward a cleaner state.
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        """Compute training or validation loss for a batch.

        Args:
            batch: Mini-batch dictionary produced by the dataset.
            is_train: Training flag controlling mask strategy and timestep use.

        Returns:
            torch.Tensor: Scalar loss value for optimization or validation.
        """

        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        """Generate forecast samples and evaluation masks for one batch.

        Args:
            batch: Mini-batch dictionary produced by the dataset.
            n_samples: Number of stochastic forecast samples to draw.

        Returns:
            tuple: Generated samples and tensors needed for metric computation.
        """

        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class RATD_Forecasting(RATD_base):
    """Forecasting-specific RATD wrapper used by ``exe_forecasting.py``.

    Args:
        config: Nested configuration dictionary loaded from YAML.
        device: Torch device string used for tensors and modules.
        target_dim: Number of variables in the forecasting target.

    Returns:
        None: The constructor initializes the forecasting wrapper in-place.
    """

    def __init__(self, config, device, target_dim):
        """Initialize the forecasting-specific RATD wrapper.

        Args:
            config: Nested configuration dictionary loaded from YAML.
            device: Torch device string used for tensors and modules.
            target_dim: Number of variables in the forecasting target.

        Returns:
            None: The constructor initializes the forecasting wrapper in-place.
        """

        super(RATD_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]
        self.use_reference = config["model"]["use_reference"]

    def process_data(self, batch):
        """Convert a raw dataset batch into model-ready tensors.

        Args:
            batch: Dictionary emitted by the forecasting dataset.

        Returns:
            tuple: Permuted tensors, masks, and optional reference data.
        """

        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        if self.use_reference:
            # References are stored as ``(B, 3 * pred_len, K)`` and are
            # transposed here to the ``(B, K, 3 * pred_len)`` convention used
            # inside the denoising network.
            reference = batch["reference"].to(self.device).float()
            reference = reference.permute(0, 2, 1)
        else:
            reference = None
        # The model operates on ``(B, K, L)`` tensors, so sequence-first arrays
        # from the dataset are transposed before use.
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            reference, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        """Randomly sample a subset of features from each batch item.

        Args:
            observed_data: Full observed tensor.
            observed_mask: Full observation mask.
            feature_id: Feature identifier tensor.
            gt_mask: Ground-truth conditioning mask.

        Returns:
            tuple: Feature-subsampled data, masks, ids, and ground-truth mask.
        """

        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask):
        """Build forecasting side information using the active target dimension.

        Args:
            observed_tp: Time index tensor with shape ``(B, L)``.
            cond_mask: Conditioning mask with shape ``(B, K, L)``.

        Returns:
            torch.Tensor: Side information tensor with shape ``(B, C, K, L)``.
        """

        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        """Compute the forecasting loss for one batch.

        Args:
            batch: Dataset batch dictionary.
            is_train: Training flag controlling mask selection.

        Returns:
            torch.Tensor: Scalar loss value.
        """

        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            reference, 
        ) = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            # Forecasting uses the explicit future mask supplied by the dataset
            # rather than random missingness during training.
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, reference=reference)

    def evaluate(self, batch, n_samples):
        """Generate forecasts and masks for metric evaluation.

        Args:
            batch: Dataset batch dictionary.
            n_samples: Number of forecast samples to draw.

        Returns:
            tuple: Generated samples and tensors needed for evaluation.
        """

        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            # Only the held-out forecast horizon contributes to evaluation.
            target_mask = observed_mask * (1-gt_mask)
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
