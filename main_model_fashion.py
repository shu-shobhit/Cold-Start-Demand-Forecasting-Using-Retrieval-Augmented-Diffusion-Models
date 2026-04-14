"""main_model_fashion.py

RATD_Fashion: cold-start forecasting wrapper for the Visuelle 2.0 dataset.

Extends ``RATD_base`` with attribute conditioning via a single path:
  - Path (b)  RMA key/value injection: 513-dim product embedding concatenated
              into cross-attention K/V inside every RMA layer.

The side_info tensor is unchanged from the base RATD (time + feature embeddings
+ conditioning mask).  The diffmodel is rebuilt after super().__init__() only
to inject ``attr_dim=513`` for path (b).

Batch contract (from Dataset_Visuelle2):
  observed_data : (B, 12, 2) — L×K, transposed to (B, K, L) here
  observed_mask : (B, 12, 2) — all ones
  gt_mask       : (B, 12, 2) — first n_obs rows=1, rest=0
  timepoints    : (B, 12)
  reference     : (B, 36, 2) — (L_ref, K), transposed to (B, K, 3*pred_len)
  product_emb   : (B, 513)
"""

import numpy as np
import torch
import torch.nn as nn

from diff_models import diff_RATD
from main_model import RATD_base


class RATD_Fashion(RATD_base):
    """Cold-start forecasting wrapper that extends RATD with attribute conditioning.

    The 513-dim CLIP+price product embedding is injected via cross-attention
    inside every RMA layer (path b).  The denoiser can selectively attend to
    relevant parts of the embedding at each layer rather than receiving a
    globally broadcast projection.

    Args:
        config: Nested configuration dict loaded from a YAML file.
        device: Torch device string, e.g. ``'cuda'`` or ``'cpu'``.

    Returns:
        None: The constructor initializes the model in-place.
    """

    # Dimensionality of the fused product embedding produced by compute_embeddings.py
    ATTR_DIM: int = 513

    def __init__(self, config: dict, device: str, target_dim: int = 2):
        """Initialize the fashion cold-start forecasting model.

        Args:
            config:     Nested configuration dict loaded from a YAML file.
            device:     Torch device string.
            target_dim: Number of feature channels K.  Default 2 (sales +
                        discount) for Visuelle 2.  Pass 1 for H&M (sales only).

        Returns:
            None: The constructor initializes the model in-place.
        """

        super().__init__(target_dim, config, device)

        # ------------------------------------------------------------------
        # Rebuild diffmodel with attr_dim set for path (b) RMA injection.
        #
        # RATD_base.__init__ already created self.diffmodel without attr_dim.
        # We discard it and build a new one that knows about:
        #   side_dim = emb_total_dim   (unchanged, no path-a channels)
        #   attr_dim = from config, default 513 (path b inside RMA cross-attention)
        # ------------------------------------------------------------------
        input_dim   = 1 if self.is_unconditional else 2
        attr_dim    = config["model"].get("attr_dim", self.ATTR_DIM)
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["attr_dim"] = attr_dim
        self.diffmodel = diff_RATD(config_diff, input_dim)

        # Per-batch tensors shared between forward, calc_loss, and impute.
        # Always initialized so no AttributeError if called in unexpected order.
        self._attr_emb:  torch.Tensor | None = None
        self._reference: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def process_data(self, batch: dict) -> tuple:
        """Convert a raw ``Dataset_Visuelle2`` batch into model-ready tensors.

        Args:
            batch: Dictionary emitted by ``Dataset_Visuelle2.__getitem__``.

        Returns:
            tuple of 8 tensors:
              (observed_data, observed_mask, observed_tp, gt_mask,
               for_pattern_mask, cut_length, reference, product_emb)
            All data tensors are on ``self.device`` in ``float32``.
        """

        observed_data = batch["observed_data"].to(self.device).float()  # (B,12,2)
        observed_mask = batch["observed_mask"].to(self.device).float()  # (B,12,2)
        observed_tp   = batch["timepoints"].to(self.device).float()     # (B,12)
        gt_mask       = batch["gt_mask"].to(self.device).float()        # (B,12,2)
        product_emb   = batch["product_emb"].to(self.device).float()    # (B,513)

        if self.use_reference:
            # Dataset stores reference as (B, L_ref, K); transpose to (B, K, L_ref)
            # so the denoiser receives (B, K, 3*pred_len).
            reference = batch["reference"].to(self.device).float()      # (B,36,2)
            reference = reference.permute(0, 2, 1)                      # (B,2,36)
        else:
            reference = None

        # Model convention: (B, K, L) rather than (B, L, K)
        observed_data = observed_data.permute(0, 2, 1)    # (B,2,12)
        observed_mask = observed_mask.permute(0, 2, 1)    # (B,2,12)
        gt_mask       = gt_mask.permute(0, 2, 1)          # (B,2,12)

        # cut_length is used by the base evaluate() to prevent double counting;
        # for cold-start forecasting there is no overlap so it stays zero.
        cut_length       = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            reference,
            product_emb,
        )

    # ------------------------------------------------------------------
    # Side information (standard RATD, no path-a modification)
    # ------------------------------------------------------------------

    def get_side_info(
        self,
        observed_tp: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build standard RATD auxiliary conditioning features.

        Constructs time + feature embeddings and the conditioning mask.
        Product attributes are not injected here; they enter the model
        through RMA cross-attention (path b) via ``self._attr_emb``.

        Args:
            observed_tp: Time index tensor ``(B, L)``.
            cond_mask: Conditioning mask ``(B, K, L)``.

        Returns:
            torch.Tensor: Side information ``(B, emb_total_dim [+1], K, L)``.
        """

        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K, emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,emb_total)
        side_info = side_info.permute(0, 3, 2, 1)                   # (B,emb_total,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)                       # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)     # (B,emb_total+1,K,L)

        return side_info

    # ------------------------------------------------------------------
    # Loss (override to pass attr_emb to diffmodel for path-b)
    # ------------------------------------------------------------------

    def calc_loss(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        side_info: torch.Tensor,
        is_train: int,
        reference: torch.Tensor | None,
        set_t: int = -1,
    ) -> torch.Tensor:
        """DDPM noise-prediction loss with path-b attribute conditioning.

        Identical to ``RATD_base.calc_loss`` except the diffmodel call receives
        ``attr_emb=self._attr_emb`` to activate RMA cross-attention injection.

        Args:
            observed_data: Clean target ``(B, K, L)``.
            cond_mask: Conditioning mask.
            observed_mask: Valid-value mask.
            side_info: Auxiliary features from ``get_side_info``.
            is_train: ``1`` for training, ``0`` for validation.
            reference: Retrieved reference tensor or ``None``.
            set_t: Fixed diffusion timestep for validation (ignored in training).

        Returns:
            torch.Tensor: Scalar mean squared noise-prediction loss.
        """

        B, K, L = observed_data.shape
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)

        noise      = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(
            total_input, side_info, t,
            reference=reference,
            attr_emb=self._attr_emb,
        )  # (B, K, L)

        target_mask = observed_mask - cond_mask
        residual    = (noise - predicted) * target_mask
        num_eval    = target_mask.sum()
        loss        = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    # ------------------------------------------------------------------
    # Sampling (override to pass reference + attr_emb at inference)
    # ------------------------------------------------------------------

    def impute(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        side_info: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Draw reverse-diffusion forecast samples with path-b attribute conditioning.

        Args:
            observed_data: Clean observations ``(B, K, L)``.
            cond_mask: Conditioning mask.
            side_info: Side information tensor from ``get_side_info``.
            n_samples: Number of independent forecast samples to draw.

        Returns:
            torch.Tensor: Sampled forecasts ``(B, n_samples, K, L)``.
        """

        B, K, L    = observed_data.shape
        imputed    = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs     = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input   = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

                predicted = self.diffmodel(
                    diff_input,
                    side_info,
                    torch.tensor([t]).to(self.device),
                    reference=self._reference,
                    attr_emb=self._attr_emb,
                )

                coeff1         = 1 / self.alpha_hat[t] ** 0.5
                coeff2         = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise  = torch.randn_like(current_sample)
                    sigma  = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample = current_sample + sigma * noise

            imputed[:, i] = current_sample.detach()

        return imputed

    # ------------------------------------------------------------------
    # Training / validation forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict, is_train: int = 1) -> torch.Tensor:
        """Compute the cold-start forecasting loss for one mini-batch.

        Args:
            batch: Mini-batch dictionary from ``Dataset_Visuelle2``.
            is_train: ``1`` for training, ``0`` for validation.

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
            product_emb,
        ) = self.process_data(batch)

        # Store for downstream calc_loss and impute calls this forward pass
        self._attr_emb  = product_emb
        self._reference = reference

        cond_mask = gt_mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(
            observed_data, cond_mask, observed_mask, side_info, is_train,
            reference=reference,
        )

    # ------------------------------------------------------------------
    # Evaluation (metric computation at test time)
    # ------------------------------------------------------------------

    def evaluate(self, batch: dict, n_samples: int) -> tuple:
        """Generate forecast samples and evaluation masks for one batch.

        Args:
            batch: Mini-batch dictionary from ``Dataset_Visuelle2``.
            n_samples: Number of forecast samples to draw.

        Returns:
            tuple:
              samples       (B, n_samples, K, L) — imputed forecasts
              observed_data (B, K, L)            — clean ground truth
              target_mask   (B, K, L)            — held-out forecast positions
              observed_mask (B, K, L)            — all valid positions
              observed_tp   (B, L)               — time indices
        """

        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            reference,
            product_emb,
        ) = self.process_data(batch)

        with torch.no_grad():
            self._attr_emb  = product_emb
            self._reference = reference

            cond_mask   = gt_mask
            target_mask = observed_mask * (1 - gt_mask)   # held-out positions
            side_info   = self.get_side_info(observed_tp, cond_mask)
            samples     = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
