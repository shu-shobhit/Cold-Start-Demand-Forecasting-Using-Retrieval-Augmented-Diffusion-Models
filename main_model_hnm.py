"""main_model_hnm.py

RATD_HnM: cold-start forecasting wrapper for the H&M dataset.

Thin subclass of RATD_Fashion specialised for K=1 (sales-only channel).
All conditioning logic (path-b RMA attribute injection, get_side_info,
calc_loss, impute) is inherited unchanged from RATD_Fashion.

The only differences from RATD_Fashion are:
  - target_dim = 1  (sales only; no discount channel)
  - process_data handles the H&M batch format (article_id key instead of
    external_code / retail; observed tensors are (B, 12, 1))

Batch contract (from Dataset_HnM):
  observed_data : (B, 12, 1) — weekly log-normalised sales
  observed_mask : (B, 12, 1) — all ones
  gt_mask       : (B, 12, 1) — first n_obs rows=1, rest=0
  timepoints    : (B, 12)
  reference     : (B, 36, 1) — k*pred_len rows × K=1 col
  product_emb   : (B, 513)
"""

import torch

from diff_models import diff_RATD
from main_model import RATD_base
from main_model_fashion import RATD_Fashion


class RATD_HnM(RATD_Fashion):
    """K=1 cold-start wrapper for the H&M dataset.

    Inherits all conditioning paths and the diffusion loss / sampling loop
    from RATD_Fashion.  Only ``__init__`` and ``process_data`` are
    overridden to account for the single sales channel and the H&M batch
    key names.

    Args:
        config: YAML config dict.  ``model.attr_dim`` controls the
            attribute embedding dimension (default 513).
        device: Torch device string.
    """

    def __init__(self, config: dict, device: str):
        # Call RATD_base directly with target_dim=1, bypassing
        # RATD_Fashion.__init__ which would hardcode target_dim=2.
        RATD_base.__init__(self, 1, config, device)

        input_dim   = 1 if self.is_unconditional else 2
        attr_dim    = config["model"].get("attr_dim", self.ATTR_DIM)
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["attr_dim"] = attr_dim
        self.diffmodel = diff_RATD(config_diff, input_dim)

        self._attr_emb:  torch.Tensor | None = None
        self._reference: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Data preparation (override for K=1 and H&M key names)
    # ------------------------------------------------------------------

    def process_data(self, batch: dict) -> tuple:
        """Convert a raw Dataset_HnM batch into model-ready tensors.

        Args:
            batch: Dictionary emitted by Dataset_HnM.__getitem__.

        Returns:
            tuple of 8 tensors matching the RATD_Fashion contract:
              (observed_data, observed_mask, observed_tp, gt_mask,
               for_pattern_mask, cut_length, reference, product_emb)
        """

        observed_data = batch["observed_data"].to(self.device).float()  # (B,12,1)
        observed_mask = batch["observed_mask"].to(self.device).float()  # (B,12,1)
        observed_tp   = batch["timepoints"].to(self.device).float()     # (B,12)
        gt_mask       = batch["gt_mask"].to(self.device).float()        # (B,12,1)
        product_emb   = batch["product_emb"].to(self.device).float()    # (B,513)

        if self.use_reference:
            reference = batch["reference"].to(self.device).float()      # (B,36,1)
            reference = reference.permute(0, 2, 1)                      # (B,1,36)
        else:
            reference = None

        # Model convention: (B, K, L)
        observed_data = observed_data.permute(0, 2, 1)    # (B,1,12)
        observed_mask = observed_mask.permute(0, 2, 1)    # (B,1,12)
        gt_mask       = gt_mask.permute(0, 2, 1)          # (B,1,12)

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
