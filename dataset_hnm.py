"""dataset_hnm.py

PyTorch Dataset for the H&M fashion sales forecasting task.

Each sample is one article (product).  There is no store-level split in
the public H&M Kaggle dataset, so the granularity is per-article.

Expected processed files (produced by notebooks/hnm_kaggle.py):
  hnm_processed/
    article_ids_train.pt    list[str]  — article IDs in the train split
    article_ids_val.pt      list[str]
    article_ids_test.pt     list[str]
    sales_train.npy         ndarray (N_train, 12) float32 — log1p+scaled sales
    sales_val.npy           ndarray (N_val, 12)   float32
    sales_test.npy          ndarray (N_test, 12)  float32
    product_embeddings.pt   dict[str -> Tensor(513)]
    train_references.pt     dict[str -> Tensor(1, 36)]
    val_references.pt       dict[str -> Tensor(1, 36)]
    test_references.pt      dict[str -> Tensor(1, 36)]
    scaler.pkl              sklearn StandardScaler fitted on log1p(train_sales)

Data layout expected by the RATD model (matches Dataset_Visuelle2):
  observed_data : (12, 1)  — L×K, transposed to (K,L) inside the model
  observed_mask : (12, 1)  — all ones
  gt_mask       : (12, 1)  — first n_obs rows=1, rest=0
  timepoints    : (12,)    — week indices [0..11]
  feature_id    : (1,)     — channel index [0]
  reference     : (36, 1)  — (k*pred_len, K), transposed in model
  product_emb   : (513,)   — CLIP text + price embedding
"""

import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Dataset_HnM(Dataset):
    """Article-level H&M fashion sales forecasting dataset.

    Samples are individual articles (products).  Each sample provides a
    12-period sales time series together with k=3 retrieved reference
    trajectories from the most attribute-similar training articles.

    Masking strategy (same as Dataset_Visuelle2):
      - n_obs=0  → pure cold-start
      - n_obs=1-4 → few-shot: first n_obs periods visible

    Args:
        processed_dir: Path to hnm_processed/ directory.
        flag:          'train', 'val', or 'test'.
        n_obs:         Fixed observation count for evaluation (None = random).
        n_obs_range:   (min, max) inclusive range for random n_obs.
        scale:         If True, load pre-scaled sales (recommended).
    """

    PRED_LEN   = 12
    K_CHANNELS = 1   # sales only
    K_REFS     = 3

    def __init__(
        self,
        processed_dir: str  = "/home/shu_sho_bhit/BTP_2/hnm_processed",
        flag:          str  = "train",
        n_obs:         int  = None,
        n_obs_range:   tuple = (0, 4),
        scale:         bool = True,
    ):
        assert flag in ("train", "val", "test"), \
            f"flag must be 'train', 'val', or 'test', got '{flag}'"
        self.flag        = flag
        self.n_obs       = n_obs
        self.n_obs_range = n_obs_range
        self.scale       = scale
        self.proc_dir    = Path(processed_dir)

        self._load_data()

    def _load_data(self):
        p = self.proc_dir

        # ---- 1. Article IDs and sales -----------------------------------
        self.article_ids: list[str] = torch.load(
            p / f"article_ids_{self.flag}.pt", map_location="cpu", weights_only=True
        )

        sales_file = p / f"sales_{self.flag}.npy"
        self.data: np.ndarray = np.load(sales_file).astype(np.float32)  # (N, 12)

        # ---- 2. Scaler (for inverse transform at eval time) ------------
        if self.scale:
            scaler_path = p / "scaler.pkl"
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

        # ---- 3. Product embeddings -------------------------------------
        self.product_emb: dict = torch.load(
            p / "product_embeddings.pt", map_location="cpu", weights_only=True
        )

        # ---- 4. Reference trajectories ---------------------------------
        self.references: dict = torch.load(
            p / f"{self.flag}_references.pt", map_location="cpu", weights_only=True
        )

        print(
            f"  [Dataset_HnM | {self.flag}] "
            f"{len(self.data)} samples | "
            f"{len(self.product_emb)} unique articles"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """Return one article sample.

        Returns:
            dict with keys:
              observed_data : Tensor (12, 1)
              observed_mask : Tensor (12, 1)  — all ones
              gt_mask       : Tensor (12, 1)  — first n_obs rows = 1
              timepoints    : Tensor (12,)
              feature_id    : Tensor (1,)
              reference     : Tensor (36, 1)  — (k*pred_len, K)
              product_emb   : Tensor (513,)
              article_id    : str
        """
        # Sales as (12, 1) to match (L, K) convention
        sales = torch.from_numpy(self.data[index]).unsqueeze(-1)   # (12, 1)

        observed_mask = torch.ones(self.PRED_LEN, self.K_CHANNELS)

        n_obs = (
            self.n_obs
            if self.n_obs is not None
            else random.randint(*self.n_obs_range)
        )
        gt_mask = torch.zeros(self.PRED_LEN, self.K_CHANNELS)
        if n_obs > 0:
            gt_mask[:n_obs, :] = 1.0

        timepoints = torch.arange(self.PRED_LEN, dtype=torch.float32)
        feature_id = torch.zeros(self.K_CHANNELS, dtype=torch.float32)

        art_id = self.article_ids[index]

        # Reference: stored as (1, 36) = (K, k*pred_len)
        # Return (36, 1) = (L_ref, K) — model permutes to (B, K, L_ref)
        ref_kl    = self.references[art_id]   # (1, 36)
        reference = ref_kl.T                   # (36, 1)

        product_emb = self.product_emb[art_id]  # (513,)

        return {
            "observed_data": sales,           # (12, 1)
            "observed_mask": observed_mask,   # (12, 1)
            "gt_mask":       gt_mask,         # (12, 1)
            "timepoints":    timepoints,      # (12,)
            "feature_id":    feature_id,      # (1,)
            "reference":     reference,       # (36, 1)
            "product_emb":   product_emb,     # (513,)
            "article_id":    art_id,
        }

    # ------------------------------------------------------------------
    # Inverse transform (evaluation)
    # ------------------------------------------------------------------

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Convert log1p-scaled sales back to raw unit counts.

        Args:
            data: Array of any shape containing scaled values.

        Returns:
            Raw sales counts (same shape).
        """
        if self.scaler is None:
            return data
        flat = data.reshape(-1, 1)
        scaled_inv = self.scaler.inverse_transform(flat).reshape(data.shape)
        return np.expm1(scaled_inv)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    processed_dir: str   = "/home/shu_sho_bhit/BTP_2/hnm_processed",
    batch_size:    int   = 32,
    n_obs_train:   tuple = (0, 4),
    n_obs_eval:    int   = 2,
    num_workers:   int   = 4,
) -> tuple:
    """Create train, val, and test DataLoaders for H&M.

    Args:
        processed_dir: Path to hnm_processed/.
        batch_size:    Mini-batch size.
        n_obs_train:   (min, max) n_obs range sampled during training.
        n_obs_eval:    Fixed n_obs used for val/test evaluation.
        num_workers:   DataLoader worker processes.

    Returns:
        tuple: (train_loader, val_loader, test_loader,
                train_dataset, val_dataset, test_dataset)
    """
    train_dataset = Dataset_HnM(
        processed_dir=processed_dir,
        flag="train",
        n_obs=None,
        n_obs_range=n_obs_train,
    )
    val_dataset = Dataset_HnM(
        processed_dir=processed_dir,
        flag="val",
        n_obs=n_obs_eval,
    )
    test_dataset = Dataset_HnM(
        processed_dir=processed_dir,
        flag="test",
        n_obs=n_obs_eval,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
