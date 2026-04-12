"""dataset_visuelle2.py

PyTorch Dataset for Visuelle 2.0 store-level sales forecasting.

Each sample corresponds to one (product, store) pair.  The model receives:
  - 12-week time series with 2 channels: [sales, discount]
  - A conditioning mask that reveals the first n_obs weeks (0–4)
  - 3 retrieved reference trajectories (product-level mean, k=3)
  - A 513-dim product attribute embedding (CLIP image+text + price)

Data layout expected by the RATD model (matches Dataset_Electricity):
  observed_data : (12, 2)   — L×K, transposed to (K,L) inside the model
  observed_mask : (12, 2)   — all ones (no structural missing values)
  gt_mask       : (12, 2)   — first n_obs rows=1, rest=0  (conditioning mask)
  timepoints    : (12,)     — week indices [0..11]
  feature_id    : (2,)      — channel indices [0, 1]
  reference     : (36, 2)   — k*pred_len rows × K cols, transposed in model
  product_emb   : (513,)    — NEW: CLIP embedding used for attribute conditioning
"""

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset_Visuelle2(Dataset):
    """Store-level fashion sales forecasting dataset for Visuelle 2.0.

    Samples are individual (product, store) pairs.  Each sample provides a
    12-week sales+discount time series together with retrieved reference
    trajectories from the 3 most attribute-similar training products.

    Masking strategy (unified across all observation settings):
      - n_obs=0  → pure cold-start: no sales history visible
      - n_obs=1–4 → few-shot: first n_obs weeks visible
    During training, n_obs is drawn uniformly from ``n_obs_range`` per sample.
    During evaluation, n_obs is fixed by the caller.

    Args:
        data_root:     Path to the BTP_2 directory (contains visuelle2/).
        processed_dir: Path to visuelle2_processed/ (precomputed .pt files).
        flag:          'train' or 'test'.
        n_obs:         Fixed observation count for evaluation.
                       None = random sampling during training.
        n_obs_range:   (min, max) inclusive range for random n_obs.
                       Default (0, 4) covers pure cold-start through 4-week.
        scale:         If True, apply per-channel StandardScaler fitted on train.
    """

    SALES_COLS   = [str(i) for i in range(12)]
    DISC_COLS    = [f"d_{i}" for i in range(12)]
    PRED_LEN     = 12     # full season horizon
    K_CHANNELS   = 2      # sales + discount
    K_REFS       = 3      # number of retrieved references (k)

    def __init__(
        self,
        data_root:     str  = "/home/shu_sho_bhit/BTP_2",
        processed_dir: str  = "/home/shu_sho_bhit/BTP_2/visuelle2_processed",
        flag:          str  = "train",
        n_obs:         int  = None,
        n_obs_range:   tuple = (0, 4),
        scale:         bool = True,
    ):
        assert flag in ("train", "test"), f"flag must be 'train' or 'test', got {flag}"
        self.flag        = flag
        self.n_obs       = n_obs
        self.n_obs_range = n_obs_range
        self.scale       = scale
        self.data_root   = Path(data_root)
        self.proc_dir    = Path(processed_dir)

        self._load_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Loads and merges all data sources; fits/applies the scaler."""

        vis_dir = self.data_root / "visuelle2"

        # ---- 1. Core time-series (sales + discount) -------------------
        train_df = pd.read_csv(vis_dir / "stfore_train.csv")
        test_df  = pd.read_csv(vis_dir / "stfore_test.csv")

        disc_df  = pd.read_csv(vis_dir / "price_discount_series.csv")
        disc_df  = disc_df.rename(columns={str(i): f"d_{i}" for i in range(12)})

        merge_cols = ["external_code", "retail"] + self.DISC_COLS
        train_df = train_df.merge(disc_df[merge_cols],
                                  on=["external_code", "retail"], how="left")
        test_df  = test_df.merge( disc_df[merge_cols],
                                  on=["external_code", "retail"], how="left")

        # Fill any missing discount values with 0 (no discount applied)
        train_df[self.DISC_COLS] = train_df[self.DISC_COLS].fillna(0.0)
        test_df[self.DISC_COLS]  = test_df[self.DISC_COLS].fillna(0.0)

        # ---- 2. Per-channel StandardScaler (fit on train only) --------
        scaler_path = self.proc_dir / "scaler.pkl"

        if self.scale:
            # Stack (N_rows * 12, 2) to fit the scaler across all weeks + rows
            train_sales = train_df[self.SALES_COLS].values.astype(np.float32)  # (N,12)
            train_disc  = train_df[self.DISC_COLS].values.astype(np.float32)   # (N,12)
            # Reshape to (N*12, 2) for StandardScaler
            train_flat = np.stack(
                [train_sales.reshape(-1), train_disc.reshape(-1)], axis=1
            )                                                                   # (N*12, 2)

            if scaler_path.exists() and self.flag == "test":
                # Load the train-fitted scaler for test-set normalisation
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            else:
                self.scaler = StandardScaler()
                self.scaler.fit(train_flat)
                with open(scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)

        # Apply scaler and store per-row tensors
        df = train_df if self.flag == "train" else test_df
        self.data = self._build_tensor(df)       # (N, 12, 2)
        self.codes  = df["external_code"].values.astype(int)
        self.retails = df["retail"].values.astype(int)

        # ---- 3. Precomputed product embeddings ------------------------
        self.product_emb: dict = torch.load(
            self.proc_dir / "product_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )

        # ---- 4. Precomputed reference trajectories --------------------
        ref_file = (
            self.proc_dir / "train_references.pt"
            if self.flag == "train"
            else self.proc_dir / "test_references.pt"
        )
        self.references: dict = torch.load(
            ref_file, map_location="cpu", weights_only=True
        )

        print(
            f"  [Dataset_Visuelle2 | {self.flag}] "
            f"{len(self.data)} samples | "
            f"{len(self.product_emb)} unique products"
        )

    def _build_tensor(self, df: pd.DataFrame) -> np.ndarray:
        """Stack sales and discount into (N, 12, 2) normalised array.

        Args:
            df: DataFrame with SALES_COLS and DISC_COLS already filled.

        Returns:
            numpy.ndarray of shape (N, 12, 2), dtype float32.
        """
        sales = df[self.SALES_COLS].values.astype(np.float32)   # (N, 12)
        disc  = df[self.DISC_COLS].values.astype(np.float32)    # (N, 12)
        raw   = np.stack([sales, disc], axis=2)                  # (N, 12, 2)

        if self.scale:
            N = raw.shape[0]
            flat = raw.reshape(N * 12, 2)                        # (N*12, 2)
            flat = self.scaler.transform(flat).astype(np.float32)
            raw  = flat.reshape(N, 12, 2)

        return raw

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """Return one (product, store) sample.

        Args:
            index: Row index into the split DataFrame.

        Returns:
            dict with keys:
              observed_data : Tensor (12, 2)
              observed_mask : Tensor (12, 2)  — all ones
              gt_mask       : Tensor (12, 2)  — first n_obs rows = 1
              timepoints    : Tensor (12,)
              feature_id    : Tensor (2,)
              reference     : Tensor (36, 2)  — (k*pred_len, K), transposed in model
              product_emb   : Tensor (513,)
              external_code : int
              retail        : int
        """
        observed_data = torch.from_numpy(self.data[index])     # (12, 2)

        # All time steps are structurally observed (no missing sensors)
        observed_mask = torch.ones(self.PRED_LEN, self.K_CHANNELS)

        # Conditioning mask: first n_obs weeks revealed, rest masked
        n_obs = (
            self.n_obs
            if self.n_obs is not None
            else random.randint(*self.n_obs_range)
        )
        gt_mask = torch.zeros(self.PRED_LEN, self.K_CHANNELS)
        if n_obs > 0:
            gt_mask[:n_obs, :] = 1.0

        timepoints = torch.arange(self.PRED_LEN, dtype=torch.float32)
        feature_id = torch.arange(self.K_CHANNELS, dtype=torch.float32)

        code = int(self.codes[index])

        # Reference tensor: (k*pred_len, K) = (36, 2)
        # Stored as (K, k*pred_len) in the dict — model transposes to (K, k*pred_len)
        # To match Dataset_Electricity convention (L_ref, K), we transpose here.
        ref_kl = self.references[code]                          # (2, 36)
        reference = ref_kl.T                                    # (36, 2) → (L_ref, K)

        product_emb = self.product_emb[code]                    # (513,)

        return {
            "observed_data": observed_data,    # (12, 2)
            "observed_mask": observed_mask,    # (12, 2)
            "gt_mask":       gt_mask,          # (12, 2)
            "timepoints":    timepoints,       # (12,)
            "feature_id":    feature_id,       # (2,)
            "reference":     reference,        # (36, 2)
            "product_emb":   product_emb,      # (513,)
            "external_code": code,
            "retail":        int(self.retails[index]),
        }

    # ------------------------------------------------------------------
    # Inverse transform (evaluation only — sales channel)
    # ------------------------------------------------------------------

    def inverse_transform_sales(self, data: np.ndarray) -> np.ndarray:
        """Inverse-normalise the sales channel (channel 0) only.

        At evaluation, predictions are un-scaled back to the normalised [0,1]
        range used in stfore_*.csv.  Multiply by 53.0 afterward to get
        approximate unit counts.

        Args:
            data: Array of shape (..., 12) containing normalised sales values.

        Returns:
            Array of the same shape, de-standardised.
        """
        flat = data.reshape(-1, 1)
        # Build (N, 2) with a dummy discount column to match scaler input shape
        dummy = np.zeros_like(flat)
        combined = np.concatenate([flat, dummy], axis=1)
        recovered = self.scaler.inverse_transform(combined)
        return recovered[:, 0].reshape(data.shape)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root:     str   = "/home/shu_sho_bhit/BTP_2",
    processed_dir: str   = "/home/shu_sho_bhit/BTP_2/visuelle2_processed",
    batch_size:    int   = 16,
    n_obs_train:   tuple = (0, 4),
    n_obs_eval:    int   = 2,
    num_workers:   int   = 4,
) -> tuple:
    """Create train and test DataLoaders for Visuelle 2.0.

    Args:
        data_root:     Path to BTP_2/.
        processed_dir: Path to visuelle2_processed/.
        batch_size:    Mini-batch size.
        n_obs_train:   (min, max) n_obs range sampled during training.
        n_obs_eval:    Fixed n_obs used for test evaluation.
        num_workers:   DataLoader worker processes.

    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    train_dataset = Dataset_Visuelle2(
        data_root=data_root,
        processed_dir=processed_dir,
        flag="train",
        n_obs=None,
        n_obs_range=n_obs_train,
    )
    test_dataset = Dataset_Visuelle2(
        data_root=data_root,
        processed_dir=processed_dir,
        flag="test",
        n_obs=n_obs_eval,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset, test_dataset
