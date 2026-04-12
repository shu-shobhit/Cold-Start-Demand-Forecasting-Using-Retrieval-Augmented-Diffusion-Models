"""tests/smoke/test_dataset_visuelle2.py

Smoke tests for dataset_visuelle2.py (Dataset_Visuelle2 and get_dataloader).

Tests:
  - Schema check: __getitem__ returns a dict with all required keys and the
    correct tensor shapes and dtypes.
  - Masking correctness: n_obs=0 → gt_mask is all zeros; n_obs=4 → first 4
    rows are ones.
  - get_dataloader factory returns (train_loader, test_loader, train_ds, test_ds)
    and a single batch from each loader has the right keys.
  - inverse_transform_sales: round-trip test (scale → unscale).
  - Data availability check: if any required file is absent the whole test
    module is skipped with an informative message.

Expected outcome:
  If all required data files are present — all assertions pass, PASS printed.
  If data files are absent — module prints SKIP and exits with code 0
  (so CI does not count this as a failure).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np

SEP = "-" * 60

# ---------------------------------------------------------------------------
# Data-availability guard
# ---------------------------------------------------------------------------

DATA_ROOT     = "/home/shu_sho_bhit/BTP_2"
PROCESSED_DIR = "/home/shu_sho_bhit/BTP_2/visuelle2_processed"

REQUIRED_FILES = [
    os.path.join(DATA_ROOT, "visuelle2", "stfore_train.csv"),
    os.path.join(DATA_ROOT, "visuelle2", "stfore_test.csv"),
    os.path.join(DATA_ROOT, "visuelle2", "price_discount_series.csv"),
    os.path.join(PROCESSED_DIR, "product_embeddings.pt"),
    os.path.join(PROCESSED_DIR, "train_references.pt"),
    os.path.join(PROCESSED_DIR, "test_references.pt"),
]

def _check_data_available() -> bool:
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print(f"\nSKIP: {len(missing)} required data file(s) absent:")
        for m in missing:
            print(f"  {m}")
        print("Run scripts/compute_embeddings.py and scripts/compute_retrieval.py first.")
        return False
    return True


def check(condition: bool, msg: str):
    if not condition:
        raise AssertionError(f"FAIL: {msg}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_getitem_schema(ds):
    print(SEP)
    print("TEST 1: __getitem__ — required keys and shapes")

    sample = ds[0]

    REQUIRED_KEYS = {
        "observed_data": (12, 2),
        "observed_mask": (12, 2),
        "gt_mask":       (12, 2),
        "timepoints":    (12,),
        "feature_id":    (2,),
        "reference":     (36, 2),
        "product_emb":   (513,),
    }

    for key, expected_shape in REQUIRED_KEYS.items():
        check(key in sample, f"key '{key}' missing from sample")
        t = sample[key]
        check(isinstance(t, torch.Tensor), f"sample['{key}'] is not a Tensor")
        check(tuple(t.shape) == expected_shape,
              f"sample['{key}'].shape {tuple(t.shape)} != {expected_shape}")
        check(t.dtype == torch.float32,
              f"sample['{key}'].dtype {t.dtype} != float32")

    check("external_code" in sample, "missing 'external_code'")
    check("retail"        in sample, "missing 'retail'")
    print(f"  All {len(REQUIRED_KEYS)} tensor keys correct  PASS")


def test_getitem_mask_nobs_zero(ds_nobs0):
    print(SEP)
    print("TEST 2: n_obs=0 → gt_mask is all zeros (pure cold-start)")

    sample = ds_nobs0[0]
    gt = sample["gt_mask"]
    check(gt.sum().item() == 0.0, f"gt_mask sum={gt.sum().item()} != 0 for n_obs=0")
    print(f"  gt_mask sum={gt.sum().item()}  PASS")


def test_getitem_mask_nobs_four(ds_nobs4):
    print(SEP)
    print("TEST 3: n_obs=4 → first 4 rows = 1, rest = 0")

    sample = ds_nobs4[0]
    gt = sample["gt_mask"]   # (12, 2)
    check(gt[:4, :].sum().item() == 8.0,
          f"first 4 rows sum={gt[:4,:].sum().item()} != 8")
    check(gt[4:, :].sum().item() == 0.0,
          f"rows 4-11 sum={gt[4:,:].sum().item()} != 0")
    print(f"  gt_mask rows 0-3 = 1, rows 4-11 = 0  PASS")


def test_observed_mask_all_ones(ds):
    print(SEP)
    print("TEST 4: observed_mask — all ones (no structural missing values)")

    for idx in range(min(10, len(ds))):
        obs_m = ds[idx]["observed_mask"]
        check(obs_m.sum().item() == 24.0,
              f"idx={idx} observed_mask sum={obs_m.sum().item()} != 24")
    print(f"  Checked 10 samples, all observed_masks are all-ones  PASS")


def test_timepoints_range(ds):
    print(SEP)
    print("TEST 5: timepoints — range [0, 11]")

    tp = ds[0]["timepoints"]
    check(tp[0].item() == 0.0,  f"timepoints[0]={tp[0].item()} != 0")
    check(tp[-1].item() == 11.0, f"timepoints[-1]={tp[-1].item()} != 11")
    check(len(tp) == 12, f"len(timepoints)={len(tp)} != 12")
    print(f"  timepoints = {tp.tolist()}  PASS")


def test_reference_finite(ds):
    print(SEP)
    print("TEST 6: reference tensor — finite values, shape (36, 2)")

    for idx in range(min(10, len(ds))):
        ref = ds[idx]["reference"]
        check(ref.shape == (36, 2), f"idx={idx} ref.shape={ref.shape}")
        check(not torch.isnan(ref).any(), f"idx={idx} NaN in reference")
        check(not torch.isinf(ref).any(), f"idx={idx} Inf in reference")
    print(f"  Checked 10 samples, all references finite (36,2)  PASS")


def test_product_emb_shape(ds):
    print(SEP)
    print("TEST 7: product_emb — shape (513,) and finite")

    for idx in range(min(10, len(ds))):
        emb = ds[idx]["product_emb"]
        check(emb.shape == (513,), f"idx={idx} product_emb.shape={emb.shape}")
        check(not torch.isnan(emb).any(), f"idx={idx} NaN in product_emb")
    print(f"  Checked 10 samples, all product_embs are (513,)  PASS")


def test_dataloader_batch(train_loader, test_loader):
    print(SEP)
    print("TEST 8: get_dataloader — batch has correct keys")

    for name, loader in [("train", train_loader), ("test", test_loader)]:
        batch = next(iter(loader))
        for key in ("observed_data", "observed_mask", "gt_mask",
                    "timepoints", "reference", "product_emb"):
            check(key in batch, f"{name} batch missing key '{key}'")
        obs_d = batch["observed_data"]
        B = obs_d.shape[0]
        check(obs_d.shape == (B, 12, 2), f"{name} observed_data shape {obs_d.shape}")
        check(batch["reference"].shape == (B, 36, 2),
              f"{name} reference shape {batch['reference'].shape}")
        check(batch["product_emb"].shape == (B, 513),
              f"{name} product_emb shape {batch['product_emb'].shape}")
        print(f"  {name} batch OK: B={B}  PASS")


def test_inverse_transform(train_ds):
    print(SEP)
    print("TEST 9: inverse_transform_sales — round-trip preserves scale")

    sample = train_ds[0]
    sales_scaled = sample["observed_data"][:, 0].numpy()   # (12,)

    sales_original = train_ds.inverse_transform_sales(sales_scaled.reshape(1, 12))
    check(sales_original.shape == (1, 12),
          f"inverse_transform output shape {sales_original.shape}")
    # Re-normalise and compare (relative tolerance 1e-5)
    re_scaled = (sales_original.reshape(-1, 1) - train_ds.scaler.mean_[0]) / train_ds.scaler.scale_[0]
    check(np.allclose(re_scaled.reshape(12), sales_scaled, atol=1e-5),
          "round-trip scaling mismatch")
    print(f"  round-trip OK  PASS")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE: dataset_visuelle2.py")
    print("=" * 60)

    if not _check_data_available():
        sys.exit(0)   # graceful skip — not a test failure

    from dataset_visuelle2 import Dataset_Visuelle2, get_dataloader

    # Build datasets with fixed n_obs for deterministic tests
    print("\nLoading train dataset (n_obs=2) ...")
    train_ds    = Dataset_Visuelle2(DATA_ROOT, PROCESSED_DIR, flag="train", n_obs=2)
    ds_nobs0    = Dataset_Visuelle2(DATA_ROOT, PROCESSED_DIR, flag="test",  n_obs=0)
    ds_nobs4    = Dataset_Visuelle2(DATA_ROOT, PROCESSED_DIR, flag="test",  n_obs=4)

    test_getitem_schema(train_ds)
    test_getitem_mask_nobs_zero(ds_nobs0)
    test_getitem_mask_nobs_four(ds_nobs4)
    test_observed_mask_all_ones(train_ds)
    test_timepoints_range(train_ds)
    test_reference_finite(train_ds)
    test_product_emb_shape(train_ds)

    print("\nLoading dataloaders ...")
    train_loader, test_loader, _, _ = get_dataloader(
        data_root=DATA_ROOT,
        processed_dir=PROCESSED_DIR,
        batch_size=4,
        n_obs_eval=0,
        num_workers=0,    # avoid multiprocessing in smoke test
    )
    test_dataloader_batch(train_loader, test_loader)
    test_inverse_transform(train_ds)

    print(SEP)
    print("ALL DATASET_VISUELLE2 SMOKE TESTS PASSED")
    print(SEP)
