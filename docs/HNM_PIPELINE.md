# H&M Cold-Start Forecasting Pipeline

Complete guide for preprocessing, training, and evaluating `RATD_HnM` on the
H&M Personalized Fashion Recommendations dataset.

---

## Overview

The H&M pipeline mirrors the Visuelle 2.0 pipeline with three differences:

| Property         | Visuelle 2.0           | H&M                        |
|------------------|------------------------|----------------------------|
| Channels (K)     | 2 (sales + discount)   | 1 (sales only)             |
| Granularity      | store × product        | product only               |
| Splits           | train / test           | train / val / test         |
| Sales scale      | already [0,1] norm     | log1p + StandardScaler     |
| Product embedding| 513-dim (CLIP img+text)| 513-dim (CLIP text + price)|

---

## Step 1: Preprocess on Kaggle

The H&M dataset is 3.8 GB and the CLIP encoding step needs a GPU. Run
the preprocessing notebook on Kaggle.

### 1a. Set up the Kaggle notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. Click **Add data** (top right) and add:
   - **H&M Personalized Fashion Recommendations**
     (`h-and-m-personalized-fashion-recommendations`)
3. In **Settings** (right panel):
   - Accelerator: **GPU T4 x2** (or GPU P100)
   - Internet: **On**
4. Upload `notebooks/hnm_kaggle.py` from this repo.
   - Click **File > Upload notebook** OR paste the file contents.
5. Click **Run All** (or run cells top to bottom).

> Runtime estimate: ~30-45 minutes total (CLIP encoding dominates).

### 1b. What the notebook does

| Cell | Stage | Output |
|------|-------|--------|
| 2 | Load raw CSVs | - |
| 3 | Compute 12-period sales matrices (14-day periods) | - |
| 4 | Density filter (>= 100 total units, >= 12 periods) | ~33K articles |
| 5 | Mean price per article | - |
| 6 | Chronological 70/10/20 split | train/val/test lists |
| 7 | log1p + StandardScaler | `scaler.pkl`, normalised arrays |
| 8 | CLIP text encoder (512-dim) + price (1-dim) = 513-dim | `product_embeddings.pt` |
| 9 | FAISS inner-product retrieval, k=3 | ref index dicts |
| 10 | Build (1, 36) reference tensors | `{split}_references.pt` |
| 11 | Save all files | `hnm_processed/` |
| 12 | Verification round-trip | printed checks |

### 1c. Download the output

After the notebook finishes:

1. Click the **Output** tab in the right panel.
2. Find `/kaggle/working/hnm_processed/`.
3. Click the three dots next to the folder and select **Download**.
   (It will download as a `.zip`.)
4. Extract the zip and place the folder at:
   ```
   /home/shu_sho_bhit/BTP_2/hnm_processed/
   ```

Verify the folder contains:
```
hnm_processed/
  article_ids_train.pt        (~0.7 MB)
  article_ids_val.pt          (~0.1 MB)
  article_ids_test.pt         (~0.2 MB)
  sales_train.npy             (~1.1 MB)
  sales_val.npy               (~0.2 MB)
  sales_test.npy              (~0.3 MB)
  product_embeddings.pt       (~65 MB)
  train_references.pt         (~3.3 MB)
  val_references.pt           (~0.5 MB)
  test_references.pt          (~1.0 MB)
  scaler.pkl                  (<1 KB)
```

---

## Step 2: Train

```bash
# Standard training (100 epochs, validates on val split every 10 epochs)
conda run -n ML python exe_hnm.py \
    --device cuda \
    --seed 42

# The run folder is created automatically under save/:
#   save/hnm_n2_YYYYMMDD_HHMMSS/
#   Contains: model_best.pth, model.pth, config.json, checkpoint_latest.pth
```

### Resume an interrupted run

```bash
conda run -n ML python exe_hnm.py \
    --resume save/hnm_n2_YYYYMMDD_HHMMSS \
    --device cuda
```

The `--resume` flag accepts either a folder (auto-picks `checkpoint_latest.pth`)
or a specific checkpoint file.

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n_obs_min` | 0 | Min observation weeks sampled during training |
| `--n_obs_max` | 4 | Max observation weeks sampled during training |
| `--save_every` | 10 | Save full checkpoint every N epochs |
| `--val_interval` | 10 | Validate on val split every N epochs |
| `--nsample` | 50 | Reverse-diffusion samples at eval time |

---

## Step 3: Evaluate

### Single n_obs evaluation

```bash
conda run -n ML python exe_hnm.py \
    --modelfolder save/hnm_n2_YYYYMMDD_HHMMSS \
    --n_obs_eval 2 \
    --nsample 50
```

### Sweep all n_obs (0-4)

```bash
conda run -n ML python exe_hnm.py \
    --modelfolder save/hnm_n2_YYYYMMDD_HHMMSS \
    --sweep \
    --nsample 50
```

This prints a table like:

```
n_obs     RMSE      MAE    WAPE%     CRPS   CRPS_sum
    0   0.8123   0.6201    41.2%   0.4312     0.3980
    1   0.7654   0.5892    38.1%   0.4071     0.3741
    2   0.6901   0.5203    33.4%   0.3628     0.3301
    3   0.6234   0.4801    30.8%   0.3341     0.3001
    4   0.5901   0.4502    28.2%   0.3102     0.2789
```

Results are saved to `save/hnm_.../metrics_sweep.json`.

---

## Metric definitions

| Metric | Space | Description |
|--------|-------|-------------|
| RMSE | log-normalised | Root mean squared error on standardised log1p sales |
| MAE | log-normalised | Mean absolute error on standardised log1p sales |
| WAPE | raw unit sales | `sum(|pred-actual|) / sum(|actual|)` after full inverse transform |
| CRPS | log-normalised | Probabilistic accuracy (lower = better) |
| CRPS_sum | log-normalised | CRPS of the summed-over-features forecast |

> WAPE is the most interpretable metric for the thesis since it is in raw
> sales units and directly comparable to baseline methods.

---

## File placement reference

```
BTP_2/
  hnm_dataset_orig/              Raw Kaggle files (do not modify)
    articles.csv
    transactions_train.csv
    customers.csv

  hnm_processed/                 Generated by Kaggle notebook
    article_ids_train.pt
    ...                          (see list above)

  Cold-Start-.../
    dataset_hnm.py               Dataset class (reads hnm_processed/)
    main_model_hnm.py            RATD_HnM model class
    exe_hnm.py                   Training / evaluation entrypoint
    config/hnm.yaml              Hyperparameters
    scripts/preprocess_hnm.py    Standalone preprocessing script
    notebooks/hnm_kaggle.py      Kaggle notebook source
    save/hnm_*/                  Training checkpoints (auto-created)
```

---

## Troubleshooting

**`FileNotFoundError: hnm_processed/article_ids_train.pt`**
The processed files are missing. Run the Kaggle notebook and download
the output before training.

**`KeyError: article_id ...` in dataset**
An article in the split list has no embedding or reference. This means
the preprocessing was run with a different filter threshold. Re-run
the Kaggle notebook with the same `MIN_UNITS=100` setting.

**CLIP import error on Kaggle**
Cell 0 installs `transformers` and `faiss-cpu`. If it fails, run:
```python
!pip install transformers faiss-cpu
```
in a cell before re-running.

**CUDA out of memory during CLIP encoding**
Reduce `BATCH_SIZE` in Cell 1 from 512 to 128 or 64.

**Training on CPU is slow**
Pass `--device cuda` and make sure a GPU is available. On the lab
machine, check with `nvidia-smi`. The training loop itself is fast
(K=1, smaller tensors than Visuelle2).
