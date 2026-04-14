# BTP Project Plan: Tackling the Cold-Start Problem with Retrieval-Augmented Diffusion Models

**Working directory:** `/home/shu_sho_bhit/BTP_2/`
**Main code repo:** `Cold-Start-Demand-Forecasting-Using-Retrieval-Augmented-Diffusion-Models/`
**Original RATD reference:** `RATD/`
**Base paper:** `Papers/NeurIPS-2024-retrieval-augmented-diffusion-models-for-time-series-forecasting-Paper-Conference.pdf`
**Paper breakdown:** `RATD_paper_breakdown.md`

---

## Project Summary

We adapt RATD (Retrieval-Augmented Time series Diffusion Model, NeurIPS 2024) to tackle the **New Product Introduction (NPI) cold-start forecasting problem** in fashion retail. Standard autoregressive models require historical sales data, which is unavailable at product launch. Our approach replaces time-series-based retrieval with **product-attribute-based retrieval**: given only a new product's image, category, color, fabric, and price, we retrieve the most similar historical products and use their sales trajectories as reference guidance for a diffusion model.

We target two tasks of increasing difficulty:

- **Task A — Few-Shot Forecasting (primary):** 1–4 weeks of early sales are observed. The model forecasts the remaining weeks of the 12-week season. Maps cleanly onto RATD's existing `cond_mask` conditioning mechanism; the observed weeks become `x_H`.
- **Task B — Pure Cold-Start (harder, secondary):** Zero sales observed. Only product attributes are available. Requires injecting product attribute embeddings as a new conditioning signal into the diffusion backbone.

Datasets:

- **Visuelle 2.0** (`visuelle2/`) — Primary benchmark. Purpose-built for NPI, shop-level 12-week weekly sales.
- **H&M** (`hnm_dataset_orig/`) — Secondary validation. Larger scale (105K articles, 31.8M transactions).

---

## Open Design Decisions (to be resolved before/during implementation)

> These are flagged explicitly. When a decision is made, update this section and the relevant phase below.

| #  | Decision                                  | Options                                                                                                                                             | Status                                     |
| -- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| D1 | Feature dimensionality K for Visuelle 2.0 | (a) K=1 sales only, (b) K=2 sales+discount, (c) K=3 sales+discount+restock flag                                                                     | **Resolved: (b) K=2**                |
| D2 | Product attribute injection into model    | (a) Extend `side_info` channels — least disruptive, (b) New cross-attention input alongside references, (c) Encode as synthetic history — hacky | **Resolved: (a) + (b) both active**  |
| D3 | Image encoder for retrieval               | (a) CLIP ViT-B/32 — strong semantic alignment, (b) ResNet-50 pretrained on ImageNet, (c) Fine-tuned on fashion images                              | **Resolved: (a) CLIP ViT-B/32**      |
| D4 | Categorical tag encoding                  | (a) One-hot, (b) Learned embeddings, (c) CLIP text encoder for zero-shot generalization                                                             | **Resolved: (c) CLIP text encoder**  |
| D5 | Number of retrieved references k          | Original RATD uses k=3; try k=1,3,5                                                                                                                 | **Resolved: k=3 (ablate k=1,5 later)**     |
| D6 | H&M temporal resolution                   | 2-week periods (6 columns, matches exp.ipynb) or weekly (12 columns, matches Visuelle2)                                                             | **Resolved: (a) weekly, 12 columns** |

---

## Phase 0: Environment Setup

**Goal:** Reproducible Python environment for the project.

### Steps

- [ ] **0.1** Read `Cold-Start.../requirements.txt` and create a `requirements_cold_start.txt` at repo root that adds any new dependencies (FAISS, CLIP, timm, Pillow, scikit-learn).
- [ ] **0.2** Create a `conda` or `venv` environment and verify all imports work.
- [ ] **0.3** Verify GPU availability and CUDA version compatibility with PyTorch version in requirements.
- [ ] **0.4** Create a top-level `config/` directory structure in the project repo mirroring RATD's `config/` but with new YAML files for each dataset/task variant.

**Deliverable:** `requirements_cold_start.txt`, confirmed working environment.

---

## Phase 1: Data Preprocessing — Visuelle 2.0

**Goal:** Produce a clean, model-ready dataset with product attribute embeddings and precomputed retrieval indices, formatted to match RATD's expected input schema.

### 1.1 Understand the Existing Splits

The dataset already provides clean train/test splits:

- `stfore_train.csv` (96,166 rows) — training store-product pairs, SS17/SS18 seasons
- `stfore_test.csv` (10,684 rows) — test store-product pairs, primarily SS19
- Both have columns: `external_code`, `retail`, `season`, `category`, `color`, `fabric`, `image_path`, `release_date`, `restock`, `0`–`11` (normalized weekly sales, float)
- `stfore_sales_norm_scalar.npy` = 53.0 (multiply by this to recover approximate unit counts)
- `sales.csv` — absolute integer unit counts, same product-store pairs

**Action:** No resplitting needed. Use these splits as-is. The `season` column already enforces temporal integrity (test on unseen future seasons).

### 1.2 Feature Dimensionality: K=2 ✓

**Resolved:** K=2 — channels are `[sales, discount]` per product-store per week.

- Channel 0: normalized weekly sales (from `stfore_train/test.csv`, columns `0`–`11`)
- Channel 1: weekly discount ratio (from `price_discount_series.csv`, columns `0`–`11`, values in [0, 1])

Discount is a first-order demand driver and is known to the retailer in advance (planned promotions), making it valid conditioning information at inference time.

### 1.3 Build the Core Sales DataFrame

**Granularity decision:** Training and evaluation are done at the **store-product level** (96,166 rows in train). This preserves maximum training examples and allows store-specific weather features. Aggregation across stores is only used when constructing **reference trajectories** (see §1.5).

- [ ] **1.3.1** Load `stfore_train.csv` and `stfore_test.csv`.
- [ ] **1.3.2** Keep sales in normalized form (columns `0`–`11`, already divided by 53.0). Denormalize only at evaluation time using `stfore_sales_norm_scalar.npy` = 53.0.
- [ ] **1.3.3** Merge `price_discount_series.csv` on (`external_code`, `retail`) to add discount columns `d_0`–`d_11`.
- [ ] **1.3.4** Stack into a tensor of shape `(N, 2, 12)` — channel 0 = sales, channel 1 = discount — for train and test separately.
- [ ] **1.3.5** Apply `StandardScaler` **per channel** fit only on train, applied to both train and test. Save scaler for inverse transform at evaluation.

### 1.4 Compute Product Attribute Embeddings

These embeddings serve as the product representation for retrieval AND as conditioning signals fed into the model. Each unique `external_code` gets one embedding vector.

#### 1.4.1 Image Embeddings ✓

**Resolved:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`).

- [ ] Load product images from `visuelle2/images/{image_path}`.
- [ ] Preprocess with CLIP's own image transforms (resize to 224×224, normalize).
- [ ] Run `clip_model.encode_image()` in batches (batch_size=64) → 512-dim vectors.
- [ ] L2-normalize all vectors (unit length → cosine similarity = dot product).
- [ ] Save as `visuelle2_processed/image_embeddings.pt` (dict: `external_code → Tensor(512)`).

#### 1.4.2 Tag Embeddings ✓

**Resolved:** CLIP text encoder (same `openai/clip-vit-base-patch32` model as images).

- [ ] For each unique product, compose a single descriptive string from its three tags:
  `text = f"a {category} top, {color} color, {fabric} fabric"`
  (e.g. `"a long sleeve top, black color, matte jersey fabric"`)
- [ ] Run `clip_model.encode_text()` in batches → 512-dim vectors.
- [ ] L2-normalize all vectors.
- [ ] Save as `visuelle2_processed/tag_embeddings.pt` (dict: `external_code → Tensor(512)`).

#### 1.4.3 Price Scalar

- [ ] Extract the `price` column from `price_discount_series.csv` (one value per product-store). Take product-level mean across stores.
- [ ] Verify it is already in [0, 1] (dataset is max-normalized).
- [ ] Save as `visuelle2_processed/price_scalars.pt` (dict: `external_code → float`).

#### 1.4.4 Combined Product Embedding ✓

Since CLIP image and text encoders share a joint embedding space, we fuse them via **element-wise sum then re-normalize** rather than concatenation. Price is kept separate as a scalar appended after.

- [ ] For each product: `product_emb = L2_normalize(image_emb + tag_emb)` → 512-dim.
- [ ] Append price scalar: `full_emb = concat([product_emb, price_scalar])` → 513-dim.
- [ ] Save as `visuelle2_processed/product_embeddings.pt` (dict: `external_code → Tensor(513)`).
- [ ] Inside the model, project `513 → attr_emb_dim` (e.g. 64) via `nn.Linear(513, attr_emb_dim)` — this is a learned projection trained end-to-end with the diffusion model.

### 1.5 Build the Retrieval Database and Precompute Reference Indices

This is a **full replacement** of the TCN-based retrieval in the original RATD.

**Key design:** Retrieval is done at the **product level** (one embedding per `external_code`, based on attributes). Reference trajectories are the **mean sales+discount trajectory averaged across all stores** for that product in the training set. This decouples retrieval (what products are similar?) from the store-level granularity of training samples.

- [ ] **1.5.1** Build a FAISS index over training-set **product-level** embeddings (one vector per unique `external_code` in train).
  - Index type: `faiss.IndexFlatIP` (inner product over L2-normalized vectors = cosine similarity).
  - Index contains only training products — no test product embeddings in the index.
- [ ] **1.5.2** For each unique product in train + test, query the index for top-**3** nearest neighbors.
  - Exclude the query product itself from its own neighbors (train only).
  - Store as `visuelle2_processed/train_ref_indices.pt` and `visuelle2_processed/test_ref_indices.pt`.
  - Format: `dict[external_code → List[external_code]]` — k nearest neighbor product codes.
- [ ] **1.5.3** For each retrieved neighbor product, compute its **mean trajectory across all stores** from the train split.
  - Average the `(N_stores, 2, 12)` tensor over the store dimension → `(2, 12)` per product.
  - Concatenate k such trajectories → reference shape `(2, k*12)` per query product.
  - This matches RATD's `(K, k*pred_len)` reference format.
- [ ] **1.5.4** Save precomputed per-product reference tensors as `visuelle2_processed/train_references.pt` and `visuelle2_processed/test_references.pt`.
  - Format: `dict[external_code → Tensor(2, k*12)]`
  - At dataset `__getitem__` time, look up the reference by `external_code` — same reference tensor reused for all stores of that product.

### 1.6 Masking Strategy (All Settings)

A single masking scheme covers all evaluation settings. The `cond_mask` reveals the first `n_obs` weeks of the 12-week window:

- **n_obs=0:** pure cold-start — `cond_mask` is all zeros; no sales history visible
- **n_obs=1,2,3,4:** few-shot — first n_obs columns of `cond_mask` are 1, rest are 0

**Training:** randomly sample `n_obs ∈ {0, 1, 2, 3, 4}` per batch item. This trains a single model that handles all settings. The model learns to rely on attributes+references when n_obs=0 and on history+attributes+references when n_obs>0.

**Evaluation:** fix `n_obs` via `--n_obs` CLI argument. The same checkpoint is used for all benchmarks:

- `--n_obs 0` → pure cold-start results
- `--n_obs 2` → 2-week few-shot results (standard Visuelle2 benchmark)
- `--n_obs 4` → 4-week few-shot results

`observed_data` always contains all 12 weeks. The mask controls what the model can "see".

### 1.7 Dataset Class: `Dataset_Visuelle2`

Create `Cold-Start.../dataset_visuelle2.py` with the following interface:

```python
class Dataset_Visuelle2(Dataset):
    """
    Returns per-sample dict:
    - observed_data:  (2, 12)       — channel 0: normalized sales, channel 1: discount ratio
    - observed_mask:  (2, 12)       — all ones (no missing values)
    - gt_mask:        (2, 12)       — 1 for observed weeks, 0 for target weeks
    - timepoints:     (12,)         — [0, 1, ..., 11]
    - feature_id:     (2,)          — [0, 1]
    - reference:      (2, k*12)     — k retrieved product mean trajectories concatenated
    - product_emb:    (d,)          — product attribute embedding
    - external_code:  int           — product ID
    - retail:         int           — store ID
    """
```

**Deliverable:** `dataset_visuelle2.py`, all `.pt` files under `visuelle2_processed/`.

---

## Phase 2: Data Preprocessing — H&M Dataset

**Goal:** Create a comparable dataset structure from H&M for secondary validation, following the preprocessing logic already prototyped in `visuelle2/exp.ipynb`.

### 2.1 Extract and Formalize exp.ipynb Logic

The `exp.ipynb` notebook (in `visuelle2/`) already contains H&M preprocessing. Extract it into `scripts/preprocess_hnm.py`:

- [ ] **2.1.1** Load `hnm_dataset_orig/transactions_train.csv` (31.8M rows).
- [ ] **2.1.2** Load `hnm_dataset_orig/articles.csv` (105,542 articles).
- [ ] **2.1.3** Compute each article's launch date (first transaction date).
- [ ] **2.1.4** Filter to articles with ≥ 12 weeks of sales history (from notebook: 33,483 articles after density filter).
- [ ] **2.1.5** Create 12-week sales matrices at **weekly resolution** (12 columns, matching Visuelle 2.0). For each article, take the 12 consecutive weeks starting from its launch date.
  - Apply density filter: ≥ 100 total units sold across the 12 weeks (matches notebook threshold).
- [ ] **2.1.6** Split by launch date (chronological split to prevent leakage):
  - Train: articles launched before a cutoff date
  - Val: next 10% by launch date
  - Test: most recent 20% by launch date

### 2.2 Compute Product Attribute Embeddings for H&M

H&M articles have richer metadata (25 columns) and images.

- [ ] **2.2.1** Image embeddings: `images/{article_id[:3]}/{article_id}.jpg` → CLIP ViT-B/32.
  - Not all articles have images — for missing images, use a zero vector or the mean embedding over train articles.
  - Flag: ~1% of articles lack images; document fallback strategy.
- [ ] **2.2.2** Text embeddings: Combine `prod_name + product_type_name + colour_group_name + garment_group_name + detail_desc` into a single string → CLIP text encoder → 512-dim.
- [ ] **2.2.3** Price: `price` column in `transactions_train.csv` (already normalized). Use per-article mean price.
- [ ] **2.2.4** Concatenate and save as `hnm_processed/product_embeddings.pt`.

### 2.3 Build Retrieval Database for H&M

Same FAISS workflow as Visuelle 2.0 (§1.5), applied to H&M:

- [ ] Build FAISS index over train article embeddings.
- [ ] For each article in train/val/test, retrieve top-k=3 nearest training articles.
- [ ] Construct reference tensors `(K=1, 3*12)` from retrieved trajectories.

### 2.4 Dataset Class: `Dataset_HnM`

Create `Cold-Start.../dataset_hnm.py` with the same interface as `Dataset_Visuelle2`.

**Deliverable:** `scripts/preprocess_hnm.py`, `dataset_hnm.py`, all `.pt` files under `hnm_processed/`.

---

## Phase 3: Model Architecture Adaptation

**Goal:** Modify RATD to accept product attribute conditioning and support Task A (few-shot) and Task B (pure cold-start).

### 3.1 Product Attribute Injection: (a) + (b) simultaneously ✓

**Resolved:** Both injection paths are active in the same model. They are complementary, not mutually exclusive:

- **(a) `side_info` extension** — project `product_emb (B, d)` → `(B, attr_emb_dim)` → expand to `(B, attr_emb_dim, K, L)` → concatenate into `side_info`. This conditions **every residual block at every timestep**. `side_dim` in config is updated to include `attr_emb_dim`.
- **(b) RMA attribute input** — modify `ReferenceModulatedCrossAttention` to accept `attr_embedding` as an additional Key/Value source alongside retrieved references. The attention becomes: attend over `[noisy_state, reference_trajectories, product_attributes]`. This conditions **how references are weighted and blended** — critical when no sales history is available.

**Why both?** When `n_obs=0` (pure cold-start), there is no history signal; the model relies entirely on attributes + references. Having attributes in both the residual stream (a) and the reference fusion (b) gives the model two complementary pathways to use attribute information. When `n_obs>0`, history is available through `cond_mask` and attributes provide supplementary context through both paths.

**Single model, all settings:** Training randomly samples `n_obs ∈ {0, 1, 2, 3, 4}` per batch item (including 0). The model learns one set of weights covering all observation counts. At evaluation, `--n_obs N` fixes the setting — same checkpoint used for pure cold-start, 2-week few-shot, 4-week few-shot, etc. No separate code files needed.

### 3.2 Implement `RATD_Fashion` — New Model Subclass

Create `main_model_fashion.py`:

```python
class RATD_Fashion(RATD_base):
    """
    Subclass of RATD_base for cold-start fashion sales forecasting.

    Key differences from RATD_Forecasting:
    - Processes (K, 12) tensors instead of (K, seq_len+pred_len)
    - Accepts product_emb as conditioning signal
    - Supports n_obs-week partial conditioning (Task A)
    - Supports zero-observation pure cold-start (Task B)
    """
```

**Methods to implement:**

- [ ] **3.2.1** `process_data(batch)` — Converts Dataset dict to model tensors. Handles `product_emb` field (absent in base class). Returns 8-tuple including `product_emb`.
- [ ] **3.2.2** `get_side_info(observed_tp, cond_mask, product_emb)` — Override to include attribute embedding. Project `product_emb` via `nn.Linear(d_attr, emb_attr_dim)`. Expand across L dimension: `(B, emb_attr_dim, 1, 1)` → `(B, emb_attr_dim, K, L)`. Concatenate with standard time + feature embeddings.
- [ ] **3.2.3** `get_nobs_mask(n_obs)` — Returns `cond_mask` with first `n_obs` columns = 1, rest = 0. Used in Task A evaluation.
- [ ] **3.2.4** `forward(batch, is_train)` — Uses `get_nobs_mask` with random `n_obs ~ U{1,4}` during training.
- [ ] **3.2.5** `evaluate(batch, n_samples, n_obs)` — Generates forecast samples for a fixed `n_obs`.

### 3.3 Modify `diff_models.py` for Attribute Conditioning

- [ ] **3.3.1** `side_dim` is passed as a config parameter to `diff_RATD` and flows into `ResidualBlock.side_info_projection` (a `Conv2d(side_dim, 2*channels, 1)`). No code change needed here — updating `side_dim` in the config and in `get_side_info()` is sufficient for path (a).
- [ ] **3.3.2** Modify `ReferenceModulatedCrossAttention.__init__` to accept `attr_dim` parameter. Add a linear projection `attr_to_kv: nn.Linear(attr_dim, C)` for path (b).
- [ ] **3.3.3** Modify `ReferenceModulatedCrossAttention.forward` to accept `attr_embedding (B, d_attr)`:
  - Project: `attr_kv = attr_to_kv(attr_embedding)` → `(B, C)`
  - Expand to match spatial dims: `(B, C, 1, 1)` → broadcast with K and L
  - Concatenate into Key and Value alongside the existing reference tensor
- [ ] **3.3.4** Propagate `attr_embedding` through `ResidualBlock.forward` → `diff_RATD.forward` → `RATD_Fashion.impute/evaluate`.

### 3.4 Configuration Files

Create new YAML configs under `Cold-Start.../config/`:

- [ ] **`visuelle2_few_shot.yaml`** — Task A on Visuelle 2.0

  ```yaml
  train:
    epochs: 100
    batch_size: 16
    lr: 3.0e-4

  data:
    pred_len: 12         # full 12-week season
    n_obs_min: 0         # 0 = pure cold-start included in training
    n_obs_max: 4         # max observed weeks during training

  diffusion:
    layers: 4
    channels: 64
    nheads: 8
    diffusion_embedding_dim: 128
    beta_start: 0.0001
    beta_end: 0.5
    num_steps: 50
    schedule: "quad"
    is_linear: False     # K is small; no need for linear attention

  model:
    is_unconditional: 0
    timeemb: 128
    featureemb: 16
    attremb: 64          # NEW: product attribute embedding dim after projection
    target_strategy: "nobs"
    use_reference: True
    k_references: 3      # resolved: k=3; ablate k=1 and k=5 separately
  ```
- [ ] **`visuelle2_cold_start.yaml`** — Task B on Visuelle 2.0 (n_obs=0)
- [ ] **`hnm_few_shot.yaml`** — Task A on H&M
- [ ] **`hnm_cold_start.yaml`** — Task B on H&M

### 3.5 Update Experiment Entrypoint

Create `Cold-Start.../exe_fashion.py` (modeled on `exe_forecasting.py`):

- [ ] Remove hardcoded server paths (all paths should be config-driven or relative).
- [ ] Add `--dataset` argument (`visuelle2` or `hnm`).
- [ ] Add `--n_obs` argument for evaluation (fixed observation weeks; default 2 for Visuelle2).
- [ ] Add `--task` argument (`few_shot` or `cold_start`).
- [ ] Route to correct Dataset class based on `--dataset`.

**Deliverable:** `main_model_fashion.py`, updated `diff_models.py`, new config YAMLs, `exe_fashion.py`.

---

## Phase 4: Retrieval Pipeline (Offline Scripts)

**Goal:** Scripts that build FAISS indices and precompute reference indices. Runs once before training.

### 4.1 Embedding Computation Script

Create `scripts/compute_embeddings.py`:

- [ ] Accept `--dataset` and `--encoder` (clip/resnet) arguments.
- [ ] Load images + metadata for all products.
- [ ] Run encoder in batches (batch_size=64) to extract embeddings.
- [ ] Handle missing images (H&M only) with zero-vector or mean-vector fallback.
- [ ] L2-normalize all embeddings.
- [ ] Save to `{dataset}_processed/product_embeddings.pt`.

### 4.2 FAISS Index + Retrieval Script

Create `scripts/compute_retrieval.py`:

- [ ] Load train product embeddings.
- [ ] Build `faiss.IndexFlatIP` (cosine similarity via L2-normalized dot product).
  - For H&M (33K+ products), consider `faiss.IndexHNSWFlat` for faster approximate search.
- [ ] Query for each train + test product (top-k neighbors, excluding self for train).
- [ ] For each retrieved neighbor, extract their 12-week sales tensor from the train split.
- [ ] Save reference tensors as `{dataset}_processed/{split}_references.pt`.

**Deliverable:** `scripts/compute_embeddings.py`, `scripts/compute_retrieval.py`.

---

## Phase 5: Training

**Goal:** Fully reproducible training runs for both datasets and both tasks.

### 5.1 Training Run — Task A, Visuelle 2.0

- [ ] **5.1.1** Run `exe_fashion.py --dataset visuelle2 --config visuelle2_few_shot.yaml --task few_shot`
- [ ] **5.1.2** Monitor train loss curve; expect diffusion loss to decrease steadily.
- [ ] **5.1.3** Checkpoint every 10 epochs. Save best model on validation MAE.

### 5.2 Training Run — Task B, Visuelle 2.0

- [ ] Run with `--config visuelle2_cold_start.yaml --task cold_start`.

### 5.3 Training Run — H&M (Task A)

- [ ] Same procedure with `--dataset hnm`.

### 5.4 Ablation: No Reference

- [ ] Train with `use_reference: False` in config to isolate the contribution of retrieval augmentation.

**Deliverable:** Saved checkpoints under `save/` directory.

---

## Phase 6: Evaluation

**Goal:** Report standard metrics and compare against baselines.

### 6.1 Metrics

All metrics computed on the held-out prediction weeks (not the observed weeks):

| Metric             | Formula                             | Notes                                 |
| ------------------ | ----------------------------------- | ------------------------------------- |
| **MAE**      | Mean Absolute Error                 | Primary point forecast metric         |
| **RMSE**     | Root Mean Squared Error             | Penalizes large errors                |
| **WAPE**     | `sum(                               | y-ŷ                                  |
| **CRPS**     | Continuous Ranked Probability Score | Probabilistic forecast quality        |
| **CRPS_sum** | Sum-then-CRPS                       | Store-aggregated probabilistic metric |

Implement WAPE in `utils.py` alongside existing `calc_quantile_CRPS`.

### 6.2 Evaluation Protocol — Visuelle 2.0

- Evaluate at n_obs ∈ {1, 2, 3, 4} to show performance vs. early sales exposure.
- Report per n_obs on the test set (10,684 rows).
- Denormalize predictions before computing metrics.

### 6.3 Evaluation Protocol — H&M

- Evaluate at n_obs ∈ {1, 2, 4} weeks.
- Report on the chronological test split.

### 6.4 Baselines

Baselines are structured in four tiers by complexity. All are evaluated on the **same test split** and the **same n_obs settings** as RATD_Fashion for a fair comparison. Implementation targets Visuelle 2.0 first; H&M follows with the same code.

#### Tier 0: Attribute-Only Non-Parametric Baselines (n_obs = 0)

These require no training and establish a sanity-check floor.

| Baseline | Description | n_obs |
| -------- | ----------- | ----- |
| **Global Mean** | Predict the per-channel mean sales trajectory across all training products. No product-specific information used. | 0 |
| **K-NN Mean (k=3)** | Retrieve the 3 most similar training products (same FAISS index used by RATD), average their trajectories. Product-specific but no learning. | 0 |

#### Tier 1: RNN-LSTM Few-Shot Baselines (n_obs = 1,2,3,4)

A seq2seq LSTM conditioned on observed sales history + product embedding. Tests whether a simpler recurrent model can match diffusion performance when sales history is available. Only valid for n_obs >= 1 (no pure cold-start mode).

| Baseline | Description | n_obs |
| -------- | ----------- | ----- |
| **LSTM-NoAttr** | Encoder LSTM on observed sales only; decoder predicts remaining weeks. No product attributes. | 1,2,3,4 |
| **LSTM-Attr** | Same LSTM but encoder input is concatenated with product embedding (513-dim). Shows attribute value in recurrent setting. | 1,2,3,4 |

#### Tier 2: Time Series Foundation Model — Chronos Zero-Shot (n_obs = 1,2,3,4)

Chronos (Amazon, 2024) is a T5-based probabilistic time series model pre-trained on a large corpus of time series. Used zero-shot with no fine-tuning. Only the sales channel is passed (univariate); discount channel is dropped for this baseline. Only valid for n_obs >= 1 since Chronos requires at least one context observation.

| Baseline | Description | n_obs |
| -------- | ----------- | ----- |
| **Chronos-Mini** | `amazon/chronos-t5-mini` (21M params), zero-shot, sales channel only | 1,2,3,4 |

#### Tier 3: RATD_Fashion — Our Model

The main model is also evaluated across all n_obs settings to show the cold-start vs. few-shot progression.

| Setting | Description | n_obs |
| ------- | ----------- | ----- |
| **RATD (cold-start)** | No sales history; attributes + references only | 0 |
| **RATD (1-week)** | 1 week of sales + attributes + references | 1 |
| **RATD (2-week)** | 2 weeks (standard Visuelle2 benchmark point) | 2 |
| **RATD (3-week)** | 3 weeks | 3 |
| **RATD (4-week)** | 4 weeks | 4 |

#### Full Comparison Matrix

| Model | n_obs=0 | n_obs=1 | n_obs=2 | n_obs=3 | n_obs=4 |
| ----- | ------- | ------- | ------- | ------- | ------- |
| Global Mean | yes | - | - | - | - |
| K-NN Mean | yes | - | - | - | - |
| LSTM-NoAttr | - | yes | yes | yes | yes |
| LSTM-Attr | - | yes | yes | yes | yes |
| Chronos-Mini | - | yes | yes | yes | yes |
| RATD_Fashion (ours) | yes | yes | yes | yes | yes |

### 6.5 Evaluation Script Updates

- [ ] Add `calc_wape()` to `utils.py`.
- [ ] Modify `evaluate()` to also report WAPE.
- [ ] Save per-product metrics to allow stratified analysis (by category, season).

**Deliverable:** Evaluation results in `results/` directory; comparison table.

---

## Phase 7: Analysis & Ablations

**Goal:** Understand what drives performance; provide evidence for thesis claims.

### 7.1 Ablation Studies

| Ablation                                             | What it tests                       |
| ---------------------------------------------------- | ----------------------------------- |
| No retrieval (`use_reference=False`)               | Value of retrieved references       |
| Random retrieval (random neighbors instead of FAISS) | Importance of meaningful retrieval  |
| Image only (no text tags in embedding)               | Contribution of text attributes     |
| Tags only (no image embedding)                       | Contribution of visual features     |
| k=1 vs k=3 vs k=5 references                         | Sensitivity to number of references |
| n_obs=0 vs 1 vs 2 vs 4                               | Impact of early sales observations  |

### 7.2 Qualitative Analysis

- [ ] Visualize retrieved neighbors: show query product image + top-3 retrieved images.
- [ ] Plot predicted sales trajectories vs. ground truth for example products, annotating references used.
- [ ] Show failure cases: products with no close neighbors in embedding space.

### 7.3 Cross-Dataset Generalization (Stretch Goal)

- [ ] Train on H&M, evaluate on Visuelle 2.0 (zero-shot transfer). Tests whether learned representations generalize across retailers.

---

## Phase 8: Baselines Implementation

**Goal:** Implement and benchmark all Tier 0/1/2 baselines against RATD_Fashion. Visuelle 2.0 first; H&M follows with the same scripts parameterized by `--dataset`.

### 8.1 Tier 0: Global Mean and K-NN Mean

Create `baselines/run_knn_mean.py`. No PyTorch model, no training loop needed.

**Algorithm:**
1. Load `visuelle2_processed/product_embeddings.pt`, train references, train sales tensors.
2. For Global Mean: compute per-channel mean over all training products → `(K, 12)`. Broadcast to all test products.
3. For K-NN Mean: for each test product, load its precomputed `k=3` reference tensors (already in `test_references.pt`) and average them → `(K, 12)`.
4. Compute RMSE, MAE, WAPE, CRPS (degenerate: samples = single prediction repeated `nsample` times for CRPS compatibility).

**CLI:**
```bash
python baselines/run_knn_mean.py \
    --dataset visuelle2 \
    --processed_dir visuelle2_processed/ \
    --nsample 1
```

**Deliverable:** `baselines/run_knn_mean.py`, results saved to `results/baselines/knn_mean_{dataset}.json`.

### 8.2 Tier 1: LSTM Baselines

Create `baselines/lstm_baseline.py`. Single file containing both model definition and training/evaluation loop.

**Architecture:**

```
Input per timestep: sales[t] (+ optionally discount[t])      shape: (B, K)
Product embedding (LSTM-Attr only): concat to encoder input  shape: (B, K + attr_proj_dim)

Encoder LSTM: 2-layer unidirectional, hidden_dim=128
  - Input:  (B, n_obs, K) or (B, n_obs, K + attr_proj_dim)
  - Output: final hidden state h (B, 128)

Decoder: linear  h → (B, (L - n_obs) * K)
  - Reshape to (B, K, L - n_obs)
  - This is a direct regression decoder (no autoregressive loop needed for fixed L=12)
```

For LSTM-Attr: project `product_emb (B, 513)` → `(B, 32)` via `nn.Linear(513, 32)`, then concatenate to each encoder input step before feeding the LSTM.

**Training:**
- Fix `n_obs` at train time (train a separate checkpoint per n_obs value: 1, 2, 3, 4).
- Loss: MSE on normalised sales for the `(L - n_obs)` predicted steps.
- 50 epochs, Adam, lr=1e-3, batch_size=64.
- Validate on val split (same split strategy as RATD).

**CLI:**
```bash
# Train
python baselines/lstm_baseline.py \
    --dataset visuelle2 \
    --processed_dir visuelle2_processed/ \
    --n_obs 2 \
    --use_attr \
    --mode train \
    --save baselines/save/lstm_attr_n2/

# Evaluate
python baselines/lstm_baseline.py \
    --dataset visuelle2 \
    --modelfolder baselines/save/lstm_attr_n2/ \
    --n_obs 2 \
    --use_attr \
    --mode eval
```

**Deliverable:** `baselines/lstm_baseline.py`, checkpoints under `baselines/save/lstm_{attr|noattr}_n{1,2,3,4}/`.

### 8.3 Tier 2: Chronos Zero-Shot

Create `baselines/run_chronos.py`.

**Dependencies:**
```
pip install chronos-forecasting
```

**Usage pattern:**
```python
from chronos import ChronosPipeline
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-mini",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
```

**Evaluation loop:**
1. For each test product with `n_obs` observed weeks, extract context = sales channel, weeks `[0 : n_obs]` as a 1-D tensor.
2. Call `pipeline.predict(context, prediction_length=(12 - n_obs), num_samples=50)`.
3. Collect 50 sample trajectories → shape `(50, 12 - n_obs)`.
4. Compute median forecast for point metrics (RMSE, MAE, WAPE).
5. Compute CRPS directly from the 50-sample quantile distribution.

**Notes:**
- Chronos is **univariate only**: pass sales channel, drop discount. Note this in the results table.
- Context must be provided as a CPU tensor or list. Move to CPU before passing.
- Run in `torch.no_grad()` + `torch.inference_mode()` for speed.
- Batch size: use `pipeline.predict` with a list of contexts for throughput.

**CLI:**
```bash
python baselines/run_chronos.py \
    --dataset visuelle2 \
    --processed_dir visuelle2_processed/ \
    --n_obs 2 \
    --nsample 50 \
    --output results/baselines/chronos_n2_visuelle2.json
```

**Deliverable:** `baselines/run_chronos.py`, result JSONs under `results/baselines/`.

### 8.4 Results Aggregation Script

Create `baselines/aggregate_results.py`:
- Reads all `results/baselines/*.json` and `save/*/metrics_sweep.json`.
- Produces a single Markdown/CSV comparison table.
- Columns: Model, n_obs, RMSE, MAE, WAPE%, CRPS, CRPS_sum.
- Rows sorted by n_obs then model tier.

```bash
python baselines/aggregate_results.py \
    --ratd_folder save/vis2_run_20260414_120000 \
    --output results/comparison_table.md
```

### 8.5 Implementation Order

```
8.1 (K-NN Mean) → 8.4 partial (can aggregate Tier 0 immediately)
8.2 (LSTM)      → train n_obs=1,2,3,4 runs sequentially
8.3 (Chronos)   → run after LSTM (GPU contention)
8.4 (aggregate) → run after all baselines complete
```

All four scripts share the same processed data files; no additional preprocessing is needed beyond Phases 1 and 2.

---

## Target File Structure

```
Cold-Start-Demand-Forecasting-Using-Retrieval-Augmented-Diffusion-Models/
├── config/
│   ├── base_forecasting.yaml            (existing)
│   ├── visuelle2_few_shot.yaml          (NEW — Phase 3)
│   ├── visuelle2_cold_start.yaml        (NEW — Phase 3)
│   ├── hnm_few_shot.yaml                (NEW — Phase 3)
│   └── hnm_cold_start.yaml             (NEW — Phase 3)
├── scripts/
│   ├── preprocess_hnm.py               (NEW — Phase 2)
│   ├── compute_embeddings.py           (NEW — Phase 4)
│   └── compute_retrieval.py            (NEW — Phase 4)
├── baselines/
│   ├── run_knn_mean.py                  (NEW — Phase 8.1)
│   ├── lstm_baseline.py                 (NEW — Phase 8.2)
│   ├── run_chronos.py                   (NEW — Phase 8.3)
│   ├── aggregate_results.py             (NEW — Phase 8.4)
│   └── save/                            (LSTM checkpoints, auto-created)
├── dataset_forecasting.py              (existing — keep as-is)
├── dataset_visuelle2.py                (NEW — Phase 1)
├── dataset_hnm.py                      (NEW — Phase 2)
├── diff_models.py                      (MODIFY — Phase 3)
├── main_model.py                       (existing — keep base classes)
├── main_model_fashion.py               (NEW — Phase 3)
├── exe_forecasting.py                  (existing — keep as-is)
├── exe_fashion.py                      (NEW — Phase 3)
├── utils.py                            (MODIFY — Phase 6, add WAPE)
├── requirements.txt                    (existing)
└── requirements_cold_start.txt         (NEW — Phase 0)

results/
├── baselines/
│   ├── knn_mean_visuelle2.json          (Phase 8.1 output)
│   ├── knn_mean_hnm.json
│   ├── chronos_n{1,2,3,4}_visuelle2.json (Phase 8.3 output)
│   └── chronos_n{1,2,3,4}_hnm.json
└── comparison_table.md                  (Phase 8.4 output)

visuelle2_processed/                    (generated by Phase 1+4 scripts)
├── product_embeddings.pt
├── image_embeddings.pt
├── tag_embeddings.pt
├── price_scalars.pt
├── train_ref_indices.pt
├── test_ref_indices.pt
├── train_references.pt
├── test_references.pt
└── scaler.pkl

hnm_processed/                          (generated by Phase 2+4 scripts)
├── product_embeddings.pt
├── train_ref_indices.pt
├── test_ref_indices.pt
├── train_references.pt
├── test_references.pt
└── scaler.pkl
```

---

## Execution Order

```
Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 3 → Phase 5 → Phase 6 → Phase 7
Setup     Vis2      H&M       Retrieval  Model      Training  Eval      Analysis
          data      data      scripts    code

                                         Phase 8 (Baselines) runs in parallel with
                                         Phases 5-6; only needs processed data (Phases 1+4).
```

Phases 1 and 2 can overlap. Phase 4 depends on Phases 1 and 2. Phases 3 and 4 can overlap — retrieval scripts can be precomputed while the model code is being written.

Phase 8 dependency: only needs `visuelle2_processed/` (from Phases 1+4). Can start as soon as preprocessing is done, independently of model training.

**Baseline implementation order within Phase 8:** 8.1 (K-NN Mean, no GPU needed, 10 min) → 8.2 LSTM training (GPU, ~2 hrs for 4 n_obs variants) → 8.3 Chronos inference (GPU, ~1 hr) → 8.4 aggregate. For H&M, repeat 8.1–8.3 after `hnm_processed/` is ready.

---

## Standing Notes & Reminders

- **Never train on test data:** FAISS index must contain only `stfore_train.csv` products. Test products query the index but are never added to it.
- **Season-aware splits:** Visuelle2 temporal integrity is already enforced by the dataset's own splits (SS19 test). Respect it — do not re-split randomly.
- **Hardcoded server paths in original RATD:** `exe_forecasting.py` and `dataset_forecasting.py` reference `/data/0shared/liujingwei/...`. Never use these in new code; all paths must be config-driven or relative.
- **exp.ipynb is H&M preprocessing, not Visuelle2:** The notebook in `visuelle2/` processes H&M data. Formalize its logic in `scripts/preprocess_hnm.py` in Phase 2.
- **CRPS requires multiple samples:** Set `nsample >= 20` during evaluation to get stable quantile estimates.
- **K=1 feature attention:** If K=1, `forward_feature()` attention in `ResidualBlock` reduces to a trivial single-element sequence. Architecturally harmless but wasteful. If staying with K=1, consider disabling feature attention for efficiency (config flag).
- **This file is the living plan:** When a design decision is made or the approach changes, update the relevant section here and the Open Design Decisions table. Do not let this file go stale.
