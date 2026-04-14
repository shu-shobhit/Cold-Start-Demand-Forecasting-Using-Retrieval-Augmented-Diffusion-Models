# Implementation Log: Cold-Start Demand Forecasting with Retrieval-Augmented Diffusion Models

**Working directory:** `/home/shu_sho_bhit/BTP_2/`
**Main repo:** `Cold-Start-Demand-Forecasting-Using-Retrieval-Augmented-Diffusion-Models/`
**Last updated:** 2026-04-14

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Base Paper: RATD](#2-base-paper-ratd)
3. [Datasets](#3-datasets)
4. [Design Decisions Resolved](#4-design-decisions-resolved)
5. [Data Preprocessing Pipeline (Phase 1)](#5-data-preprocessing-pipeline-phase-1)
6. [Retrieval Pipeline (Phase 4)](#6-retrieval-pipeline-phase-4)
7. [Model Architecture (Phase 3)](#7-model-architecture-phase-3)
8. [Training Infrastructure (Phase 5)](#8-training-infrastructure-phase-5)
9. [Evaluation Infrastructure (Phase 6)](#9-evaluation-infrastructure-phase-6)
10. [Baselines (Phase 8)](#10-baselines-phase-8)
11. [File Structure and What Each File Does](#11-file-structure-and-what-each-file-does)
12. [What Is Not Yet Implemented](#12-what-is-not-yet-implemented)
13. [How to Run Everything](#13-how-to-run-everything)

---

## 1. Project Overview

The thesis problem is **new product introduction (NPI) cold-start forecasting** in fashion retail. At product launch, there is zero historical sales data. Standard autoregressive forecasting models fail here because they require a history window. The goal is to forecast the full 12-week sales trajectory of a new product using only:

- Product attributes: product image, category tag, color tag, fabric tag, price.
- Retrieved similar products: the sales histories of the k=3 most attribute-similar products seen during training.
- Optionally: 1-4 weeks of early sales once the season begins (few-shot mode).

The approach adapts **RATD (Retrieval-Augmented Time series Diffusion, NeurIPS 2024)** to this domain. The core change is replacing RATD's TCN-based time-series retrieval with **product-attribute-based retrieval** using CLIP embeddings and FAISS, and adding product attribute conditioning to the denoising network.

Two tasks are addressed:
- **Task A (primary):** 1-4 weeks of early sales observed, forecast the rest of the 12-week season.
- **Task B (secondary):** Zero sales observed, forecast from attributes + references alone.

A single model handles both tasks by training with randomly sampled `n_obs` in [0, 4].

**Primary dataset:** Visuelle 2.0 (Italian fashion, 5355 products, 110 stores, shop-level weekly sales).
**Secondary dataset:** H&M (Kaggle, 105K articles, 31.8M transactions) -- pipeline defined but not fully run yet.

---

## 2. Base Paper: RATD

**Full title:** "Retrieval Augmented Diffusion Models for Time Series Forecasting" (NeurIPS 2024)
**PDF location:** `Papers/NeurIPS-2024-retrieval-augmented-diffusion-models-for-time-series-forecasting-Paper-Conference.pdf`
**Breakdown notes:** `RATD_paper_breakdown.md`

Key ideas from RATD:
1. For each query time series, retrieve k nearest neighbor time series from a database (originally using TCN embeddings, cosine similarity).
2. Use the retrieved series as conditioning in a DDPM-style denoising network via a custom **Reference Modulated Attention (RMA)** cross-attention module.
3. The denoising network is non-autoregressive: the full forecast horizon is predicted simultaneously (like CSDI).
4. The model is a masked conditional diffusion model where the conditioning mask reveals observed history and the target mask is the held-out future.

The original RATD code lives in `RATD/` and `TCN-master/`. All fashion-specific code is new in the repo root.

---

## 3. Datasets

### 3.1 Visuelle 2.0

**Location:** `visuelle2/`
**Products:** 5,355 unique products from Italian retailer Nuna Lie across 6 seasons (AW/SS 2017-2019).
**Granularity:** Store-product level. 110 stores.
**Train split:** `stfore_train.csv` -- 96,166 rows (SS17, SS18 seasons).
**Test split:** `stfore_test.csv` -- 10,684 rows (primarily SS19 season).

Key columns in `stfore_*.csv`:
- `external_code`: unique product ID.
- `retail`: store ID.
- `category`, `color`, `fabric`: text attribute tags.
- `image_path`: relative path to product image under `visuelle2/images/`.
- `0` through `11`: normalized weekly sales (float, divide by 53.0 for original units).
- `release_date`, `season`, `restock`.

Additional files:
- `price_discount_series.csv`: per (external_code, retail) weekly discount ratios `d_0` through `d_11` and `price` column.
- `sales.csv`: absolute integer unit counts (same 106,850 rows).
- `stfore_sales_norm_scalar.npy` = 53.0 (global normalization denominator).
- `shop_weather_pairs.pt`: 110 stores mapped to weather locality (12 stores have None).
- Images in `visuelle2/images/` under Italian season subdirectories (PE17/PE18/PE19 for Spring/Summer, AI17/AI18/AI19 for Autumn/Winter).

**Important:** The season column enforces temporal integrity. SS17/SS18 are train, SS19 is test. No re-splitting.

### 3.2 H&M

**Location:** `hnm_dataset_orig/`
**Articles:** 105,542 articles.
**Transactions:** 31.8M rows in `transactions_train.csv`.
**Status:** Pipeline scripts written but Kaggle notebook not yet run; `hnm_processed/` files do not exist yet.

---

## 4. Design Decisions Resolved

| Decision | Resolution |
| -------- | ---------- |
| Feature channels K | K=2: sales + discount |
| Product attribute injection | Both path (a) side_info and path (b) RMA K/V -- in practice path (b) only is active in current model |
| Image encoder | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) |
| Tag encoder | CLIP text encoder (same model) |
| Retrieval similarity metric | Cosine similarity via L2-normalized inner product (FAISS IndexFlatIP) |
| Number of retrieved references | k=3 |
| H&M temporal resolution | Weekly, 12 columns (matching Visuelle 2.0) |
| Training masking | n_obs drawn uniformly from [0, 4] per sample during training |

---

## 5. Data Preprocessing Pipeline (Phase 1)

### 5.1 Output Files

All outputs live in `visuelle2_processed/`:

```
product_embeddings.pt     dict[int -> Tensor(513)]   fused CLIP embedding per product
image_embeddings.pt       dict[int -> Tensor(512)]   CLIP image features, L2-normalized
tag_embeddings.pt         dict[int -> Tensor(512)]   CLIP text features, L2-normalized
price_scalars.pt          dict[int -> float]         per-product mean price (already in [0,1])
train_ref_indices.pt      dict[int -> List[int]]     k=3 neighbor codes per train product
test_ref_indices.pt       dict[int -> List[int]]     k=3 neighbor codes per test product
train_references.pt       dict[int -> Tensor(2, 36)] precomputed reference trajectories (train)
test_references.pt        dict[int -> Tensor(2, 36)] precomputed reference trajectories (test)
scaler.pkl                StandardScaler fitted on train data
```

### 5.2 `scripts/compute_embeddings.py`

Computes CLIP-based product embeddings for all unique products.

**Step 1: Load products.**
Reads `stfore_train.csv` and `stfore_test.csv`, deduplicates by `external_code`, takes the first occurrence per product for metadata (category/color/fabric/image_path). Price is the mean of the `price` column across all stores from `price_discount_series.csv`.

**Step 2: Image embeddings.**
For each product, loads the image from `visuelle2/images/{image_path}`. Uses `CLIPProcessor` from HuggingFace (`openai/clip-vit-base-patch32`) to resize and normalize (224x224, standard CLIP normalization). Runs `CLIPModel.get_image_features()` in batches of 64. Output is 512-dim. L2-normalizes to unit length so cosine similarity equals dot product.

Handles missing images gracefully (not all products may have images).

Saves `image_embeddings.pt`: `dict[external_code -> Tensor(512)]`.

**Step 3: Tag embeddings.**
Composes a text description string per product:
```
"a {category} top, {color} color, {fabric} fabric"
e.g. "a long sleeve top, black color, matte jersey fabric"
```
Runs `CLIPModel.get_text_features()` in batches. Output is 512-dim. L2-normalizes.

Saves `tag_embeddings.pt`: `dict[external_code -> Tensor(512)]`.

**Step 4: Price scalars.**
Extracts per-product mean price. Price is already normalized to [0, 1] in the dataset.

Saves `price_scalars.pt`: `dict[external_code -> float]`.

**Step 5: Fused product embedding.**
Because CLIP image and text encoders share the same 512-dim joint embedding space, they are fused via element-wise sum then re-normalization (rather than concatenation, which would double the dimension). Price is appended as a scalar after fusion:

```python
fused_512 = L2_normalize(image_emb + tag_emb)     # (512,)
product_emb = concat([fused_512, price_scalar])     # (513,)
```

Saves `product_embeddings.pt`: `dict[external_code -> Tensor(513)]`.

**CLI:**
```bash
python scripts/compute_embeddings.py \
    --dataset visuelle2 \
    --data_root /home/shu_sho_bhit/BTP_2 \
    --out_dir /home/shu_sho_bhit/BTP_2/visuelle2_processed \
    --batch_size 64 \
    --device cuda
```

---

## 6. Retrieval Pipeline (Phase 4)

### `scripts/compute_retrieval.py`

Builds a FAISS index and precomputes per-product reference trajectories.

**Step 1: Build FAISS index.**
Loads `product_embeddings.pt` for all training products. Builds a `faiss.IndexFlatIP` (exact inner product search = cosine similarity with L2-normalized vectors). Only training products are added to the index -- test products are never added, only queried.

**Step 2: Query for neighbors.**
For train products: retrieves top-(k+1) neighbors, then removes the product itself (self is always the nearest neighbor in a train-set query). Keeps k=3.
For test products: retrieves top-k=3 neighbors directly (no self-exclusion needed).

Saves `train_ref_indices.pt` and `test_ref_indices.pt`: `dict[external_code -> List[external_code]]`.

**Step 3: Build reference trajectories.**
For each retrieved neighbor product, computes its **mean trajectory across all stores** from the training split. This aggregates the (N_stores, 12, 2) per-product data to a single (12, 2) representative trajectory.

For a query product with k=3 neighbors, concatenates the 3 mean trajectories along the time axis:
```python
reference = concat([traj1 (2,12), traj2 (2,12), traj3 (2,12)], axis=1)  # (2, 36)
```

This matches RATD's expected `(K, k*pred_len)` reference format.

Saves `train_references.pt` and `test_references.pt`: `dict[external_code -> Tensor(2, 36)]`.

**CLI:**
```bash
python scripts/compute_retrieval.py \
    --dataset visuelle2 \
    --data_root /home/shu_sho_bhit/BTP_2 \
    --emb_dir /home/shu_sho_bhit/BTP_2/visuelle2_processed \
    --out_dir /home/shu_sho_bhit/BTP_2/visuelle2_processed \
    --k 3
```

---

## 7. Model Architecture (Phase 3)

### 7.1 Overview

The model stack is:

```
RATD_Fashion (main_model_fashion.py)
  └── inherits RATD_base (main_model.py)
        └── contains diff_RATD (diff_models.py)
              └── N=4 ResidualBlock layers
                    each with:
                      DiffusionEmbedding projection
                      cond_projection (Conv1d)
                      ReferenceModulatedCrossAttention (RMA)  ← modified for attr
                      CrossAttention (temporal mixing)
                      CrossAttention (feature mixing)
                      gated residual + skip connection
```

### 7.2 `main_model.py` -- `RATD_base`

The base class manages all diffusion math and DDPM training that is dataset-agnostic.

**Initialization params from config:**
- `target_dim`: K=2 channels (sales + discount).
- `emb_time_dim`: 128 (sinusoidal time embedding dimension).
- `emb_feature_dim`: 16 (learned feature identity embedding dimension).
- `is_unconditional`: 0 (we use conditional mode; the conditioning mask is concatenated into side_info).
- `emb_total_dim = 128 + 16 + 1 = 145` (time + feature + mask channel).

**Diffusion schedule (from config `visuelle2.yaml`):**
- `num_steps = 50`
- `beta_start = 0.0001`, `beta_end = 0.5`
- `schedule = "quad"`: beta values are spaced quadratically. `beta = linspace(sqrt(beta_start), sqrt(beta_end), T)^2`.
- `alpha_hat = 1 - beta` (noise variance schedule).
- `alpha = cumprod(alpha_hat)` (cumulative product for arbitrary-step forward process).

**DDPM loss (noise prediction):**
```
noise ~ N(0, I)
noisy_data = sqrt(alpha[t]) * x + sqrt(1 - alpha[t]) * noise
predicted_noise = diffmodel(concat(cond_obs, noisy_target), side_info, t, reference, attr_emb)
target_mask = observed_mask - cond_mask   # held-out positions
loss = MSE(noise, predicted_noise) on target_mask positions
```

**Side information construction (time + feature embeddings):**
```python
time_embed   = sinusoidal(timepoints)                   # (B, L, 128)
feature_embed = nn.Embedding(K)(arange(K))              # (K, 16)
side_info = cat([time_embed, feature_embed], dim=-1)    # (B, L, K, 145)
side_info = permute(0, 3, 2, 1)                         # (B, 145, K, L)
# Append conditioning mask as extra channel
side_info = cat([side_info, cond_mask.unsqueeze(1)], dim=1)  # (B, 146, K, L)
```

**Inference (reverse diffusion):**
```python
current_sample = randn(B, K, L)  # start from noise
for t in range(T-1, -1, -1):
    cond_obs     = cond_mask * observed_data       # known values
    noisy_target = (1 - cond_mask) * current_sample
    diff_input   = cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
    predicted    = diffmodel(diff_input, side_info, t, reference, attr_emb)
    # DDPM update step
    coeff1 = 1 / alpha_hat[t]^0.5
    coeff2 = (1 - alpha_hat[t]) / (1 - alpha[t])^0.5
    current_sample = coeff1 * (current_sample - coeff2 * predicted)
    if t > 0:
        sigma = sqrt((1 - alpha[t-1]) / (1 - alpha[t]) * beta[t])
        current_sample += sigma * randn_like(current_sample)
```

### 7.3 `main_model_fashion.py` -- `RATD_Fashion`

Subclass of `RATD_base`. Key additions:

**Constructor:**
Calls `super().__init__(target_dim=2, config, device)` which builds the base diffmodel without `attr_dim`. Then **discards** that diffmodel and rebuilds it with `attr_dim=513`:
```python
config_diff["attr_dim"] = 513
self.diffmodel = diff_RATD(config_diff, inputdim=2)
```
This wires the product embedding into every RMA layer (path b).

**`process_data(batch)` -- batch preparation:**
- Takes raw `Dataset_Visuelle2` dict.
- Permutes tensors from `(B, L, K)` to `(B, K, L)` as the diffmodel expects `(B, K, L)`.
- Reference: `(B, 36, 2)` from dataset is transposed to `(B, 2, 36)`.
- `cut_length` is set to all zeros (no history/forecast split overlap in cold-start mode).
- Returns 8-tuple: `(observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length, reference, product_emb)`.

**`get_side_info(observed_tp, cond_mask)`:**
Builds standard RATD side information (time + feature embeddings + conditional mask). Product attributes are NOT added here (no path-a implementation in current code). Side info shape: `(B, 146, K=2, L=12)`.

**`calc_loss(...)` -- training loss:**
Identical to base except passes `attr_emb=self._attr_emb` to `self.diffmodel(...)`. This activates path-b conditioning inside every RMA layer.

**`impute(...)` -- reverse diffusion sampling:**
Same DDPM reverse loop but passes both `reference=self._reference` and `attr_emb=self._attr_emb` to `self.diffmodel(...)` at every step.

**`forward(batch, is_train=1)` -- training forward pass:**
1. Calls `process_data(batch)` to get all tensors.
2. Stores `self._attr_emb = product_emb` and `self._reference = reference` as instance variables (shared between `calc_loss` and `impute` within one forward pass).
3. `cond_mask = gt_mask` (the gt_mask already encodes which weeks are observed).
4. Calls `get_side_info` then `calc_loss` (or `calc_loss_valid`).

**`evaluate(batch, n_samples)`:**
Calls `process_data`, then `impute` with `n_samples` samples to produce `(B, n_samples, K, L)` output.

### 7.4 `diff_models.py` -- Diffusion Backbone

#### `diff_RATD`

The main denoising network. Architecture:

```
Input: (B, 2, K, L)   [2 = cond + noisy channels]
  └── input_projection: Conv1d(2, 64, 1) on K*L  → (B, 64, K, L)
  └── 4 × ResidualBlock
        each outputs a skip connection
  └── sum(skips) / sqrt(4)  → (B, 64, K, L)
  └── output_projection1: Conv1d(64, 64, 1)
  └── ReLU
  └── output_projection2: Conv1d(64, 1, 1) [zero-initialized]
  └── reshape → (B, K, L)  [predicted noise]
```

Config values used:
- `channels = 64` (hidden dimension throughout).
- `layers = 4` (number of residual blocks).
- `nheads = 8` (attention heads in time/feature transformers and RMA).
- `diffusion_embedding_dim = 128`.
- `h_size = 0` (no separate history window; masking handles this).
- `ref_size = 12` (single reference trajectory length = pred_len).
- `is_linear = False` (uses standard PyTorch transformer, not linear attention).
- `attr_dim = 513` (from RATD_Fashion constructor, enables path-b RMA conditioning).

#### `DiffusionEmbedding`

Sinusoidal embedding for discrete diffusion timesteps, then two linear projections with SiLU activations:
```python
table[t] = cat(sin(t * 10^(arange(64)/(63)*4)), cos(t * 10^(arange(64)/(63)*4)))  # 128-dim
x = Linear(128, 128)(table[t])
x = SiLU(x)
x = Linear(128, 128)(x)
x = SiLU(x)
```

#### `ResidualBlock`

One residual denoising block. Processes shape `(B, C=64, K=2, L=12)`.

**Components:**
- `diffusion_projection`: `nn.Linear(128, 64)` -- injects timestep embedding into hidden state.
- `cond_projection`: `Conv1d(side_dim=146, 64, 1)` -- projects side info to channel width.
- `RMA`: `ReferenceModulatedCrossAttention(dim=ref_size+h_size=12, context_dim=ref_size*3=36, attr_dim=513)`.
- `attn1` (CrossAttention from diffusers): standard self-attention on queries.
- `time_layer`: `nn.TransformerEncoder` (1 layer, 8 heads, d_model=64, ff=64, GELU) for temporal mixing.
- `feature_layer`: same transformer for feature mixing.
- `mid_projection`: `Conv1d(64, 128, 1)` -- gated activation split.
- `output_projection`: `Conv1d(64, 128, 1)` -- residual + skip split.

**Forward pass:**
```
1. x_flat = x.reshape(B, 64, K*L)
2. y = x_flat + diffusion_projection(diffusion_emb)  # inject timestep
3. cond_info = cond_projection(side_info)             # (B, 64, K*L)
4. [if reference present, fusion_type=1]:
      cond_info = RMA(y as (B,64,K,L), cond_info as (B,64,K,L), reference, attr_emb)
5. y = y + cond_info.reshape(B, 64, K*L)
6. y = forward_time(y)    # transformer over L dim, per-K
7. y = forward_feature(y) # transformer over K dim, per-L
8. y = mid_projection(y)  # (B, 128, K*L)
9. [gate]: y = sigmoid(y[:,:64]) * tanh(y[:,64:])   # gated activation
10. residual, skip = output_projection(y).split(64, dim=1)
11. return (x_flat + residual)/sqrt(2), skip
```

`forward_time`: reshapes to `(B*K, L, 64)`, applies transformer, reshapes back.
`forward_feature`: reshapes to `(B*L, K, 64)`, applies transformer, reshapes back.

#### `ReferenceModulatedCrossAttention` (Modified from original RATD)

This is the core innovation. It fuses the noisy latent state with retrieved reference trajectories using bidirectional cross-attention. We added path-b attribute conditioning on top of the original RATD module.

**Constructor parameters:**
- `dim = ref_size + h_size = 12` -- the sequence length of the latent state seen by this attention.
- `context_dim = ref_size * 3 = 36` -- total reference length (3 trajectories × 12 steps).
- `attr_dim = 513` -- product embedding dimension (activates path-b).
- `attr_proj_dim = 32` -- dimension the product embedding is projected to before concatenation inside RMA.
- `heads = 8`, `dim_head = 64`.

**Linear projections (with attr_dim=513 active):**
```python
self.attr_proj  = nn.Linear(513, 32)
self.y_to_q     = nn.Linear(12, 8*64=512)      # queries from noisy state
self.cond_to_k  = nn.Linear(12+12+36+32, 512)  # keys: state+cond+ref+attr
self.ref_to_v   = nn.Linear(12+36+32, 512)     # values: state+ref+attr
self.to_out     = nn.Linear(512, 12)
self.context_to_out = nn.Linear(512, 36)
```

**Forward pass with attr_emb:**
```python
# Expand reference across C channels: (B, K, 36) → (B*C, K, 36)
reference = repeat(reference, 'b n c -> (b f) n c', f=C)

# Project attr to (B*C, K, 32)
attr = attr_proj(attr_emb)                         # (B, 32)
attr = attr.unsqueeze(1).unsqueeze(1).expand(B, C, K, 32).reshape(B*C, K, 32)

# Build queries, keys, values
q = y_to_q(x.reshape(B*C, K, L))                  # (B*C, K, 512)
k = cond_to_k(cat([x, cond_info, reference, attr], dim=-1))  # (B*C, K, 512)
v = ref_to_v(cat([x, reference, attr], dim=-1))    # (B*C, K, 512)

# Bidirectional attention
sim = einsum(k, v) * scale                         # (B*C, 8, K, K)
attn         = softmax(sim, dim=-1)   # forward: each position attends to refs
context_attn = softmax(sim, dim=-2)   # backward: refs attend back to position

out         = einsum(attn, v)          # (B*C, K, 512)
context_out = einsum(context_attn, k)  # (B*C, K, 512)

return to_out(out)   # (B*C, K, 12) = (B*C, K, L)
```

The output replaces `cond_info` in the residual block, meaning reference + attribute information flows into every denoising step at every layer.

**Note on zero-fill fallback:** When `attr_dim` is set but `attr_emb=None` at runtime, the module zero-fills the attribute tensor instead of crashing, preserving exact RATD behavior in that case.

### 7.5 `config/visuelle2.yaml`

```yaml
train:
  epochs: 100
  batch_size: 32
  lr: 1.0e-3
  itr_per_epoch: 1.0e+8   # effectively unlimited; uses full dataset each epoch

diffusion:
  layers: 4
  channels: 64
  nheads: 8
  diffusion_embedding_dim: 128
  beta_start: 0.0001
  beta_end: 0.5
  num_steps: 50
  schedule: "quad"
  is_linear: False
  h_size: 0      # no separate history window
  ref_size: 12   # single reference length = pred_len

model:
  is_unconditional: 0
  timeemb: 128
  featureemb: 16
  target_strategy: "test"   # required by RATD_base but unused in RATD_Fashion
  use_reference: True
```

`attr_dim` is not in the YAML; it defaults to 513 (the `ATTR_DIM` class constant in `RATD_Fashion`).

---

## 8. Training Infrastructure (Phase 5)

### 8.1 `dataset_visuelle2.py` -- `Dataset_Visuelle2`

The PyTorch Dataset class. Each sample is one `(product, store)` pair.

**`__init__` parameters:**
- `data_root`: path to `BTP_2/` (contains `visuelle2/`).
- `processed_dir`: path to `visuelle2_processed/` (precomputed `.pt` files).
- `flag`: `'train'` or `'test'`.
- `n_obs`: fixed integer (0-4) for evaluation. `None` = random draw during training.
- `n_obs_range`: `(0, 4)` range for random training draws.
- `scale`: `True` applies StandardScaler.

**Data loading (`_load_data`):**
1. Reads `stfore_train.csv` and `stfore_test.csv`.
2. Reads `price_discount_series.csv` and merges on `(external_code, retail)` to add `d_0`...`d_11` discount columns.
3. Missing discount values are filled with 0.0.
4. **StandardScaler**: stacks train data into `(N*12, 2)` format (flattened weeks × channels), fits scaler on train. For test, loads saved `scaler.pkl`. Saves scaler on first train run.
5. Builds `self.data`: `(N, 12, 2)` float32 normalized array using `_build_tensor`.
6. Stores `self.codes` (external_code per row) and `self.retails` (store ID per row).
7. Loads `product_embeddings.pt` (dict: code → Tensor(513)).
8. Loads `train_references.pt` or `test_references.pt` (dict: code → Tensor(2, 36)).

**`_build_tensor(df)` -- normalization:**
```python
sales = df[SALES_COLS].values   # (N, 12) float32
disc  = df[DISC_COLS].values    # (N, 12) float32
raw   = stack([sales, disc], axis=2)    # (N, 12, 2)
flat  = raw.reshape(N*12, 2)
flat  = scaler.transform(flat)
return flat.reshape(N, 12, 2)
```

**`__getitem__(index)` -- sample construction:**
```python
observed_data = Tensor(data[index])         # (12, 2)
observed_mask = ones(12, 2)                  # no structural missing
gt_mask = zeros(12, 2)
gt_mask[:n_obs, :] = 1.0                    # first n_obs weeks revealed

reference = references[code].T              # stored as (2,36), transposed to (36, 2)
product_emb = product_emb_dict[code]        # (513,)
```

Returns dict with keys: `observed_data (12,2)`, `observed_mask (12,2)`, `gt_mask (12,2)`, `timepoints (12,)`, `feature_id (2,)`, `reference (36,2)`, `product_emb (513,)`, `external_code (int)`, `retail (int)`.

**`inverse_transform_sales(data)`:**
Builds dummy `(N, 2)` array with a zero discount column to satisfy the 2-channel scaler, then inverse transforms and extracts channel 0 only. Used during evaluation to convert normalized predictions back to [0, 1] sales units (multiply by 53.0 for approximate integer counts).

**`get_dataloader` factory function:**
Creates train (shuffle=True, n_obs random from range) and test (shuffle=False, fixed n_obs) DataLoaders with configurable `batch_size`, `num_workers`, and `pin_memory=True`.

### 8.2 `utils.py` -- Training Loop and Metrics

#### `train()` function

Main training loop called by `exe_fashion.py`.

**Optimizer:** Adam, lr=1e-3 (from config), weight_decay=1e-6.

**LR schedule:** MultiStepLR, drops LR by 10x at epoch 75 and epoch 90 (75% and 90% of 100 total epochs).

**Checkpoint files written to `foldername`:**
- `checkpoint_epochN.pth`: full resumable checkpoint (model state + optimizer state + scheduler state + epoch + best_valid_loss). Written every `save_every` epochs.
- `checkpoint_latest.pth`: always overwritten with the most recent full checkpoint.
- `model_best.pth`: model weights only, saved when validation loss improves.
- `model.pth`: model weights only, saved at end of training.

**Resume support:** Accepts `resume_checkpoint` path. Restores model/optimizer/scheduler states and continues from `start_epoch = ckpt["epoch"] + 1`.

**Validation:** Runs every `valid_epoch_interval` epochs. Uses `is_train=1` (random timestep per batch, same as training) for speed. Saves `model_best.pth` when validation improves.

**Progress display:** Uses `rich` library (`Progress`, `SpinnerColumn`, `BarColumn`, etc.) for clean terminal output.

#### Metric functions in `utils.py`

**`quantile_loss(target, forecast, q, eval_points)`:**
Pinball loss for quantile `q`: `2 * sum(|forecast - target| * eval_points * (1_{target<=forecast} - q))`.

**`calc_denominator(target, eval_points)`:**
`sum(|target * eval_points|)` -- normalization denominator for CRPS.

**`calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler)`:**
De-normalizes target and forecast first (`x * scaler + mean_scaler`). Evaluates quantile loss at 19 quantile levels (0.05 to 0.95 step 0.05). Returns mean quantile loss normalized by denominator.

**`calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler)`:**
Same but sums across the K feature channels first, then computes CRPS on the summed series.

**`calc_wape(target, forecast_median, eval_points, scaler, mean_scaler)`:**
`sum(|forecast_denorm - target_denorm| * eval_mask) / sum(|target_denorm| * eval_mask)`. Returns fraction (multiply by 100 for %). De-normalizes before computing.

### 8.3 `exe_fashion.py` -- Training Entrypoint

**CLI arguments:**
```
--config          config/visuelle2.yaml
--data_root       /home/shu_sho_bhit/BTP_2
--processed_dir   /home/shu_sho_bhit/BTP_2/visuelle2_processed
--device          cuda (auto-detected)
--seed            42
--n_obs_eval      0    (for single-n_obs evaluation)
--n_obs_min       0    (training range min)
--n_obs_max       4    (training range max)
--nsample         50   (reverse diffusion samples at eval)
--modelfolder     ""   (if set, skips training and loads checkpoint)
--save_every      1    (checkpoint every N epochs -- default 1 creates many files)
--val_interval    1    (validate every N epochs)
--resume          ""   (path to checkpoint for resumption)
--sweep           (flag: evaluate at all n_obs 0-4)
```

**Run folder naming:** `save/fashion_visuelle2_{tag}_{YYYYMMDD_HHMMSS}/`.

**`evaluate_fashion()` function:**
Evaluates on sales channel (channel 0) only. Collects all batch outputs, computes streaming RMSE/MAE during the loop, then aggregates for WAPE/CRPS/CRPS_sum at the end. Saves:
- `generated_outputs_nsample{N}_{tag}.pk` (all samples, targets, masks).
- `result_nsample{N}_{tag}.pk` (metrics dict).
- `metrics_{tag}.json` (metrics dict, human-readable).

**`sweep_evaluate()` function:**
Mutates `test_dataset.n_obs` to cycle through 0, 1, 2, 3, 4. Builds a fresh DataLoader for each setting. Prints a comparison table. Saves `metrics_sweep.json` with all 5 settings.

**`_resolve_resume_checkpoint(path)`:**
Accepts a file path or folder. If folder: prefers `checkpoint_latest.pth`, falls back to highest-epoch `checkpoint_epoch*.pth`. Raises `FileNotFoundError` if nothing found.

**Main flow:**
```
1. Load config from YAML
2. Resolve resume checkpoint (if --resume given)
3. Create output folder
4. Build DataLoaders via get_dataloader()
5. Extract scaler params (scale_, mean_) from train_dataset.scaler
6. Build RATD_Fashion(config, device)
7. If no --modelfolder: call train(...)
   Else: load model_best.pth (or model.pth as fallback)
8. If --sweep: call sweep_evaluate(...)
   Else: build test loader with fixed n_obs_eval, call evaluate_fashion(...)
```

---

## 9. Evaluation Infrastructure (Phase 6)

### Metrics Reported

All metrics are computed only on held-out positions (`eval_mask = 1 - gt_mask`).

| Metric | Space | Channel | Formula |
| ------ | ----- | ------- | ------- |
| RMSE | Normalized (StandardScaler) | Sales (ch 0) | `sqrt(mean((pred - target)^2 * eval_mask * scale^2))` |
| MAE | Normalized | Sales (ch 0) | `mean(|pred - target| * eval_mask * scale)` |
| WAPE | De-normalized (original [0,1] scale) | Sales (ch 0) | `sum(|pred - target| * mask) / sum(|target| * mask)` |
| CRPS | De-normalized | Sales (ch 0) | Quantile pinball loss over 19 quantile levels, normalized by denominator |
| CRPS_sum | De-normalized | Sum over K channels | CRPS on channel-summed forecasts |

RMSE/MAE are reported in normalized space (after StandardScaler, before dividing by 53.0) to match RATD's original paper reporting convention. WAPE and CRPS are fully de-normalized.

### Evaluation at Multiple n_obs Settings

The same trained checkpoint is evaluated at `n_obs = 0, 1, 2, 3, 4` via `--sweep`. Only `test_dataset.n_obs` is mutated between runs; the model weights, dataset files, and scaler are reused.

### Degenerate CRPS for Point-Forecast Baselines

For the non-probabilistic baselines (Global Mean, K-NN Mean, LSTM), CRPS is computed by replicating the single point prediction 20 times:
```python
samples_t = pred_t.unsqueeze(1).expand(-1, 20, -1, -1)  # (N, 20, L, K)
crps = calc_quantile_CRPS(targets_t, samples_t, eval_mask, mean_scaler, scaler_val)
```
This gives a degenerate distribution where all mass is at the prediction. Numerically equals MAE for a point forecast. Included for table completeness.

---

## 10. Baselines (Phase 8)

### 10.1 Four-Tier Baseline Structure

| Tier | Baseline | Description | n_obs |
| ---- | -------- | ----------- | ----- |
| 0 | Global Mean | Per-channel mean trajectory over all training products, broadcast to all test samples | 0 |
| 0 | K-NN Mean | Average the 3 retrieved reference trajectories already in test_references.pt | 0 |
| 1 | LSTM-NoAttr | 2-layer encoder LSTM on observed sales only; linear decoder. hidden=256, ~796K params | 1,2,3,4 |
| 1 | LSTM-Attr | Same LSTM with 513-dim product embedding projected to 64-dim and concatenated to each encoder step. ~894K params | 1,2,3,4 |
| 2 | Chronos-Mini | amazon/chronos-t5-mini zero-shot, sales channel only, genuine 50-sample CRPS | 1,2,3,4 |
| 3 | RATD_Fashion (ours) | Full model | 0,1,2,3,4 |

### 10.2 `baselines/run_chronos.py` -- Tier 2

Zero-shot foundation model baseline. Requires `pip install chronos-forecasting`.
Supports both Visuelle 2.0 (K=2) and H&M (K=1) via a `--dataset` flag.

**Model:** `amazon/chronos-t5-mini` (21M parameters), T5-based probabilistic time series model pre-trained on a large public corpus. Used entirely zero-shot with no fine-tuning.

**Univariate constraint:** Chronos accepts only a 1-D context. Only the sales channel (index 0) is passed as context. The discount channel (Visuelle2) is ignored entirely.

**Dataset-agnostic inverse transform (`_inverse_transform_ctx`):**
Chronos's internal mean-scaling normalisation works best on non-negative inputs. The StandardScaler-normalised values used by all other baselines are zero-centred and can be negative. The context is inverse-transformed back to the original non-negative space before being passed to Chronos:
- Visuelle2: calls `dataset.inverse_transform_sales(ctx_scaled)` to get values in [0, 1].
- H&M: calls `dataset.scaler.inverse_transform(ctx_scaled.reshape(-1,1))` to undo the 1-D StandardScaler, returning log1p-scaled sales (always >= 0).
```python
ctx_01 = _inverse_transform_ctx(test_dataset, ctx_scaled)   # (n_obs,) non-negative
ctx_01 = np.clip(ctx_01, 0.0, None)                         # clip numerical noise
```

**Output re-normalisation:** Chronos returns predictions in the same non-negative space as the input context. These are then re-normalised to StandardScaler space for consistent metric computation:
```python
forecast_norm = (forecast_01 - mean_scaler) / scaler_val
```

**Dataset-agnostic data collection (`_collect_contexts_and_targets`):**
Uses `K = test_dataset.K_CHANNELS` (2 for Visuelle2, 1 for H&M) to allocate `targets_np (N, 12, K)` and `eval_mask_np (N, 12, K)` arrays. Metric functions only ever look at index 0 (sales channel), so both dataset shapes work transparently.

**Batched inference:**
```python
batch_ctx = torch.stack(contexts_01[start:end], dim=0)   # (B, n_obs) -- same length
forecast_01 = pipeline.predict(
    context=batch_ctx,
    prediction_length=pred_len,    # 12 - n_obs
    num_samples=nsample,           # 50
    limit_prediction_length=False,
)  # returns (B, nsample, pred_len)
```

**Full-length embedding:** Predictions cover weeks n_obs..11. They are embedded into a (N, nsample, 12) tensor with zeros at observed positions (0..n_obs-1). The eval_mask zeroes those positions out so they never contribute to metrics.

**Metrics:** All metrics are computed on the **sales channel only**:
- RMSE/MAE: in StandardScaler-normalised space on the median forecast.
- WAPE: de-normalised, median forecast.
- CRPS: de-normalised, uses the genuine 50-sample distribution (not degenerate). Directly comparable to RATD_Fashion's CRPS.
- CRPS_sum: K=1 (sales only), equals CRPS numerically for K=1.

**CLI:**
```bash
# Sweep n_obs=1,2,3,4 on Visuelle 2.0 (default)
python baselines/run_chronos.py

# Single n_obs
python baselines/run_chronos.py --mode single --n_obs 2

# H&M dataset
python baselines/run_chronos.py --dataset hnm \
    --hnm_processed_dir /path/to/hnm_processed

# Save results
python baselines/run_chronos.py --output results/baselines/chronos_visuelle2.json
```

---

### 10.3 `baselines/run_knn_mean.py` -- Tier 0

No training, no GPU required. Runs in minutes.

**`_build_eval_arrays(test_dataset, n_obs)`:**
Iterates the test set once, collects `targets_np (N, 12, 2)` from `observed_data` and `eval_mask_np (N, 12, 2)` from `1 - gt_mask`.

**`_metrics(targets_np, preds_np, eval_mask_np, scaler_val, mean_scaler, tag)`:**
Computes and logs RMSE, MAE, WAPE, CRPS, CRPS_sum using the same metric functions as the main model. Returns a dict.

**`run_global_mean(train_dataset, test_dataset, n_obs, scaler_val, mean_scaler)`:**
```python
mean_traj = train_dataset.data.mean(axis=0)          # (12, 2)
preds_np  = broadcast(mean_traj, (N, 12, 2))          # same prediction for every test sample
```
`train_dataset.data` is the `(N_train, 12, 2)` normalized array stored on the dataset object.

**`run_knn_mean(test_dataset, n_obs, scaler_val, mean_scaler)`:**
For each test sample:
```python
ref_kl  = test_dataset.references[code].numpy()  # (2, 36) -- from dict, before transposing
refs    = np.split(ref_kl, 3, axis=1)            # list of 3 arrays (2, 12)
knn_mean = np.mean(refs, axis=0)                 # (2, 12)
preds_np[i] = knn_mean.T                         # (12, 2)
```
Note: `references` dict stores `(2, 36)` tensors (not transposed). The `.T` at the end converts to `(12, 2)` for consistency with `targets_np`.

**CLI:**
```bash
python baselines/run_knn_mean.py --n_obs 0
python baselines/run_knn_mean.py --n_obs 0 --output results/baselines/knn_mean_visuelle2.json
```

### 10.3 `baselines/lstm_baseline.py` -- Tier 1

#### `LSTMForecaster` model

```python
class LSTMForecaster(nn.Module):
    """
    n_obs:       int, 1-4
    pred_len:    12 (full season)
    k_channels:  2 (sales + discount)
    hidden_dim:  256   # chosen so ~796K (NoAttr) / ~894K (Attr) ≈ RATD_Fashion 985K
    n_layers:    2
    attr_dim:    513 (if use_attr) or 0
    attr_proj_dim: 64
    future_len = pred_len - n_obs
    """
```

**Components:**
- `attr_proj`: `nn.Linear(513, 32)` -- only present when `use_attr=True`.
- `lstm`: `nn.LSTM(input_size=K[+32], hidden_size=128, num_layers=2, batch_first=True, dropout=0.1)`.
- `decoder`: `nn.Linear(128, future_len * K)` -- direct regression to all future steps simultaneously.

**`forward(obs, product_emb)` logic:**
```python
# LSTM-Attr only:
attr_vec = attr_proj(product_emb)                         # (B, 64)
attr_exp = attr_vec.unsqueeze(1).expand(-1, n_obs, -1)    # (B, n_obs, 64)
enc_input = cat([obs, attr_exp], dim=2)                   # (B, n_obs, K+32)

# Both variants:
_, (h_n, _) = lstm(enc_input)     # h_n: (2, B, 128)
h_last = h_n[-1]                  # (B, 128) -- last layer hidden state
out = decoder(h_last)             # (B, future_len * K)
pred = out.view(B, future_len, K) # (B, future_len, K)
```

#### Training loop (`train_lstm`)

- Adam optimizer, lr=1e-3, weight_decay=1e-5.
- MSELoss on held-out positions (`eval_mask = 1 - gt_mask` sliced to future weeks).
- 50 epochs, batch_size=64.
- Saves `model_best.pth` on best validation MSE.
- Saves `model.pth` at end.
- Loss computed as:
  ```python
  obs       = obs_data[:, :n_obs, :]         # (B, n_obs, 2)
  target    = obs_data[:, n_obs:, :]         # (B, future_len, 2)
  eval_mask = (1 - gt_mask)[:, n_obs:, :]   # (B, future_len, 2) -- held-out positions
  pred      = model(obs, emb_arg)
  loss      = MSE(pred * eval_mask, target * eval_mask)
  ```

#### Evaluation loop (`evaluate_lstm`)

Point metrics RMSE/MAE in normalized space, WAPE and CRPS with de-normalization. Degenerate CRPS from 20 replicated samples. Optionally saves `metrics_{tag}.json`.

#### CLI modes

```bash
# Train single variant
python baselines/lstm_baseline.py --n_obs 2 --use_attr --mode train

# Evaluate saved checkpoint
python baselines/lstm_baseline.py --n_obs 2 --use_attr --mode eval \
    --modelfolder baselines/save/lstm_attr_n2

# Sweep all n_obs (train + eval) without attributes
python baselines/lstm_baseline.py --mode sweep

# Sweep all n_obs with attributes
python baselines/lstm_baseline.py --mode sweep --use_attr
```

**Checkpoint folder naming:** `baselines/save/lstm_{attr|noattr}_n{n_obs}/`

**`_build_datasets(args, n_obs)`:** Uses `Dataset_Visuelle2` with fixed `n_obs` for both train and test (does not use random n_obs during LSTM training; each checkpoint is specialized for one n_obs value).

---

## 11. File Structure and What Each File Does

```
Cold-Start-Demand-Forecasting-Using-Retrieval-Augmented-Diffusion-Models/
│
├── config/
│   ├── base.yaml                    (base RATD config; used for reference)
│   ├── base_forecasting.yaml        (original forecasting config; not used)
│   ├── visuelle2.yaml               (NEW -- active training config for RATD_Fashion)
│   └── hnm.yaml                     (NEW -- H&M config; dataset not yet run)
│
├── scripts/
│   ├── compute_embeddings.py        (NEW -- CLIP embedding computation)
│   └── compute_retrieval.py         (NEW -- FAISS index + reference precomputation)
│
├── baselines/
│   ├── run_knn_mean.py              (NEW -- Tier 0: Global Mean + K-NN Mean)
│   ├── lstm_baseline.py             (NEW -- Tier 1: LSTM-NoAttr + LSTM-Attr)
│   ├── run_chronos.py               (NEW -- Tier 2: Chronos-Mini zero-shot)
│   └── save/                        (auto-created LSTM checkpoint directories)
│
├── dataset_visuelle2.py             (NEW -- PyTorch Dataset, masking, scaler)
├── dataset_hnm.py                   (NEW -- H&M Dataset; pipeline not yet run)
├── diff_models.py                   (MODIFIED -- added attr_dim to RMA, ResidualBlock, diff_RATD)
├── main_model.py                    (original -- kept, provides RATD_base)
├── main_model_fashion.py            (NEW -- RATD_Fashion subclass)
├── main_model_hnm.py                (NEW -- H&M variant)
├── exe_fashion.py                   (NEW -- training/eval entrypoint for Visuelle 2.0)
├── exe_hnm.py                       (NEW -- H&M entrypoint)
├── exe_forecasting.py               (original -- kept, not used for fashion)
├── utils.py                         (MODIFIED -- added calc_wape, rich progress, resume, sweep support)
├── requirements_cold_start.txt      (NEW -- additional dependencies for this project)
├── My_BTP_Project.md                (living project plan with all design decisions)
└── IMPLEMENTATION_LOG.md            (this file)

visuelle2_processed/                 (generated by scripts/; NOT in repo)
├── product_embeddings.pt
├── image_embeddings.pt
├── tag_embeddings.pt
├── price_scalars.pt
├── train_ref_indices.pt
├── test_ref_indices.pt
├── train_references.pt
├── test_references.pt
└── scaler.pkl

save/fashion_visuelle2_*_YYYYMMDD/   (training run outputs)
├── config.json
├── checkpoint_epochN.pth            (full resumable checkpoint)
├── checkpoint_latest.pth            (always latest)
├── model_best.pth                   (lowest val loss weights only)
├── model.pth                        (final epoch weights only)
├── generated_outputs_nsample50_n*.pk
├── result_nsample50_n*.pk
├── metrics_n*.json
└── metrics_sweep.json               (if --sweep was used)
```

### Files not modified from original RATD

- `main_model.py`: provides `RATD_base` which RATD_Fashion inherits.
- `dataset_forecasting.py`: original electricity/ETT dataset class, kept for reference.
- `exe_forecasting.py`: original training script for ETT/electricity, kept for reference.
- `TCN-master/`: TCN retrieval code from original RATD, not used in fashion pipeline.

---

## 12. What Is Not Yet Implemented

| Component | Status | Notes |
| --------- | ------ | ----- |
| Tier 2: Chronos baseline | **Implemented** (`baselines/run_chronos.py`) | Needs `pip install chronos-forecasting` to run |
| Results aggregation script | Not started | `baselines/aggregate_results.py` planned in Phase 8.4 |
| H&M preprocessing | Not run | `scripts/preprocess_hnm.py` exists; Kaggle notebook needs to be run first |
| H&M FAISS retrieval | Not run | Depends on H&M preprocessing |
| H&M training | Not started | Depends on `hnm_processed/` files |
| Ablation: no reference | Not started | Train with `use_reference: False` |
| Ablation: random retrieval | Not started | Randomize neighbor assignment |
| Ablation: image-only / tags-only | Not started | Edit `compute_embeddings.py` |
| Phase 8.1 runtime n_obs guard | Partial | `lstm_baseline.py` help text warns n_obs=0 is invalid but no assertion/p.error() |
| Results reporting | Pending | RATD training is in progress (was at epoch 21 of 100 as of last session) |

---

## 13. How to Run Everything

### Prerequisites

```bash
conda activate ML
pip install -r requirements_cold_start.txt
# requirements include: torch, faiss-gpu, transformers (for CLIP),
#                        rich, diffusers, einops, linear-attention-transformer
```

### Step 1: Compute CLIP embeddings (once)

```bash
python scripts/compute_embeddings.py \
    --dataset visuelle2 \
    --data_root /home/shu_sho_bhit/BTP_2 \
    --out_dir /home/shu_sho_bhit/BTP_2/visuelle2_processed \
    --device cuda
```

### Step 2: Compute FAISS retrieval + reference tensors (once)

```bash
python scripts/compute_retrieval.py \
    --dataset visuelle2 \
    --data_root /home/shu_sho_bhit/BTP_2 \
    --emb_dir /home/shu_sho_bhit/BTP_2/visuelle2_processed \
    --out_dir /home/shu_sho_bhit/BTP_2/visuelle2_processed \
    --k 3
```

### Step 3: Train RATD_Fashion

```bash
# Fresh training
python exe_fashion.py --device cuda --save_every 10

# Resume a run
python exe_fashion.py --device cuda \
    --resume save/fashion_visuelle2_sweep_YYYYMMDD_HHMMSS
```

### Step 4: Evaluate RATD_Fashion

```bash
# Sweep all n_obs from a trained checkpoint
python exe_fashion.py \
    --modelfolder save/fashion_visuelle2_sweep_YYYYMMDD_HHMMSS \
    --sweep --nsample 50
```

### Step 5: Run Tier 0 baselines

```bash
python baselines/run_knn_mean.py --n_obs 0 \
    --output results/baselines/knn_mean_visuelle2.json
```

### Step 6: Run Tier 1 baselines

```bash
# LSTM-NoAttr sweep (train + eval for n_obs=1,2,3,4)
python baselines/lstm_baseline.py --mode sweep --device cuda

# LSTM-Attr sweep
python baselines/lstm_baseline.py --mode sweep --use_attr --device cuda
```

---

*This document covers all implementation work done on the project as of 2026-04-14.*
