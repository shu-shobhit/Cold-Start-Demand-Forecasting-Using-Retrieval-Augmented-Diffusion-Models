# Codebase Reference — Cold-Start Demand Forecasting with Retrieval-Augmented Diffusion Models

**Project:** BTP thesis adaptation of RATD (NeurIPS 2024) for New Product Introduction (NPI) cold-start
forecasting in fashion retail.  
**Primary dataset:** Visuelle 2.0 (Italian fashion, 12-week season, store-level sales).  
**Secondary dataset:** H&M Kaggle (planned Phase 2).

---

## Table of Contents

1. [Project Root — Core Modules](#project-root--core-modules)
2. [config/](#config)
3. [scripts/](#scripts)
4. [data/](#data)
5. [tests/smoke/](#testssmoke)
6. [TCN-master/](#tcn-master)
7. [Dependency Files](#dependency-files)

---

## Project Root — Core Modules

---

### `diff_models.py`

**Purpose:** The diffusion denoising backbone.  Defines the neural network that
predicts noise at each reverse-diffusion step.  Contains the custom
retrieval-aware attention mechanism (RMA) that fuses retrieved reference
trajectories with the latent state, and all supporting building blocks.
This file was adapted from the original RATD code to add product-attribute
conditioning (path b) via a new `attr_dim` parameter throughout the stack.

---

#### Class: `ReferenceModulatedCrossAttention`

Cross-attention module that fuses the latent denoising state with retrieved
future trajectories.  Uses asymmetric bidirectional attention (one softmax
along rows for the primary output, one along columns for the context output)
to let the denoising state and the reference mutually update each other.

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | int | Last dimension of the latent `x` tensor (`= ref_size + h_size`). |
| `heads` | int | Number of attention heads (default 8). |
| `dim_head` | int | Per-head hidden size (default 64). |
| `context_dim` | int | Last dimension of the reference tensor (`= ref_size * k`). |
| `dropout` | float | Dropout probability on attention weights (default 0). |
| `talking_heads` | bool | Whether to apply a learned 2-D conv across heads. |
| `prenorm` | bool | Whether to LayerNorm inputs before projection. |
| `attr_dim` | int or None | Dimensionality of the product attribute embedding.  `None` (default) disables path (b) and preserves original RATD behaviour. |
| `attr_proj_dim` | int | Hidden size the attribute embedding is projected to before concatenation (default 32). |

**Key sub-modules created by `__init__`:**

- `self.attr_proj` — `nn.Linear(attr_dim, attr_proj_dim)`, only when `attr_dim is not None`.
- `self.y_to_q` — Projects latent `x` into queries `(B*C, K, inner_dim)`.
- `self.cond_to_k` — Projects `[x, cond_info, reference, attr?]` into keys.  Input width = `2*dim + context_dim + attr_proj_dim` (or without attr_proj_dim when `attr_dim=None`).
- `self.ref_to_v` — Projects `[x, reference, attr?]` into values.
- `self.to_out` — Final linear mixing head outputs back to `dim`.
- `self.context_to_out` — Final linear for the reference context output.

**Methods:**

##### `forward(x, cond_info, reference, attr_emb=None, return_attn=False)`

Applies reference-aware attention.

| Argument | Shape | Description |
|----------|-------|-------------|
| `x` | `(B, C, K, L)` | Latent denoising state (B=batch, C=diffusion channels, K=series channels, L=time length). |
| `cond_info` | `(B, C, K, L)` | Projected side-information tensor. |
| `reference` | `(B, K, R)` | Retrieved reference tensor (`R = ref_size * k`). |
| `attr_emb` | `(B, attr_dim)` or None | Product attribute embedding.  When `attr_dim` is set and `attr_emb=None`, the projection is zero-filled (no attribute signal). |
| `return_attn` | bool | If True, returns `(out, context_out, attn, context_attn)` instead of just `out`. |

Returns: `(B*C, K, L)` tensor, or 4-tuple when `return_attn=True`.

---

#### Class: `DiffusionEmbedding`

Sinusoidal lookup table for discrete diffusion timesteps, with two learned
linear projection layers (SiLU activation between them).

**Constructor parameters:** `num_steps` (int), `embedding_dim` (int, default 128), `projection_dim` (int, optional).

**Methods:**

##### `forward(diffusion_step)`
Looks up the sinusoidal embedding for the given step indices, then applies the two linear projections.  
Input: integer tensor of step indices.  Output: `(B, projection_dim)`.

##### `_build_embedding(num_steps, dim=64)`
Constructs the fixed sinusoidal table `(num_steps, 2*dim)` using log-spaced frequencies.

---

#### Class: `ResidualBlock`

One residual denoising layer.  Combines:
- Diffusion timestep injection (additive).
- Side-information projection (1×1 conv).
- Optional reference fusion via `ReferenceModulatedCrossAttention` (fusion type 1) or a simpler additive sigmoid path (fusion type 2, legacy).
- Temporal mixing transformer (`time_layer`).
- Feature/channel mixing transformer (`feature_layer`).
- Gated activation + output projection splitting into residual + skip.

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `side_dim` | int | Channel count of the incoming side-information tensor. |
| `ref_size` | int | Single reference trajectory length (= pred_len). |
| `h_size` | int | Observed history length (set to 0 for cold-start). |
| `channels` | int | Internal hidden channel count. |
| `diffusion_embedding_dim` | int | Dimensionality of the timestep embedding. |
| `nheads` | int | Number of attention heads in time/feature transformers. |
| `is_linear` | bool | Use `LinearAttentionTransformer` instead of PyTorch standard attention. |
| `attr_dim` | int or None | Passed through to `ReferenceModulatedCrossAttention` for path (b). |

**Key sub-modules:** `diffusion_projection`, `cond_projection` (Conv1d), `mid_projection` (Conv1d), `output_projection` (Conv1d), `attn1` (CrossAttention — defined but currently unused, legacy dead code from original RATD), `RMA` (`ReferenceModulatedCrossAttention`), `time_layer`, `feature_layer`.

**Methods:**

##### `forward_time(y, base_shape)`
Mixes information along the temporal axis by reshaping to `(B*K, channel, L)` and passing through `time_layer`.

##### `forward_feature(y, base_shape)`
Mixes information across the feature/channel axis by reshaping to `(B*L, channel, K)` and passing through `feature_layer`.

##### `forward(x, cond_info, diffusion_emb, reference, attr_emb=None)`
Applies the full residual update.  
Input `x`: `(B, C, K, L)`.  
Returns: `(residual, skip)`, each `(B, C, K, L)`.

---

#### Class: `diff_RATD`

Top-level diffusion denoiser.  Manages the input/output projections and stacks
`N` `ResidualBlock` layers.  Aggregates all skip connections with a
`1/sqrt(N)` normalisation before the final projection to noise predictions.

**Constructor parameters:** `config` (dict), `inputdim` (int, default 2), `use_ref` (bool, default True).

Config keys consumed: `channels`, `num_steps`, `diffusion_embedding_dim`,
`side_dim`, `ref_size`, `h_size`, `nheads`, `is_linear`, `layers`,
`attr_dim` (optional, read via `config.get("attr_dim", None)`).

**Methods:**

##### `forward(x, cond_info, diffusion_step, reference=None, attr_emb=None)`
Runs the denoiser for one step.  
Input `x`: `(B, inputdim, K, L)`.  Output: `(B, K, L)` predicted noise.

---

#### Module-level helpers

| Function | Signature | Description |
|----------|-----------|-------------|
| `default` | `(val, d)` | Returns `val` if not None, else `d`. |
| `get_torch_trans` | `(heads, layers, channels)` | Builds a PyTorch `TransformerEncoder`. |
| `get_linear_trans` | `(heads, layers, channels, localheads, localwindow)` | Builds a `LinearAttentionTransformer`. |
| `Conv1d_with_init` | `(in_channels, out_channels, kernel_size)` | 1-D conv with Kaiming-normal weight init. |
| `Reference_Modulated_Attention` | `(in_channels, out_channels, kernel_size)` | Alias of `Conv1d_with_init` (legacy name). |

---

### `main_model.py`

**Purpose:** Task-agnostic RATD wrapper classes.  `RATD_base` owns the diffusion
schedule, masking utilities, side-information construction, DDPM loss, and
the reverse-diffusion sampler.  `RATD_Forecasting` extends it for the
electricity/benchmark forecasting task used by the original RATD paper.

---

#### Class: `RATD_base`

Base class shared by all RATD model variants.  Manages the full DDPM
machinery and exposes overridable hooks (`process_data`, `get_side_info`,
`calc_loss`) so subclasses can specialise the data pipeline without
reimplementing the core diffusion loop.

**Constructor parameters:** `target_dim` (int), `config` (dict), `device` (str).

**Key attributes set by `__init__`:**

- `self.emb_total_dim` — `timeemb + featureemb [+ 1 if conditional]`.
- `self.embed_layer` — `nn.Embedding(target_dim, featureemb)`.
- `self.diffmodel` — `diff_RATD` instance (may be rebuilt by subclasses).
- `self.alpha_torch` — Precomputed cumulative-product alpha schedule tensor.

**Methods:**

##### `time_embedding(pos, d_model=128)`
Builds sinusoidal positional encodings.  
Input: `(B, L)` time indices.  Output: `(B, L, d_model)`.

##### `get_randmask(observed_mask)`
Samples a random conditioning mask by keeping a per-sample fraction of
observed entries.  Used for the `random` target strategy during training.

##### `get_hist_mask(observed_mask, for_pattern_mask=None)`
Constructs a conditioning mask from the historical pattern of a neighbouring
sample.  Implements the `mix` strategy (randomly chooses between random and
historical masks per sample).

##### `get_test_pattern_mask(observed_mask, test_pattern_mask)`
Returns `observed_mask * test_pattern_mask`.  Used at test time to apply a
fixed evaluation mask pattern.

##### `get_side_info(observed_tp, cond_mask)`
Builds the auxiliary conditioning tensor `(B, emb_total_dim, K, L)` by
concatenating time embeddings, feature embeddings, and (if conditional) the
conditioning mask.

##### `calc_loss_valid(observed_data, cond_mask, observed_mask, side_info, is_train, reference=None)`
Averages `calc_loss` over all `num_steps` diffusion timesteps.  Used for
validation.

##### `calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, reference, set_t=-1)`
Core DDPM noise-prediction loss.  Corrupts `observed_data` with Gaussian
noise at step `t`, calls `self.diffmodel`, and returns the MSE on masked
positions.

##### `set_input_to_diffmodel(noisy_data, observed_data, cond_mask)`
Packages the denoiser input as `(B, 1, K, L)` (unconditional) or
`(B, 2, K, L)` (conditional: concat of observed context and noisy target).

##### `impute(observed_data, cond_mask, side_info, n_samples)`
Runs the full DDPM reverse process for `n_samples` draws.  Returns
`(B, n_samples, K, L)`.  **Note:** reference is not passed to the diffmodel
here — this is a known limitation in the base class, fixed in `RATD_Fashion`.

##### `forward(batch, is_train=1)`
Calls `process_data`, selects the masking strategy, builds `side_info`, and
returns the loss.

##### `evaluate(batch, n_samples)`
Calls `process_data`, applies `gt_mask`, builds `side_info`, calls `impute`,
and returns `(samples, observed_data, target_mask, observed_mask, observed_tp)`.

---

#### Class: `RATD_Forecasting(RATD_base)`

Forecasting-specific wrapper for the electricity/benchmark dataset.  Adds
feature subsampling (for very high-dimensional datasets) and wires the
reference tensor from the dataset batch into the loss and sampling calls.

**Constructor parameters:** `config` (dict), `device` (str), `target_dim` (int).

**Methods:**

##### `process_data(batch)`
Extracts tensors from a `Dataset_Electricity` batch dict, transposes
`(B, L, K)` → `(B, K, L)`, and assembles the reference tensor.  Returns a
7-tuple: `(observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length, reference)`.

##### `sample_features(observed_data, observed_mask, feature_id, gt_mask)`
Randomly sub-samples `num_sample_features` features per batch item.  Used
when `target_dim` is too large for full attention (e.g. 321-dim electricity).

##### `get_side_info(observed_tp, cond_mask)`
Overrides base to use `self.target_dim` (which may be reduced by feature
sampling) instead of `self.target_dim_base`.

##### `forward(batch, is_train=1)`
Unpacks the 7-tuple from `process_data` and calls `calc_loss` or
`calc_loss_valid` with the reference tensor.

##### `evaluate(batch, n_samples)`
**Known bug:** Unpacks only 6 items from the 7-tuple returned by
`process_data`, discarding the reference.  The `impute` call therefore never
receives the reference at evaluation time.  Fixed in `RATD_Fashion`.

---

### `main_model_fashion.py`

**Purpose:** `RATD_Fashion` — the cold-start forecasting wrapper for Visuelle 2.0.
Extends `RATD_base` with two complementary product-attribute conditioning paths:

- **Path (a) — side-info injection:** 513-dim CLIP+price embedding is projected
  to `attr_emb_dim` channels and broadcast spatially as extra side_info channels.
- **Path (b) — RMA key/value injection:** the same embedding is projected inside
  every `ReferenceModulatedCrossAttention` layer and concatenated to keys/values.

After calling `super().__init__()`, the class rebuilds `self.diffmodel` with the
correct `side_dim` (= base `emb_total_dim` + `attr_emb_dim`) and `attr_dim=513`.

---

#### Class: `RATD_Fashion(RATD_base)`

**Class constant:**  
- `ATTR_DIM = 513` — Dimensionality of the fused CLIP+price product embedding.

**Constructor parameters:** `config` (dict), `device` (str).

Config keys consumed beyond `RATD_base`:
- `model.attr_emb_dim` (int, default 16) — path (a) projection width.

**Key attributes:**
- `self._attr_side_dim` — projection width for path (a), from `config["model"]["attr_emb_dim"]`.
- `self.attr_proj_side` — `nn.Linear(513, attr_side_dim)` for path (a).
- `self._attr_emb` — Batch-scoped tensor `(B, 513)` set in `forward`/`evaluate`, consumed by `calc_loss`/`impute`.
- `self._reference` — Batch-scoped reference tensor set in `forward`/`evaluate`.

**Methods:**

##### `process_data(batch)`
Converts a `Dataset_Visuelle2` batch to model-ready tensors.  Transposes all
`(B, L, K)` tensors to `(B, K, L)`, transposes the reference from
`(B, 36, 2)` to `(B, 2, 36)`, and extracts `product_emb` `(B, 513)`.  
Returns an 8-tuple: `(observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length, reference, product_emb)`.

##### `get_side_info(observed_tp, cond_mask, product_emb=None)`
Builds the full side-information tensor including path (a) attribute channels.

| Step | Operation | Output shape |
|------|-----------|--------------|
| 1 | Time + feature embeddings | `(B, emb_total_dim, K, L)` |
| 2 | Append conditioning mask | `(B, emb_total_dim+1, K, L)` |
| 3 | Project `product_emb` via `attr_proj_side`, expand, cat | `(B, side_dim, K, L)` |

If `product_emb=None`, step 3 is skipped and the output is `(B, emb_total_dim, K, L)`.

##### `calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, reference, set_t=-1)`
Overrides `RATD_base.calc_loss` to forward `attr_emb=self._attr_emb` to the
diffmodel, activating path (b) in every residual block's RMA.

##### `impute(observed_data, cond_mask, side_info, n_samples)`
Overrides `RATD_base.impute` to pass `reference=self._reference` and
`attr_emb=self._attr_emb` at every reverse-diffusion step.  Only supports
`is_unconditional=False` (the fashion use case is always conditional).

##### `forward(batch, is_train=1)`
Sets `self._attr_emb` and `self._reference` from `process_data`, calls
`get_side_info` with `product_emb`, then delegates to `calc_loss` /
`calc_loss_valid`.  Uses `gt_mask` directly as `cond_mask` (no random
masking — the dataset handles n_obs randomisation per sample).

##### `evaluate(batch, n_samples)`
Sets `self._attr_emb` and `self._reference`, calls `get_side_info` and
`impute`.  Returns `(samples, observed_data, target_mask, observed_mask, observed_tp)`.

---

### `dataset_forecasting.py`

**Purpose:** Electricity time-series dataset for the original RATD benchmark
experiments.  Implements a sliding-window approach over a standardised
electricity CSV file and loads precomputed TCN-based retrieval indices.

---

#### Class: `Dataset_Electricity(Dataset)`

Sliding-window electricity forecasting dataset.

**Constructor parameters:** `root_path`, `flag` (`train`/`val`/`test`), `size` (`[seq_len, label_len, pred_len, dim]`), `features`, `data_path`, `target`, `scale` (bool), `timeenc`, `freq`.

**Methods:**

##### `__read_data__()`
Reads the CSV, splits 70/10/20 (train/val/test), fits `StandardScaler` on
train, and loads the precomputed retrieval index from
`./dataset/TCN/ele_idx_list.pt`.

##### `__getitem__(index)`
Assembles one sample: extracts the history+horizon window, stacks the top-3
retrieved future segments into a `(3*pred_len, dim)` reference tensor, and
returns a dict with keys `observed_data`, `observed_mask`, `gt_mask`,
`timepoints`, `feature_id`, `reference`.

##### `__len__()`
Returns `len(data_x) - seq_len - pred_len + 1`.

##### `inverse_transform(data)`
Applies `scaler.inverse_transform` to un-normalise predictions.

#### Function: `get_dataloader(device, batch_size=8)`
Creates train/val/test `DataLoader`s using hardcoded paths from the original
research environment.  Returns `(train_loader, valid_loader, test_loader)`.

---

### `dataset_visuelle2.py`

**Purpose:** Visuelle 2.0 store-level sales forecasting dataset.  Each sample
is one `(product, store)` pair for a 12-week fashion season.  Loads CSV sales
data, precomputed CLIP+price product embeddings, and FAISS-retrieved reference
trajectories.  Implements per-channel `StandardScaler` normalisation fit on
the training split.

---

#### Class: `Dataset_Visuelle2(Dataset)`

**Class constants:**
- `SALES_COLS = ['0', '1', ..., '11']` — Column names for the 12 weekly sales figures.
- `DISC_COLS = ['d_0', ..., 'd_11']` — Discount ratio column names.
- `PRED_LEN = 12` — Full season forecast horizon.
- `K_CHANNELS = 2` — Number of time-series channels (sales + discount).
- `K_REFS = 3` — Number of retrieved reference trajectories per product.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_root` | str | `/home/shu_sho_bhit/BTP_2` | Root containing `visuelle2/`. |
| `processed_dir` | str | `.../visuelle2_processed` | Directory with precomputed `.pt` files. |
| `flag` | str | `"train"` | `"train"` or `"test"`. |
| `n_obs` | int or None | None | Fixed observation count for evaluation; None = random. |
| `n_obs_range` | tuple | `(0, 4)` | Inclusive range for training-time random n_obs sampling. |
| `scale` | bool | True | Apply per-channel `StandardScaler`. |

**Methods:**

##### `_load_data()`
Orchestrates the full data loading pipeline:
1. Reads `stfore_train.csv` and `stfore_test.csv`.
2. Merges discount series from `price_discount_series.csv`.
3. Fits (or loads) `StandardScaler` → saves to `scaler.pkl`.
4. Calls `_build_tensor` to normalise and stack into `(N, 12, 2)`.
5. Loads `product_embeddings.pt` → `dict[code → Tensor(513)]`.
6. Loads `train_references.pt` or `test_references.pt` → `dict[code → Tensor(2, 36)]`.

##### `_build_tensor(df)`
Stacks sales and discount columns into `(N, 12, 2)`, applies the fitted
scaler.  
Input: DataFrame with `SALES_COLS` and `DISC_COLS`.  Output: `ndarray (N, 12, 2)`.

##### `__len__()`
Returns the number of `(product, store)` rows in the active split.

##### `__getitem__(index)`
Returns a dict with keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `observed_data` | `(12, 2)` | Normalised sales + discount time series. |
| `observed_mask` | `(12, 2)` | All ones (no structural missingness). |
| `gt_mask` | `(12, 2)` | First `n_obs` rows = 1, rest = 0. |
| `timepoints` | `(12,)` | Week indices `[0, 1, ..., 11]`. |
| `feature_id` | `(2,)` | Channel indices `[0, 1]`. |
| `reference` | `(36, 2)` | Concatenated mean trajectories of 3 retrieved neighbours, shape `(k*pred_len, K)`. |
| `product_emb` | `(513,)` | CLIP image+text+price product embedding. |
| `external_code` | int | Product identifier. |
| `retail` | int | Store identifier. |

##### `inverse_transform_sales(data)`
Un-scales the sales channel (channel 0) only, using the fitted
`StandardScaler`.  Accepts any leading shape `(..., 12)`.  
**Note:** multiply by 53.0 after this to approximate integer unit counts.

#### Function: `get_dataloader(...)`
Factory that creates train and test `DataLoader`s.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_root` | `/home/shu_sho_bhit/BTP_2` | Raw data root. |
| `processed_dir` | `.../visuelle2_processed` | Precomputed files. |
| `batch_size` | 16 | Mini-batch size. |
| `n_obs_train` | `(0, 4)` | Random n_obs range for training. |
| `n_obs_eval` | 2 | Fixed n_obs at test time. |
| `num_workers` | 4 | DataLoader worker count. |

Returns: `(train_loader, test_loader, train_dataset, test_dataset)`.

---

### `utils.py`

**Purpose:** Training loop, checkpoint saving, and evaluation pipeline shared
across all model variants.  Also provides probabilistic forecast metrics
(CRPS, CRPS-sum) and the WAPE metric added for the fashion experiment.

---

#### Function: `train(model, config, train_loader, valid_loader=None, valid_epoch_interval=20, foldername="")`

Trains the model for `config["epochs"]` epochs with Adam optimiser and a
two-milestone MultiStepLR schedule (75% and 90% of total epochs).  Saves
`model.pth` to `foldername` every `valid_epoch_interval` epochs and at the
end of training.  Validation loss computation is commented out in the current
snapshot (saves unconditionally instead of on best loss).

#### Function: `quantile_loss(target, forecast, q, eval_points)`
Standard pinball/quantile loss.  
Returns: `2 * sum(|f - t| * eval_points * (1{t ≤ f} - q))`.

#### Function: `calc_denominator(target, eval_points)`
Returns `sum(|target * eval_points|)` — the normalisation denominator for
CRPS.

#### Function: `calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler)`
Computes CRPS by averaging the normalised quantile loss over a 19-point grid
`[0.05, 0.10, ..., 0.95]`.  De-normalises inputs using `scaler` and
`mean_scaler` before computation.

#### Function: `calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler)`
Same as `calc_quantile_CRPS` but sums forecasts across the feature axis
before computing the quantile loss.  Measures distributional accuracy of
the aggregate demand.

#### Function: `calc_wape(target, forecast_median, eval_points, scaler=1, mean_scaler=0)`
**(Added for the fashion experiment.)**  
Weighted Absolute Percentage Error: `sum(|pred - actual|) / sum(|actual|)`.  
De-normalises before computing.  Returns a fraction (multiply by 100 for %).
WAPE is scale-invariant, so the `scaler` argument does not change the result
for a ratio metric — it is included for API consistency.

#### Function: `evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="")`
Runs full probabilistic evaluation over the test split:
- Calls `model.evaluate(batch, nsample)` per batch.
- Computes batch-wise RMSE and MAE from the per-sample median forecast.
- Aggregates all samples and targets, then computes CRPS and CRPS-sum.
- Saves generated samples to `generated_outputs_nsample{N}.pk` and metrics
  to `result_nsample{N}.pk` in `foldername`.
- Prints RMSE, MAE, CRPS, and CRPS-sum.

---

### `exe_forecasting.py`

**Purpose:** Original RATD training/evaluation entrypoint for the electricity
benchmark.  Inherited from the upstream research code with minor
documentation improvements.  Contains hardcoded absolute paths from the
authors' HPC environment — **not portable** without modification.

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `base_forecasting.yaml` | YAML config filename (looked up under a hardcoded HPC path). |
| `--datatype` | `electricity` | Dataset tag used to name the output folder. |
| `--device` | `cuda:5` | Hardcoded GPU index from the authors' environment. |
| `--seed` | 1 | Random seed. |
| `--unconditional` | flag | If set, uses unconditional diffusion. |
| `--modelfolder` | `""` | If non-empty, loads a saved checkpoint instead of training. |
| `--nsample` | 100 | Number of reverse-diffusion samples at evaluation. |
| `--target_dim` | 321 | Number of time-series channels (electricity dataset). |
| `--h_size` | 96 | History window length. |
| `--ref_size` | 168 | Prediction horizon length. |

**Flow:** loads YAML → creates output folder → builds electricity DataLoaders → instantiates `RATD_Forecasting` → trains or loads checkpoint → calls `utils.evaluate`.

---

### `exe_fashion.py`

**Purpose:** Training and evaluation entrypoint for the cold-start fashion
forecasting experiment (Visuelle 2.0).  Designed to be portable — all paths
are CLI arguments with sensible defaults, and no HPC-specific hardcoding.

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/visuelle2.yaml` | Path to YAML config (relative to repo root). |
| `--data_root` | `/home/shu_sho_bhit/BTP_2` | Root containing `visuelle2/`. |
| `--processed_dir` | `.../visuelle2_processed` | Directory with `.pt` precomputed files. |
| `--device` | auto (cuda/cpu) | Torch device. |
| `--seed` | 42 | Random seed. |
| `--n_obs_eval` | 0 | Weeks of history revealed at test time (0 = cold-start, 1–4 = few-shot). |
| `--n_obs_min` | 0 | Min n_obs sampled during training. |
| `--n_obs_max` | 4 | Max n_obs sampled during training. |
| `--nsample` | 50 | Number of forecast samples per test batch. |
| `--modelfolder` | `""` | Skip training, load `model.pth` from this folder. |
| `--num_workers` | 4 | DataLoader worker processes. |

**Key functions:**

#### Function: `parse_args()`
Declares and parses all CLI arguments listed above.

#### Function: `evaluate_fashion(model, test_loader, nsample, scaler, mean_scaler, foldername)`
Fashion-specific evaluation that computes metrics on the **sales channel
only** (channel index 0).  Metrics reported: RMSE, MAE, WAPE, CRPS.
Saves generated samples and metric JSON to `foldername`.

#### `main()`
Full pipeline: load config → set seeds → create output folder → build
DataLoaders → instantiate `RATD_Fashion` → train or load checkpoint →
call `evaluate_fashion`.

---

### `download.py`

**Purpose:** Utility inherited from the upstream CSDI-style codebase for
downloading benchmark datasets (PhysioNet and PM2.5).  Not used in the
Visuelle 2 / H&M pipeline but retained for reproducibility of related
experiments.

**Usage:** `python download.py physio` or `python download.py pm25`.

**Functions:**

#### Function: `create_normalizer_pm25()` (nested inside `__main__`)
Reads the PM2.5 ground truth CSV, excludes test months (3, 6, 9, 12), and
pickles `[mean, std]` to `./data/pm25/pm25_meanstd.pk`.

---

## config/

Configuration YAML files loaded by the entrypoint scripts.  All are passed
through `yaml.safe_load` and their keys are documented below.

---

### `config/base.yaml`

**Purpose:** Base configuration for the original RATD imputation task (not
forecasting-specific).

| Section | Key | Value | Description |
|---------|-----|-------|-------------|
| train | epochs | 200 | Training epochs. |
| train | batch_size | 16 | Mini-batch size. |
| train | lr | 1e-3 | Adam learning rate. |
| train | itr_per_epoch | 1e8 | Max batches per epoch (effectively unlimited). |
| diffusion | layers | 4 | Number of `ResidualBlock` layers. |
| diffusion | channels | 64 | Internal hidden channels. |
| diffusion | nheads | 8 | Attention heads. |
| diffusion | diffusion_embedding_dim | 128 | Timestep embedding size. |
| diffusion | beta_start | 0.0001 | Start of noise schedule. |
| diffusion | beta_end | 0.5 | End of noise schedule. |
| diffusion | num_steps | 50 | Diffusion steps. |
| diffusion | schedule | `"quad"` | Noise schedule type (`quad` or `linear`). |
| diffusion | is_linear | False | Use linear attention. |
| model | is_unconditional | 0 | 0 = conditional DDPM. |
| model | timeemb | 128 | Time embedding size. |
| model | featureemb | 16 | Feature embedding size. |
| model | target_strategy | `"random"` | Masking strategy during training. |

---

### `config/base_forecasting.yaml`

**Purpose:** Configuration for the original electricity forecasting experiments.
Differs from `base.yaml` in batch size, learning rate, target strategy, and
is_linear.  Does **not** include `h_size` or `ref_size` — these are injected
as CLI arguments in `exe_forecasting.py`.

| Section | Key | Value | Description |
|---------|-----|-------|-------------|
| train | epochs | 100 | Training epochs. |
| train | batch_size | 1 | Mini-batch size (tiny — electricity dataset is large). |
| train | lr | 3e-4 | Adam learning rate. |
| diffusion | is_linear | True | Use `LinearAttentionTransformer` for efficiency. |
| model | target_strategy | `"test"` | Uses gt_mask directly (no random masking). |
| model | num_sample_features | 64 | Features sub-sampled per batch. |
| model | use_reference | True | Enable retrieval augmentation. |

---

### `config/visuelle2.yaml`

**Purpose:** Full configuration for the `RATD_Fashion` cold-start / few-shot
fashion forecasting experiment on Visuelle 2.0.

**Key design constants:**
- `ref_size = 12` (= pred_len, single reference trajectory length)
- `h_size = 0` (n_obs variation is handled by `gt_mask`, not a fixed history window)
- `ref_size * 3 = 36` = total reference tensor length per sample (k=3 neighbours)
- `attr_emb_dim = 16` = path (a) attribute projection width

| Section | Key | Value | Description |
|---------|-----|-------|-------------|
| train | epochs | 100 | Training epochs. |
| train | batch_size | 32 | Mini-batch size. |
| train | lr | 1e-3 | Adam learning rate. |
| diffusion | layers | 4 | ResidualBlock layers. |
| diffusion | channels | 64 | Hidden channels. |
| diffusion | nheads | 8 | Attention heads. |
| diffusion | diffusion_embedding_dim | 128 | Timestep embedding size. |
| diffusion | num_steps | 50 | Diffusion steps. |
| diffusion | schedule | `"quad"` | Noise schedule. |
| diffusion | is_linear | False | Full transformer attention. |
| diffusion | h_size | 0 | No separate fixed history window. |
| diffusion | ref_size | 12 | Single reference length = pred_len. |
| model | is_unconditional | 0 | Conditional diffusion. |
| model | timeemb | 128 | Time embedding dimension. |
| model | featureemb | 16 | Feature embedding dimension. |
| model | target_strategy | `"test"` | Use gt_mask as conditioning. |
| model | use_reference | True | Enable retrieval augmentation. |
| model | attr_emb_dim | 16 | Path (a) attribute projection width. |

**Keys injected by `RATD_Fashion.__init__`** (not in the file, added at runtime):
- `diffusion.side_dim` = 145 + 16 = 161
- `diffusion.attr_dim` = 513

---

## scripts/

Offline preprocessing scripts that must be run once before training.
Both scripts write their outputs to `visuelle2_processed/`.

---

### `scripts/compute_embeddings.py`

**Purpose:** Computes CLIP-based product attribute embeddings for all 5,355
unique products in Visuelle 2.0 (train + test).  Uses
`openai/clip-vit-base-patch32` via HuggingFace `transformers`.

**Pipeline:**
1. Load product metadata (category, color, fabric, image path, mean price).
2. Load CLIP model with `use_safetensors=True` (bypasses torch < 2.6 CVE check).
3. Encode product images through `model.vision_model` + `model.visual_projection`.
4. Encode tag text (`"a {category}, {color} color, {fabric} fabric"`) through `model.text_model` + `model.text_projection`.
5. Fuse: `fused = L2_norm(image_emb + tag_emb)` (element-wise sum then re-normalise).
6. Append price scalar: `product_emb = concat(fused, price)` → shape `(N, 513)`.

**CLI arguments:** `--dataset`, `--data_root`, `--out_dir`, `--batch_size`, `--device`, `--clip_model`.

**Output files** (all in `out_dir`):

| File | Shape | Description |
|------|-------|-------------|
| `image_embeddings.pt` | `dict[int → Tensor(512)]` | L2-normalised CLIP image features. |
| `tag_embeddings.pt` | `dict[int → Tensor(512)]` | L2-normalised CLIP text features. |
| `price_scalars.pt` | `dict[int → float]` | Mean price across stores. |
| `product_embeddings.pt` | `dict[int → Tensor(513)]` | Fused embedding used by the model. |

**Functions:**

#### `parse_args()`
Declares CLI arguments with defaults pointing to the BTP_2 directory structure.

#### `load_visuelle2_products(data_root)`
Loads train + test CSVs, deduplicates to one row per `external_code`, and
merges mean prices from `price_discount_series.csv`.  Returns a DataFrame.

#### `build_tag_text(row)`
Returns `f"a {row['category']}, {row['color']} color, {row['fabric']} fabric"`.

#### `encode_images(image_paths, processor, model, device, batch_size)`
Batch-encodes images through CLIP vision encoder.  Handles missing/corrupt
images by substituting blank 224×224 RGB images.  Returns `(N, 512)` L2-
normalised CPU tensor.

#### `encode_texts(texts, processor, model, device, batch_size)`
Batch-encodes text strings through CLIP text encoder.  Returns `(N, 512)` L2-
normalised CPU tensor.

#### `main()`
Orchestrates the 5-step pipeline described above.

---

### `scripts/compute_retrieval.py`

**Purpose:** Builds a FAISS `IndexFlatIP` (cosine similarity via inner product
on L2-normalised embeddings) over the 5,109 training product embeddings, then
precomputes top-3 nearest-neighbour reference trajectories for every product
in both the train and test splits.

**Retrieval design:**
- Index: training products only (no test leakage).
- Train queries: exclude self (the product is in the index).
- Test queries: no self-exclusion (test products are not in the index).
- Reference trajectory: mean sales+discount across all stores for the neighbour
  product, shape `(2, 12)` per neighbour.  Three neighbours concatenated → `(2, 36)`.

**CLI arguments:** `--dataset`, `--data_root`, `--emb_dir`, `--out_dir`, `--k`.

**Output files** (all in `out_dir`):

| File | Content | Description |
|------|---------|-------------|
| `train_ref_indices.pt` | `dict[int → List[int]]` | k neighbour codes per train product. |
| `test_ref_indices.pt` | `dict[int → List[int]]` | k neighbour codes per test product. |
| `train_references.pt` | `dict[int → Tensor(2, 36)]` | Concatenated mean trajectories for train. |
| `test_references.pt` | `dict[int → Tensor(2, 36)]` | Concatenated mean trajectories for test. |

**Functions:**

#### `parse_args()`
Declares CLI arguments.

#### `load_visuelle2_splits(data_root)`
Loads train/test CSVs and merges discount columns.  Returns `(train_df, test_df, sales_cols, disc_cols)`.

#### `build_mean_trajectories(train_df, sales_cols, disc_cols)`
Groups train rows by `external_code` and averages sales and discount across
stores.  Returns `dict[int → Tensor(2, 12)]`.

#### `build_reference_tensors(codes, ref_indices, mean_traj, k)`
For each product code, concatenates the `k` neighbour trajectories along
dim=1.  Returns `dict[int → Tensor(2, k*12)]`.

#### `retrieve_neighbours(query_codes, train_codes, product_emb, k, exclude_self=True)`
Builds the FAISS index, L2-normalises the index and query matrices, runs
`index.search`, and optionally filters out the query product from its own
result list.  Returns `dict[int → List[int]]` of length k per query.

#### `main()`
Orchestrates the 5-step pipeline.

---

## data/

Static data directories.  These files are read by the original
`dataset_forecasting.py` and are not used in the fashion experiment.

---

### `data/electricity_nips/`

| File | Description |
|------|-------------|
| `data.pkl` | Pickled electricity consumption time series (NIPS format). |
| `meanstd.pkl` | Pickled `[mean, std]` normalisation statistics for the NIPS electricity dataset. |

---

### `data/ts2vec/`

| File | Description |
|------|-------------|
| `electricity.csv` | Electricity consumption CSV with a `date` index column and 321 consumer columns. Used by `Dataset_Electricity` via the hardcoded path in `dataset_forecasting.py`. |

---

## tests/smoke/

Smoke test scripts for validating each module without requiring a full
training run.  Each file is a standalone Python script that prints PASS/FAIL
and exits with code 0 on success, 1 on failure.

---

### `tests/smoke/test_diff_models.py`

Tests `diff_models.py`: RMA forward pass with/without `attr_emb`, ResidualBlock forward pass, `diff_RATD` end-to-end, backward gradient flow, and backward compatibility when `attr_dim=None`.

---

### `tests/smoke/test_main_model_fashion.py`

Tests `main_model_fashion.py`: model construction (checks `side_dim` and `attr_dim` mutation), `get_side_info` channel count, `process_data` shape transposition, `gt_mask` masking for all n_obs values, forward (train + val), `evaluate` output shapes, cold-start target mask coverage, and construction from the real `config/visuelle2.yaml`.

---

### `tests/smoke/test_utils.py`

Tests `utils.py`: `quantile_loss` (perfect forecast = 0, analytical value), `calc_denominator`, `calc_wape` (perfect = 0, predict-zeros = 1.0, scale-invariance), `calc_quantile_CRPS` non-negativity, `calc_quantile_CRPS_sum` non-negativity, `train` (one epoch, checkpoint saved), `evaluate` (end-to-end run, output files written).

---

### `tests/smoke/test_config.py`

Tests `config/*.yaml`: all three files load without error; `visuelle2.yaml` has all required keys in correct types and ranges; `RATD_Fashion` builds cleanly from `visuelle2.yaml`; `ref_size * 3 = 36`.

---

### `tests/smoke/test_dataset_visuelle2.py`

Tests `dataset_visuelle2.py`.  **Auto-skips with exit code 0 if any required
data file is absent.**  When data is present: validates `__getitem__` schema
(keys, shapes, dtypes), n_obs masking for 0 and 4, observed_mask = all-ones,
timepoints range, reference finiteness, product_emb shape, DataLoader batch
keys, and `inverse_transform_sales` round-trip.

---

### `tests/smoke/run_all.py`

Master runner that executes each test module as a subprocess and prints a
PASS/FAIL summary table.  Exit code 0 if all pass, 1 if any fail.

**Usage:** `conda run -n ML python tests/smoke/run_all.py`

---

## TCN-master/

**Purpose:** External TCN (Temporal Convolutional Network) encoder used by the
**original** RATD paper to compute retrieval indices for the electricity
dataset.  This component is **not used** in the fashion cold-start pipeline —
it is replaced by FAISS + CLIP embeddings (`scripts/compute_retrieval.py`).
The directory is retained from the upstream research snapshot for
reproducibility of the electricity experiments.

Relevant entry point (original usage only):
```shell
python retrieval.py --type encode
python retrieval.py --type retrieval
```

---

## Dependency Files

---

### `requirements.txt`

Original RATD dependencies:

```
torch, pandas, numpy, tqdm, pyyaml, matplotlib, wget, linear_attention_transformer
```

---

### `requirements_cold_start.txt`

Additional packages for the cold-start fashion extension:

| Package | Purpose |
|---------|---------|
| `faiss-gpu-cu12` | FAISS nearest-neighbour index with CUDA 12 support (used in `compute_retrieval.py`). |
| `scikit-learn` | `StandardScaler` for per-channel normalisation in `Dataset_Visuelle2`. |
| `Pillow` | Image loading in `compute_embeddings.py`. |
| `transformers` | HuggingFace `CLIPModel` and `CLIPProcessor` (v5.x API used: `model.vision_model`, `model.visual_projection`). |

---

*End of codebase reference.*
