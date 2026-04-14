# Architectural Improvement Suggestions

## Overview

This document analyses the current RATD-Fashion architecture and identifies the most
technically significant bottlenecks that limit forecasting performance on the cold-start
demand forecasting task. Each suggestion is grounded in specific code-level observations.
Improvements are roughly ordered by expected impact.

---

## 1. Attribute Embedding Bottleneck: `attr_proj_dim=32`

**Where:** `diff_models.py`, `ReferenceModulatedCrossAttention.__init__`, line 98.
`ReferenceModulatedCrossAttention` is constructed in `ResidualBlock.__init__` with the
default `attr_proj_dim=32`.

**Problem:** The 513-dim CLIP product embedding (512-dim ViT-L/14 joint image+text
representation plus 1 price scalar) is compressed to 32 dimensions before being
concatenated into the keys and values of the cross-attention. This is a 94% reduction in
dimensionality. CLIP ViT-L/14 embeddings are trained to preserve fine-grained semantic
differences across color, material, category, silhouette, and style. A projection to 32
dimensions must discard the vast majority of this discriminative information before the
model ever sees it. The product embedding is the *only* product-specific signal available
at inference in the pure cold-start setting (n_obs=0), so compressing it heavily is a
particularly costly choice at exactly the moment the model needs it most.

**Suggested fix:** Increase `attr_proj_dim` to at least 128 or 256. At `attr_proj_dim=128`
the attribute projection adds 128 to the key/value input width (currently 92 = 12+12+36+32),
raising it to 188. The projection linear layer itself grows from 513x32 to 513x128, adding
roughly 64K parameters at trivial compute cost. A multi-layer projection (MLP with one
hidden layer and GELU activation) instead of a single linear layer would help form more
discriminative non-linear feature combinations before the dimensionality bottleneck.

---

## 2. `dim_feedforward=64` in Temporal and Feature Transformers

**Where:** `diff_models.py`, `get_torch_trans`, line 208:
```python
nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, ...)
```
The channel dimension is also 64, so `dim_feedforward == d_model`.

**Problem:** Standard transformer design convention (and the original "Attention is All
You Need" paper) sets `dim_feedforward = 4 * d_model`. With `d_model=channels=64`, the
standard choice would be 256. The FFN is the main capacity-expanding component of a
transformer block. When `dim_feedforward == d_model`, the FFN collapses to an
expansion-free bottleneck: it has no wider intermediate space to form non-linear
combinations, effectively making it no more expressive than a second attention layer of
the same width. This constrains the time and feature mixing transformers in every
`ResidualBlock` to be severely underparameterized.

**Suggested fix:** Set `dim_feedforward=256` (4x the model dimension). This adds
approximately 128K parameters per residual block (two blocks: time_layer + feature_layer),
a 4-block total increase of roughly 1M parameters, which remains well within the budget
of a moderately sized model. If total parameter count is a constraint, increasing only
the `time_layer` FFN (temporal mixing) is more impactful than the `feature_layer` FFN
because K=2 features are too few for the feature transformer to be a bottleneck.

---

## 3. Non-Differentiable Retrieval Prevents End-to-End Learning

**Where:** The retrieval pipeline (`compute_embeddings.py`, FAISS k-NN search) runs at
preprocessing time. The resulting `references` tensors are stored in `.pt` files and
loaded by `Dataset_Visuelle2`. The diffusion model never sees retrieval scores, cannot
adjust retrieval quality, and the FAISS index is frozen throughout training.

**Problem:** The retrieval quality depends entirely on the geometric structure of the CLIP
embedding space, which was not optimized for sales trajectory similarity. Two products may
be semantically close in CLIP space (similar garments) but have very different demand
trajectories due to price-point differences, store placements, or timing. Because the
retrieval is fixed and non-differentiable, the model cannot learn to de-weight unhelpful
references or up-weight informative ones at the retrieval stage. It can only learn to
ignore bad references post-hoc through the RMA attention weights.

**Suggested improvement:** A differentiable reranking layer between the precomputed
retrieval and the diffusion model could help significantly. Concretely, retrieve a larger
candidate set (e.g., k=10) at preprocessing, then learn a lightweight attention-based
selector inside the model that scores each candidate trajectory against the query product
embedding and produces a soft weighted combination before passing it to RMA. The selector
is end-to-end differentiable with respect to the downstream DDPM loss, so it learns which
attributes of the retrieved neighbors actually correlate with future sales patterns.

---

## 4. No Explicit Observation-Count Conditioning

**Where:** `main_model_fashion.py`, `get_side_info`. The conditioning mask (gt_mask) is
included as a channel in the side information, but `n_obs` (the number of observed weeks,
0 to 4) is not embedded as a scalar.

**Problem:** The model faces qualitatively different problems depending on n_obs. At
n_obs=0, it must rely entirely on product attributes and retrieved references (pure
cold-start). At n_obs=4, it has a 4-week ramp-up trajectory and should extrapolate with
lower uncertainty. These are different inference modes, but the model has no dedicated
signal to distinguish them beyond the pattern of zeros and ones in the mask. The
transformer layers must infer the observation count implicitly by inspecting the mask,
which requires the model to solve an unnecessary auxiliary task.

**Suggested improvement:** Embed n_obs as a learned scalar embedding (analogous to the
diffusion step embedding) and broadcast it into the side_info tensor alongside the time
and feature embeddings. This is a simple change: a single `nn.Embedding(5, emb_dim)`
lookup (5 values for n_obs in {0,1,2,3,4}) summed into the time embedding. The explicit
signal lets the denoiser calibrate its noise predictions differently for zero-shot versus
few-shot inputs without having to discover this structure from the mask pattern alone.

---

## 5. Dead `attn1` Module in `ResidualBlock`

**Where:** `diff_models.py`, `ResidualBlock.__init__`, lines 481-487:
```python
self.attn1 = CrossAttention(
    query_dim=nheads*dim_heads, heads=nheads, dim_head=dim_heads, dropout=0, bias=False,
)
```
This module is initialized in every `ResidualBlock` but is never called anywhere in
`ResidualBlock.forward`.

**Problem:** `CrossAttention` from the `diffusers` library is a full multi-head attention
module with Q, K, V, and output projections. With `nheads=8` and `dim_head=8` (implied
by `dim_heads=8` at line 478), it has query/key/value dimensions of 64, adding roughly
16K parameters per block that contribute zero computation and zero gradient signal. More
importantly, the presence of this module creates the appearance of additional attention
capacity that does not exist at forward pass time. If a second attention path was
originally intended here (e.g., self-attention on the latent state before RMA), it should
either be wired up or removed to reduce parameter count confusion.

**Suggested fix:** Either remove `self.attn1` from all residual blocks and reclaim its
parameters for useful capacity elsewhere, or wire it as a proper self-attention on `y`
before the RMA call (which would add a legitimate latent-state self-attention pass missing
from the current design).

---

## 6. Unused `context_out` Path in RMA (Wasted Computation)

**Where:** `diff_models.py`, `ReferenceModulatedCrossAttention.forward`, lines 179-191.

`context_out` is computed via `context_attn = sim.softmax(dim=-2)` and the matrix
product `einsum('b h j i, b h j d -> b h i d', context_attn, cond)`. The linear layer
`self.context_to_out` and conv layer `self.context_talking_heads` are also initialized
for this path. However, when `return_attn=False` (the default during training and
evaluation), `context_out` is computed but immediately discarded: the function returns
only `out`.

**Problem:** The forward-pass computation of `context_attn` and `context_out` is
non-trivial (another `einsum` over the same `(B*C, H, K, K)` attention matrix) and runs
on every batch. The `self.context_to_out` and `self.context_talking_heads` parameters are
never updated by gradient descent because no gradient flows to `context_out`. In the
original bidirectional cross-attention design, `context_out` represents how the reference
trajectories should be updated given the current latent state. If this path is not used,
the symmetry of the design is lost and computation is wasted.

**Suggested fix:** Either (a) use `context_out` by residually adding it to the reference
tensor and passing the updated reference to the next residual block (realizing the
intended bidirectional update semantics), or (b) remove the `context_attn`, `context_out`,
`context_to_out`, and `context_talking_heads` computations from the non-`return_attn` path
to eliminate wasted forward-pass compute. Option (a) is architecturally more interesting:
it would allow references to be progressively refined across the 4 residual blocks rather
than being used as a static conditioning signal at every layer.

---

## 7. Discount Channel Treated as a Target Variable

**Where:** `config/visuelle2.yaml` (`target_dim=2`), `Dataset_Visuelle2` (K_CHANNELS=2),
`diff_models.py` (`diff_RATD.forward` over all K channels).

**Problem:** The model diffuses jointly over sales (channel 0) and discount (channel 1).
Discount rates are retailer decisions, not random variables to be forecasted. At inference
time (n_obs>0), the discounts for the observed weeks are available in the gt_mask-conditioned
data, and future discounts are either known (planned promotions) or assumed zero. Diffusing
over discount introduces an unnecessary degree of freedom that the DDPM must model. The
model spends capacity learning the distribution of discount values, which is not useful
for the downstream sales RMSE evaluation metric. Further, the noise-prediction loss
`(noise - predicted) * target_mask` averages over both channels, so discount prediction
errors dilute the sales-specific gradient signal.

**Suggested improvement:** Treat discount as a known covariate rather than a diffusion
target. Concretely: keep discount in `side_info` (broadcast it alongside the time and
feature embeddings) so the denoiser can condition on it, but set `target_dim=1` so
diffusion is performed only over the sales channel. This halves the forecast output
dimension and concentrates the entire DDPM loss on the channel that matters for evaluation.

---

## 8. Reference Concatenation Has No Boundary Markers

**Where:** `dataset_visuelle2.py` (reference tensor is `(36, 2)` = 3 references
concatenated along the sequence axis), `diff_models.py`, `RMA` (context_dim = `ref_size*3`
= 36).

**Problem:** The 3 retrieved reference trajectories are concatenated into a flat 36-length
sequence. Inside RMA, attention is computed over this flat sequence without any signal
distinguishing where reference 1 ends and reference 2 begins. The attention mechanism
must discover this structure from correlations in the data. Adding a reference-index
embedding (a learned `nn.Embedding(3, emb_dim)` repeated 12 times each) would let the
model explicitly distinguish "this part of the context is from reference 1 vs reference 2"
and assign different weights to different retrieved neighbors. This is analogous to
positional embeddings in transformers but for reference identity rather than time position.

---

## 9. Product-Level References Ignore Store Effects

**Where:** `compute_embeddings.py` / preprocessing (reference trajectories are means over
the full training split, aggregated at the product level, not the store level).

**Problem:** A product sold in a high-traffic flagship store will typically have 5-10x
the unit sales of the same product in a small regional store. The current references are
product-level averages, so they represent an average store that may not exist. At test
time the model must condition on a specific (product, store) pair. Using store-level
references (i.e., for a test product at store X, retrieve training products that also
performed well at store X) would provide more calibrated scale information. This is
especially important for the cold-start case where the reference trajectories are the
primary source of magnitude information.

---

## 10. Diffusion Noise Schedule Aggressiveness (`beta_end=0.5`)

**Where:** `config/visuelle2.yaml`, diffusion section: `beta_start: 0.0001`,
`beta_end: 0.5`, `schedule: "quad"`.

**Problem:** The quadratic schedule interpolates `beta_t` values between 0.0001 and 0.5.
For reference, the original DDPM paper (Ho et al. 2020) for image generation uses
`beta_end=0.02`. The cumulative product `alpha_bar_T = prod(1 - beta_t)` determines how
much of the original signal survives at step T. With `beta_end=0.5`, the last few steps
destroy signal very aggressively. CSDI (the closest ancestor model for time series
diffusion) uses `beta_end=0.2`. High `beta_end` values force the model to reconstruct
forecasts from near-complete noise at large t, which is a harder denoising problem and
may require more model capacity or steps than the current 4-block, 64-channel architecture
provides. A schedule search over `beta_end in {0.1, 0.2, 0.3}` with the current
architecture is a low-cost experiment that could improve convergence stability.

---

## Summary Table

| # | Bottleneck | Severity | Implementation Cost |
|---|------------|----------|---------------------|
| 1 | `attr_proj_dim=32`: 94% CLIP compression | High | Low (change one int) |
| 2 | `dim_feedforward=64`: underparameterized FFN | High | Low (change one int) |
| 3 | Non-differentiable frozen retrieval | High | Medium (new reranker module) |
| 4 | No n_obs scalar conditioning | Medium | Low (one Embedding layer) |
| 5 | Dead `attn1` module in ResidualBlock | Medium | Low (remove or wire up) |
| 6 | Unused `context_out` path in RMA | Medium | Low (use or remove) |
| 7 | Discount treated as diffusion target | Medium | Low (routing change) |
| 8 | No reference boundary markers in attention | Low-Medium | Low (add Embedding) |
| 9 | Product-level references ignore store scale | Medium | Medium (preprocessing) |
| 10 | Aggressive noise schedule (`beta_end=0.5`) | Low-Medium | Low (hyperparameter search) |

The highest-priority, lowest-cost improvements are items 1 and 2: increasing
`attr_proj_dim` and `dim_feedforward`. These are single-integer configuration changes
with no structural modifications required, and they address two independent capacity
bottlenecks that directly affect the model's ability to use its two main conditioning
signals (product attributes and temporal patterns).
