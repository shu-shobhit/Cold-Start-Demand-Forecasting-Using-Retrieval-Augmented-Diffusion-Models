# %% [markdown]
# # H&M Cold-Start Forecasting — Preprocessing Pipeline
#
# This notebook preprocesses the raw H&M Kaggle dataset into the format
# consumed by `Dataset_HnM` and `RATD_HnM`.
#
# **How to use on Kaggle:**
# 1. Create a new Kaggle notebook and add the dataset:
#    - *H&M Personalized Fashion Recommendations*
#      (`h-and-m-personalized-fashion-recommendations`)
# 2. Set accelerator to **GPU** (for CLIP text encoding).
# 3. Enable **internet access** in notebook settings (needed for HuggingFace).
# 4. Upload this file and click *Run All*.
# 5. When finished, go to the *Output* tab and download the entire
#    `hnm_processed/` folder (or save as a Kaggle dataset).
#
# **Output files produced in `/kaggle/working/hnm_processed/`:**
# ```
#   article_ids_train.pt      list[str]
#   article_ids_val.pt        list[str]
#   article_ids_test.pt       list[str]
#   sales_train.npy           ndarray (N_train, 12) float32
#   sales_val.npy             ndarray (N_val, 12)   float32
#   sales_test.npy            ndarray (N_test, 12)  float32
#   product_embeddings.pt     dict[str -> Tensor(513)]
#   train_references.pt       dict[str -> Tensor(1, 36)]
#   val_references.pt         dict[str -> Tensor(1, 36)]
#   test_references.pt        dict[str -> Tensor(1, 36)]
#   scaler.pkl                sklearn StandardScaler
# ```
#
# Total output size: ~80-100 MB.

# %% [markdown]
# ## 0. Install dependencies

# %%
# Run this cell first — faiss-cpu is not pre-installed on Kaggle
import subprocess
subprocess.run(["pip", "install", "-q", "faiss-cpu", "transformers"], check=True)

# %% [markdown]
# ## 1. Configuration

# %%
import os

# --- Paths ---
DATA_DIR = "/kaggle/input/h-and-m-personalized-fashion-recommendations"
OUT_DIR  = "/kaggle/working/hnm_processed"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Parameters ---
PERIOD_DAYS = 14     # one sales period = 2 weeks
N_PERIODS   = 12     # number of periods per lifecycle
MIN_UNITS   = 100    # density filter: minimum total units sold
K_REFS      = 3      # number of retrieved reference articles
CLIP_MODEL  = "openai/clip-vit-base-patch32"
BATCH_SIZE  = 512    # CLIP text encoding batch size
DEVICE      = "cuda" if __import__("torch").cuda.is_available() else "cpu"
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.80   # val ends at 80%, test is the remaining 20%

print(f"Data dir : {DATA_DIR}")
print(f"Out dir  : {OUT_DIR}")
print(f"Device   : {DEVICE}")

# %% [markdown]
# ## 2. Load raw data

# %%
import pandas as pd
import numpy as np

print("Loading transactions_train.csv (may take a minute) ...")
trans = pd.read_csv(
    os.path.join(DATA_DIR, "transactions_train.csv"),
    usecols=["t_dat", "article_id", "price"],
    dtype={"article_id": str},
)
trans["t_dat"] = pd.to_datetime(trans["t_dat"])

print("Loading articles.csv ...")
articles = pd.read_csv(
    os.path.join(DATA_DIR, "articles.csv"),
    dtype={"article_id": str},
)
articles = articles.set_index("article_id", drop=False)

print(f"Transactions : {len(trans):,}")
print(f"Unique articles in transactions : {trans['article_id'].nunique():,}")
print(f"Articles with metadata : {len(articles):,}")

# %% [markdown]
# ## 3. Compute 12-period sales matrices

# %%
# Find launch date (first transaction) per article
launch = (trans.groupby("article_id")["t_dat"]
          .min().reset_index()
          .rename(columns={"t_dat": "launch_date"}))

trans  = trans.merge(launch, on="article_id")
trans["days_since_launch"] = (trans["t_dat"] - trans["launch_date"]).dt.days
trans["period"]            = trans["days_since_launch"] // PERIOD_DAYS

# Keep only the first N_PERIODS periods
trans_12 = trans[trans["period"] < N_PERIODS]

# Aggregate: count transactions per (article, period)
sales_agg = (trans_12
             .groupby(["article_id", "period"])
             .size()
             .reset_index(name="units"))

# Pivot to matrix: rows = articles, columns = 0..11
sales_matrix = (sales_agg
                .pivot(index="article_id", columns="period", values="units")
                .reindex(columns=range(N_PERIODS), fill_value=0)
                .fillna(0))

print(f"Sales matrix shape (before filter): {sales_matrix.shape}")

# %% [markdown]
# ## 4. Density filter

# %%
# Keep only articles with >= N_PERIODS-1 observed periods AND >= MIN_UNITS total
max_period   = trans.groupby("article_id")["period"].max()
valid_period = max_period[max_period >= (N_PERIODS - 1)].index
valid_units  = sales_matrix[sales_matrix.sum(axis=1) >= MIN_UNITS].index
valid        = valid_period.intersection(valid_units)

sales_matrix = sales_matrix.loc[valid].copy()
print(f"Articles after density filter: {len(sales_matrix):,}")

# Quick look at the average lifecycle curve
avg_curve = sales_matrix.mean()
print("\nAverage weekly sales per period:")
print(avg_curve.round(1).to_string())

# %% [markdown]
# ## 5. Mean price per article

# %%
mean_price = (trans.groupby("article_id")["price"]
              .mean()
              .reindex(sales_matrix.index, fill_value=0.0))

# Normalise price to [0, 1]
p_min, p_max = mean_price.min(), mean_price.max()
price_norm = ((mean_price - p_min) / (p_max - p_min + 1e-8)).astype(np.float32)

print(f"Price range: {p_min:.4f} – {p_max:.4f}  (normalised to [0, 1])")

# %% [markdown]
# ## 6. Chronological 70 / 10 / 20 split

# %%
valid_launch = (launch[launch["article_id"].isin(sales_matrix.index)]
                .copy()
                .sort_values("launch_date")
                .reset_index(drop=True))

n       = len(valid_launch)
n_train = int(n * TRAIN_FRAC)
n_val   = int(n * VAL_FRAC)

train_ids = valid_launch.iloc[:n_train]["article_id"].tolist()
val_ids   = valid_launch.iloc[n_train:n_val]["article_id"].tolist()
test_ids  = valid_launch.iloc[n_val:]["article_id"].tolist()

print(f"Train : {len(train_ids):>6,}  articles  "
      f"({valid_launch.iloc[0]['launch_date'].date()} – "
      f"{valid_launch.iloc[n_train-1]['launch_date'].date()})")
print(f"Val   : {len(val_ids):>6,}  articles  "
      f"({valid_launch.iloc[n_train]['launch_date'].date()} – "
      f"{valid_launch.iloc[n_val-1]['launch_date'].date()})")
print(f"Test  : {len(test_ids):>6,}  articles  "
      f"({valid_launch.iloc[n_val]['launch_date'].date()} – "
      f"{valid_launch.iloc[-1]['launch_date'].date()})")

# %% [markdown]
# ## 7. Normalise sales (log1p + StandardScaler)

# %%
import pickle
from sklearn.preprocessing import StandardScaler

def _ids_to_array(ids):
    return sales_matrix.loc[ids].values.astype(np.float32)

train_sales_raw = _ids_to_array(train_ids)
val_sales_raw   = _ids_to_array(val_ids)
test_sales_raw  = _ids_to_array(test_ids)

log_train = np.log1p(train_sales_raw)
log_val   = np.log1p(val_sales_raw)
log_test  = np.log1p(test_sales_raw)

scaler = StandardScaler()
scaler.fit(log_train.reshape(-1, 1))

def _scale(arr):
    return scaler.transform(arr.reshape(-1, 1)).reshape(arr.shape).astype(np.float32)

train_norm = _scale(log_train)
val_norm   = _scale(log_val)
test_norm  = _scale(log_test)

print(f"Train sales: mean={train_norm.mean():.3f}  std={train_norm.std():.3f}")
print(f"Val   sales: mean={val_norm.mean():.3f}  std={val_norm.std():.3f}")
print(f"Test  sales: mean={test_norm.mean():.3f}  std={test_norm.std():.3f}")

with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("Saved scaler.pkl")

# %% [markdown]
# ## 8. CLIP text embeddings

# %%
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

def build_rag_text(art_id: str) -> str:
    """Compose a descriptive sentence from article metadata."""
    row   = articles.loc[art_id] if art_id in articles.index else {}
    parts = []
    for col in ["prod_name", "product_type_name",
                "graphical_appearance_name", "colour_group_name",
                "department_name"]:
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in ("", "nan"):
            parts.append(val)
    return ". ".join(parts) if parts else "fashion product"


print(f"Loading CLIP model: {CLIP_MODEL} ...")
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip      = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True).to(DEVICE)
clip.eval()

all_ids = train_ids + val_ids + test_ids
texts   = [build_rag_text(a) for a in all_ids]

# Encode in batches
all_feats = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc   = processor(text=batch, return_tensors="pt",
                          padding=True, truncation=True)
        ids_t  = enc["input_ids"].to(DEVICE)
        attn_t = enc["attention_mask"].to(DEVICE)
        out    = clip.text_model(input_ids=ids_t, attention_mask=attn_t)
        feat   = clip.text_projection(out.pooler_output)
        feat   = F.normalize(feat, dim=-1).cpu()
        all_feats.append(feat)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  {i}/{len(texts)} encoded", end="\r")

print(f"\n  Done — {len(texts)} articles encoded")

text_embs = torch.cat(all_feats, dim=0)          # (N_all, 512)

# Append normalised price
prices_arr = price_norm.loc[all_ids].values.astype(np.float32)
price_t    = torch.from_numpy(prices_arr).unsqueeze(1)   # (N_all, 1)
prod_embs  = torch.cat([text_embs, price_t], dim=1)      # (N_all, 513)

prod_emb_dict = {aid: prod_embs[i] for i, aid in enumerate(all_ids)}
print(f"  Embedding shape: {prod_embs.shape}")

# Free GPU memory
del clip
torch.cuda.empty_cache()

# %% [markdown]
# ## 9. FAISS retrieval

# %%
import faiss

def retrieve_neighbours(query_ids, train_ids, emb_dict, k, exclude_self=True):
    train_matrix = np.stack([emb_dict[a].numpy() for a in train_ids]).astype(np.float32)
    norms        = np.linalg.norm(train_matrix, axis=1, keepdims=True).clip(min=1e-8)
    train_matrix = train_matrix / norms

    dim   = train_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(train_matrix)

    pos_to_id = dict(enumerate(train_ids))
    id_to_pos = {a: i for i, a in enumerate(train_ids)}

    query_matrix = np.stack([emb_dict[a].numpy() for a in query_ids]).astype(np.float32)
    q_norms      = np.linalg.norm(query_matrix, axis=1, keepdims=True).clip(min=1e-8)
    query_matrix = query_matrix / q_norms

    fetch_k    = k + 1 if exclude_self else k
    _, indices = index.search(query_matrix, fetch_k)

    ref_indices = {}
    for qi, aid in enumerate(query_ids):
        nbrs = [pos_to_id[p] for p in indices[qi].tolist() if p >= 0]
        if exclude_self and aid in id_to_pos:
            nbrs = [n for n in nbrs if n != aid]
        ref_indices[aid] = nbrs[:k]
    return ref_indices


print("Building FAISS index and retrieving neighbours ...")
train_ref_idx = retrieve_neighbours(
    train_ids, train_ids, prod_emb_dict, K_REFS, exclude_self=True
)
val_ref_idx   = retrieve_neighbours(
    val_ids, train_ids, prod_emb_dict, K_REFS, exclude_self=False
)
test_ref_idx  = retrieve_neighbours(
    test_ids, train_ids, prod_emb_dict, K_REFS, exclude_self=False
)
print("  Done.")

# %% [markdown]
# ## 10. Build reference tensors

# %%
# mean trajectory per training article = that article's own normalised sales
# (article-level dataset: no store aggregation needed)
mean_traj = {
    aid: torch.from_numpy(train_norm[i]).unsqueeze(0)   # (1, 12)
    for i, aid in enumerate(train_ids)
}

def build_refs(query_ids, ref_idx):
    """dict[article_id -> Tensor(1, K_REFS*12)]"""
    refs = {}
    for aid in query_ids:
        nbrs  = ref_idx[aid]
        parts = [mean_traj[n] for n in nbrs]
        refs[aid] = torch.cat(parts, dim=1)   # (1, K_REFS*12)
    return refs

train_refs = build_refs(train_ids, train_ref_idx)
val_refs   = build_refs(val_ids,   val_ref_idx)
test_refs  = build_refs(test_ids,  test_ref_idx)

# Sanity check
sample_id  = train_ids[0]
print(f"Sample article : {sample_id}")
print(f"  Neighbours   : {train_ref_idx[sample_id]}")
print(f"  Reference    : {train_refs[sample_id].shape}")
print(f"  Embedding    : {prod_emb_dict[sample_id].shape}")

# %% [markdown]
# ## 11. Save all outputs

# %%
print("Saving output files ...")

torch.save(train_ids, os.path.join(OUT_DIR, "article_ids_train.pt"))
torch.save(val_ids,   os.path.join(OUT_DIR, "article_ids_val.pt"))
torch.save(test_ids,  os.path.join(OUT_DIR, "article_ids_test.pt"))

np.save(os.path.join(OUT_DIR, "sales_train.npy"), train_norm)
np.save(os.path.join(OUT_DIR, "sales_val.npy"),   val_norm)
np.save(os.path.join(OUT_DIR, "sales_test.npy"),  test_norm)

torch.save(prod_emb_dict, os.path.join(OUT_DIR, "product_embeddings.pt"))
torch.save(train_refs,    os.path.join(OUT_DIR, "train_references.pt"))
torch.save(val_refs,      os.path.join(OUT_DIR, "val_references.pt"))
torch.save(test_refs,     os.path.join(OUT_DIR, "test_references.pt"))

print("\nAll files saved to", OUT_DIR)
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fname)
    size  = os.path.getsize(fpath) / 1024**2
    print(f"  {fname:<40}  {size:6.1f} MB")

# %% [markdown]
# ## 12. Verification

# %%
# Quick round-trip check
import pickle

with open(os.path.join(OUT_DIR, "scaler.pkl"), "rb") as f:
    sc = pickle.load(f)

embs  = torch.load(os.path.join(OUT_DIR, "product_embeddings.pt"), weights_only=True)
t_ids = torch.load(os.path.join(OUT_DIR, "article_ids_test.pt"),   weights_only=True)
t_ref = torch.load(os.path.join(OUT_DIR, "test_references.pt"),    weights_only=True)
t_sal = np.load(os.path.join(OUT_DIR, "sales_test.npy"))

art0 = t_ids[0]
print("Verification:")
print(f"  test article_id   : {art0}")
print(f"  embedding shape   : {embs[art0].shape}")
print(f"  reference shape   : {t_ref[art0].shape}")
print(f"  sales[0] shape    : {t_sal[0].shape}")
print(f"  scaler mean_      : {sc.mean_[0]:.4f}  scale_ : {sc.scale_[0]:.4f}")

# Inverse transform check
scaled_val = t_sal[0, 0]
raw_approx = np.expm1(sc.inverse_transform([[scaled_val]])[0, 0])
print(f"  sales[0,0] normalised={scaled_val:.3f}  -> raw ~{raw_approx:.1f} units/period")

print("\nAll checks passed. Download /kaggle/working/hnm_processed/ to your local machine.")
