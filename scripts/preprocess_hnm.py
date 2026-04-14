"""scripts/preprocess_hnm.py

Full H&M preprocessing pipeline.  Reads raw Kaggle files and produces the
processed .pt / .npy / .pkl files consumed by Dataset_HnM.

This script is meant to be run on a machine with the full H&M dataset.
For Kaggle, use notebooks/hnm_kaggle.py instead (same logic, notebook cells).

Stages:
  1. Load transactions + articles, compute 12-period weekly sales matrices.
  2. Filter articles by density (>= 100 total units in 12 periods).
  3. Chronological 70/10/20 train/val/test split by article launch date.
  4. Fit StandardScaler on log1p(train_sales); apply to all splits.
  5. Load CLIP text encoder; compute 513-dim product embeddings
     (CLIP text 512-dim + normalized price 1-dim).
  6. Build FAISS index; retrieve k=3 nearest training neighbours per article.
  7. Build reference tensors (1, k*12) from normalised training sales.
  8. Save all outputs to --out_dir.

Output files (hnm_processed/):
  article_ids_{train,val,test}.pt   list[str]
  sales_{train,val,test}.npy        ndarray (N, 12) float32
  product_embeddings.pt             dict[str -> Tensor(513)]
  {train,val,test}_references.pt    dict[str -> Tensor(1, 36)]
  scaler.pkl                        sklearn StandardScaler

Usage (local):
  conda run -n ML python scripts/preprocess_hnm.py \\
      --data_dir /home/shu_sho_bhit/BTP_2/hnm_dataset_orig \\
      --out_dir  /home/shu_sho_bhit/BTP_2/hnm_processed \\
      --device cuda

Usage (Kaggle, run from notebooks/hnm_kaggle.py):
  python preprocess_hnm.py \\
      --data_dir /kaggle/input/h-and-m-personalized-fashion-recommendations \\
      --out_dir  /kaggle/working/hnm_processed \\
      --device cuda
"""

import argparse
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="H&M preprocessing pipeline")
    p.add_argument("--data_dir",   type=str,
                   default="/home/shu_sho_bhit/BTP_2/hnm_dataset_orig",
                   help="Directory containing transactions_train.csv and articles.csv")
    p.add_argument("--out_dir",    type=str,
                   default="/home/shu_sho_bhit/BTP_2/hnm_processed")
    p.add_argument("--k",          type=int, default=3,
                   help="Number of retrieved references per article")
    p.add_argument("--min_units",  type=int, default=100,
                   help="Minimum total sales to include an article (density filter)")
    p.add_argument("--period_days", type=int, default=14,
                   help="Length of one sales period in days (default 14 = 2 weeks)")
    p.add_argument("--clip_model", type=str,
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--batch_size", type=int, default=512,
                   help="Batch size for CLIP text encoding")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage 1-3: Sales matrices and splits
# ---------------------------------------------------------------------------

def build_sales_matrix(data_dir: str, min_units: int, period_days: int):
    """Load transactions, compute 12-period sales matrices, filter, and split.

    Returns:
        train_ids, val_ids, test_ids   : list[str]
        train_sales, val_sales, test_sales : ndarray (N, 12) raw unit counts
        articles_df : DataFrame with article metadata
    """
    print("[1/5] Loading transactions and articles ...")
    trans = pd.read_csv(
        os.path.join(data_dir, "transactions_train.csv"),
        usecols=["t_dat", "article_id", "price"],
        dtype={"article_id": str},
    )
    articles = pd.read_csv(
        os.path.join(data_dir, "articles.csv"),
        dtype={"article_id": str},
    )

    trans["t_dat"] = pd.to_datetime(trans["t_dat"])
    print(f"  Transactions loaded: {len(trans):,}")
    print(f"  Unique articles    : {trans['article_id'].nunique():,}")

    # Launch date per article
    launch = (trans.groupby("article_id")["t_dat"].min()
               .reset_index().rename(columns={"t_dat": "launch_date"}))
    trans  = trans.merge(launch, on="article_id")

    trans["days_since_launch"] = (trans["t_dat"] - trans["launch_date"]).dt.days
    trans["period"]            = trans["days_since_launch"] // period_days

    # Keep only first 12 periods
    trans_12 = trans[trans["period"] < 12]

    # Aggregate weekly unit sales (count transactions per period)
    sales_agg = (trans_12
                 .groupby(["article_id", "period"])
                 .size()
                 .reset_index(name="units"))

    # Pivot to (article, period) matrix
    sales_matrix = (sales_agg
                    .pivot(index="article_id", columns="period", values="units")
                    .reindex(columns=range(12), fill_value=0)
                    .fillna(0))

    # Density filter: >= 12 periods present AND >= min_units total
    max_period   = trans.groupby("article_id")["period"].max()
    valid_period = max_period[max_period >= 11].index
    valid_units  = sales_matrix[sales_matrix.sum(axis=1) >= min_units].index
    valid        = valid_period.intersection(valid_units)
    sales_matrix = sales_matrix.loc[valid]

    print(f"  Articles after density filter: {len(sales_matrix):,}")

    # Mean price per article
    mean_price = (trans.groupby("article_id")["price"].mean()
                  .reindex(sales_matrix.index, fill_value=0.0))

    # Chronological 70/10/20 split by launch date
    valid_launch = launch[launch["article_id"].isin(sales_matrix.index)].copy()
    valid_launch = valid_launch.sort_values("launch_date").reset_index(drop=True)
    n = len(valid_launch)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.80)

    train_ids = valid_launch.iloc[:n_train]["article_id"].tolist()
    val_ids   = valid_launch.iloc[n_train:n_val]["article_id"].tolist()
    test_ids  = valid_launch.iloc[n_val:]["article_id"].tolist()

    print(f"  Train: {len(train_ids):,}  Val: {len(val_ids):,}  Test: {len(test_ids):,}")

    def _sales_array(ids):
        return sales_matrix.loc[ids].values.astype(np.float32)

    train_sales = _sales_array(train_ids)
    val_sales   = _sales_array(val_ids)
    test_sales  = _sales_array(test_ids)

    # Merge price into articles
    articles    = articles.set_index("article_id", drop=False)
    articles    = articles.loc[articles.index.intersection(sales_matrix.index)].copy()
    articles["mean_price"] = mean_price.loc[articles.index].values

    return (train_ids, val_ids, test_ids,
            train_sales, val_sales, test_sales,
            articles, mean_price)


# ---------------------------------------------------------------------------
# Stage 4: Normalisation
# ---------------------------------------------------------------------------

def normalise_sales(train_sales, val_sales, test_sales):
    """Apply log1p then StandardScaler fitted on train.

    Returns:
        train_norm, val_norm, test_norm  : ndarray (N, 12) float32
        scaler                           : fitted sklearn StandardScaler
    """
    print("[2/5] Normalising sales (log1p + StandardScaler) ...")

    log_train = np.log1p(train_sales)
    log_val   = np.log1p(val_sales)
    log_test  = np.log1p(test_sales)

    scaler = StandardScaler()
    # Fit on flattened train values so the scaler sees all time steps
    scaler.fit(log_train.reshape(-1, 1))

    def _scale(arr):
        return scaler.transform(arr.reshape(-1, 1)).reshape(arr.shape).astype(np.float32)

    return _scale(log_train), _scale(log_val), _scale(log_test), scaler


# ---------------------------------------------------------------------------
# Stage 5: CLIP text embeddings
# ---------------------------------------------------------------------------

def build_rag_text(row: pd.Series) -> str:
    """Compose a single descriptive sentence from article metadata."""
    parts = []
    for col in ["prod_name", "product_type_name", "graphical_appearance_name",
                "colour_group_name", "department_name"]:
        val = str(row.get(col, "")).strip()
        if val and val.lower() != "nan":
            parts.append(val)
    return ". ".join(parts)


@torch.no_grad()
def encode_texts(texts, processor, model, device, batch_size):
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = processor(text=batch, return_tensors="pt",
                          padding=True, truncation=True)
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out  = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        feat = model.text_projection(out.pooler_output)
        feat = F.normalize(feat, dim=-1).cpu()
        all_feats.append(feat)
        if (i // batch_size) % 20 == 0:
            print(f"    {i}/{len(texts)} texts encoded", end="\r")
    print()
    return torch.cat(all_feats, dim=0)   # (N, 512)


def compute_embeddings(articles_df, mean_price, clip_model_id, device, batch_size):
    """Compute 513-dim product embeddings for all articles.

    Returns:
        dict[article_id_str -> Tensor(513)]
    """
    print(f"[3/5] Computing CLIP text embeddings ({clip_model_id}) ...")
    processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip      = CLIPModel.from_pretrained(clip_model_id,
                                          use_safetensors=True).to(device)
    clip.eval()

    article_ids = articles_df.index.tolist()
    texts       = [build_rag_text(articles_df.loc[aid]) for aid in article_ids]

    text_embs = encode_texts(texts, processor, clip, device, batch_size)  # (N, 512)

    # Normalise price to [0, 1]
    prices    = mean_price.loc[article_ids].values.astype(np.float32)
    p_min, p_max = prices.min(), prices.max()
    if p_max > p_min:
        prices_norm = (prices - p_min) / (p_max - p_min)
    else:
        prices_norm = np.zeros_like(prices)
    price_t   = torch.from_numpy(prices_norm).unsqueeze(1)           # (N, 1)

    product_embs = torch.cat([text_embs, price_t], dim=1)            # (N, 513)

    emb_dict = {aid: product_embs[i] for i, aid in enumerate(article_ids)}
    print(f"  Computed {len(emb_dict)} embeddings (dim=513)")
    return emb_dict


# ---------------------------------------------------------------------------
# Stage 6-7: FAISS retrieval + reference tensors
# ---------------------------------------------------------------------------

def retrieve_neighbours(query_ids, train_ids, emb_dict, k, exclude_self=True):
    """FAISS inner-product search over L2-normalised embeddings."""
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

    fetch_k        = k + 1 if exclude_self else k
    _, indices     = index.search(query_matrix, fetch_k)

    ref_indices = {}
    for qi, aid in enumerate(query_ids):
        nbrs = [pos_to_id[p] for p in indices[qi].tolist() if p >= 0]
        if exclude_self and aid in id_to_pos:
            nbrs = [n for n in nbrs if n != aid]
        ref_indices[aid] = nbrs[:k]
    return ref_indices


def build_mean_trajectories(train_ids, train_sales_norm):
    """Mean (normalised) trajectory per training article.

    For H&M (one sample per article), mean trajectory = the article's own
    normalised sales.

    Returns:
        dict[article_id_str -> Tensor(1, 12)]
    """
    return {
        aid: torch.from_numpy(train_sales_norm[i]).unsqueeze(0)   # (1, 12)
        for i, aid in enumerate(train_ids)
    }


def build_reference_tensors(query_ids, ref_indices, mean_traj, k):
    """Concatenate k reference trajectories per article.

    Returns:
        dict[article_id_str -> Tensor(1, k*12)]
    """
    refs = {}
    for aid in query_ids:
        nbrs  = ref_indices[aid]
        parts = [mean_traj[n] for n in nbrs]    # list of (1, 12)
        refs[aid] = torch.cat(parts, dim=1)      # (1, k*12)
    return refs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"H&M Preprocessing Pipeline")
    print(f"  data_dir : {args.data_dir}")
    print(f"  out_dir  : {args.out_dir}")
    print(f"  k        : {args.k}")
    print(f"  device   : {args.device}")
    print(f"{'='*60}\n")

    # --- Stage 1-3: Sales matrices + splits
    (train_ids, val_ids, test_ids,
     train_sales_raw, val_sales_raw, test_sales_raw,
     articles_df, mean_price) = build_sales_matrix(
        args.data_dir, args.min_units, args.period_days
    )

    # --- Stage 4: Normalise
    train_norm, val_norm, test_norm, scaler = normalise_sales(
        train_sales_raw, val_sales_raw, test_sales_raw
    )

    # --- Stage 5: Embeddings
    emb_dict = compute_embeddings(
        articles_df, mean_price, args.clip_model, args.device, args.batch_size
    )

    # --- Stage 6: Retrieval
    print("[4/5] Building FAISS index and retrieving neighbours ...")
    all_ids = train_ids + val_ids + test_ids
    # Only keep embeddings for articles that survived the density filter
    valid_emb = {a: emb_dict[a] for a in all_ids if a in emb_dict}

    train_ref_idx = retrieve_neighbours(
        train_ids, train_ids, valid_emb, args.k, exclude_self=True
    )
    val_ref_idx   = retrieve_neighbours(
        val_ids, train_ids, valid_emb, args.k, exclude_self=False
    )
    test_ref_idx  = retrieve_neighbours(
        test_ids, train_ids, valid_emb, args.k, exclude_self=False
    )

    # --- Stage 7: Reference tensors
    print("[5/5] Building reference tensors and saving ...")
    mean_traj = build_mean_trajectories(train_ids, train_norm)

    train_refs = build_reference_tensors(train_ids, train_ref_idx, mean_traj, args.k)
    val_refs   = build_reference_tensors(val_ids,   val_ref_idx,   mean_traj, args.k)
    test_refs  = build_reference_tensors(test_ids,  test_ref_idx,  mean_traj, args.k)

    # --- Save
    torch.save(train_ids, out / "article_ids_train.pt")
    torch.save(val_ids,   out / "article_ids_val.pt")
    torch.save(test_ids,  out / "article_ids_test.pt")

    np.save(out / "sales_train.npy", train_norm)
    np.save(out / "sales_val.npy",   val_norm)
    np.save(out / "sales_test.npy",  test_norm)

    # product_embeddings.pt: only the articles in any split
    prod_emb_filtered = {a: valid_emb[a] for a in all_ids if a in valid_emb}
    torch.save(prod_emb_filtered, out / "product_embeddings.pt")

    torch.save(train_refs, out / "train_references.pt")
    torch.save(val_refs,   out / "val_references.pt")
    torch.save(test_refs,  out / "test_references.pt")

    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Sanity check
    sample_id  = train_ids[0]
    sample_ref = train_refs[sample_id]
    print(f"\nSample article {sample_id}:")
    print(f"  neighbours  : {train_ref_idx[sample_id]}")
    print(f"  reference   : {sample_ref.shape}  dtype={sample_ref.dtype}")
    print(f"  embedding   : {valid_emb[sample_id].shape}")
    print(f"\nSaved all files to {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
