"""
scripts/compute_retrieval.py

Builds a FAISS index over training-product embeddings and precomputes
top-k nearest-neighbour reference trajectories for every product.

For Visuelle 2.0 the reference trajectory is the mean of a retrieved product's
sales+discount series averaged across all stores in the training split.
Reference shape per query product: (2, k*12)  — k=3 concatenated mean trajectories.

Saves:
  visuelle2_processed/train_ref_indices.pt   dict[int -> List[int]]  k neighbour codes per train product
  visuelle2_processed/test_ref_indices.pt    dict[int -> List[int]]  k neighbour codes per test product
  visuelle2_processed/train_references.pt    dict[int -> Tensor(2, k*12)]
  visuelle2_processed/test_references.pt     dict[int -> Tensor(2, k*12)]

Usage:
    conda run -n ML python scripts/compute_retrieval.py \
        --dataset visuelle2 \
        --data_root /home/shu_sho_bhit/BTP_2 \
        --emb_dir  /home/shu_sho_bhit/BTP_2/visuelle2_processed \
        --out_dir  /home/shu_sho_bhit/BTP_2/visuelle2_processed \
        --k 3
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build FAISS retrieval index")
    p.add_argument("--dataset",   type=str, default="visuelle2",
                   choices=["visuelle2"])
    p.add_argument("--data_root", type=str,
                   default="/home/shu_sho_bhit/BTP_2")
    p.add_argument("--emb_dir",   type=str,
                   default="/home/shu_sho_bhit/BTP_2/visuelle2_processed",
                   help="Directory containing product_embeddings.pt")
    p.add_argument("--out_dir",   type=str,
                   default="/home/shu_sho_bhit/BTP_2/visuelle2_processed")
    p.add_argument("--k",         type=int, default=3,
                   help="Number of nearest neighbours to retrieve")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visuelle 2.0 helpers
# ---------------------------------------------------------------------------

def load_visuelle2_splits(data_root: str):
    """
    Returns (train_df, test_df) DataFrames from stfore_train/test.csv.

    The weekly sales columns ('0'–'11') are kept in their normalised float form.
    The discount columns ('d_0'–'d_11') are merged from price_discount_series.csv.
    """
    vis_dir = os.path.join(data_root, "visuelle2")
    sales_cols = [str(i) for i in range(12)]

    train = pd.read_csv(os.path.join(vis_dir, "stfore_train.csv"))
    test  = pd.read_csv(os.path.join(vis_dir, "stfore_test.csv"))

    disc  = pd.read_csv(os.path.join(vis_dir, "price_discount_series.csv"))
    disc  = disc.rename(columns={str(i): f"d_{i}" for i in range(12)})
    disc_cols = [f"d_{i}" for i in range(12)]

    train = train.merge(disc[["external_code", "retail"] + disc_cols],
                        on=["external_code", "retail"], how="left")
    test  = test.merge( disc[["external_code", "retail"] + disc_cols],
                        on=["external_code", "retail"], how="left")

    print(f"  Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test, sales_cols, disc_cols


def build_mean_trajectories(train_df: pd.DataFrame,
                             sales_cols: list[str],
                             disc_cols:  list[str]) -> dict:
    """
    Computes the mean sales+discount trajectory for each product across all stores.

    Args:
        train_df:   Training DataFrame with per-store rows.
        sales_cols: Column names for the 12 weekly sales values.
        disc_cols:  Column names for the 12 weekly discount values.

    Returns:
        dict[external_code (int) -> Tensor(2, 12)]
        Channel 0 = mean normalised sales across stores.
        Channel 1 = mean discount ratio across stores.
    """
    mean_traj = {}
    for code, group in tqdm(train_df.groupby("external_code"),
                            desc="  Computing mean trajectories"):
        sales = group[sales_cols].values.astype(np.float32)   # (n_stores, 12)
        disc  = group[disc_cols].values.astype(np.float32)    # (n_stores, 12)
        # Fill any NaN discount values with 0 (no discount)
        disc  = np.nan_to_num(disc, nan=0.0)
        mean_sales = sales.mean(axis=0)                        # (12,)
        mean_disc  = disc.mean(axis=0)                         # (12,)
        traj = torch.from_numpy(np.stack([mean_sales, mean_disc], axis=0))  # (2,12)
        mean_traj[int(code)] = traj
    return mean_traj


def build_reference_tensors(codes: list[int],
                             ref_indices: dict,
                             mean_traj: dict,
                             k: int) -> dict:
    """
    Builds the concatenated reference tensor for each product.

    Args:
        codes:       List of external_codes to build references for.
        ref_indices: dict[code -> [code_1, ..., code_k]] (retrieved neighbours).
        mean_traj:   dict[code -> Tensor(2, 12)] (mean trajectory per train product).
        k:           Number of neighbours.

    Returns:
        dict[external_code -> Tensor(2, k*12)]
    """
    references = {}
    for code in tqdm(codes, desc="  Building reference tensors"):
        neighbours = ref_indices[code]                          # list of k codes
        parts = [mean_traj[n] for n in neighbours]             # list of (2,12)
        references[code] = torch.cat(parts, dim=1)             # (2, k*12)
    return references


# ---------------------------------------------------------------------------
# FAISS retrieval
# ---------------------------------------------------------------------------

def retrieve_neighbours(query_codes: list[int],
                         train_codes: list[int],
                         product_emb: dict,
                         k: int,
                         exclude_self: bool = True) -> dict:
    """
    Builds a FAISS flat inner-product index over train embeddings and queries
    each product for its top-k nearest neighbours.

    Args:
        query_codes:  Products to retrieve neighbours for.
        train_codes:  Products in the index (train set only — no leakage).
        product_emb:  dict[code -> Tensor(513)] pre-computed L2-normalised embeddings.
        k:            Number of neighbours.
        exclude_self: If True, the query itself is excluded (for train queries).

    Returns:
        dict[code -> List[code]] — k neighbour codes per query.
    """
    # Build index matrix from train embeddings
    train_matrix = np.stack(
        [product_emb[c].numpy() for c in train_codes], axis=0
    ).astype(np.float32)                                        # (N_train, 513)

    dim = train_matrix.shape[1]

    # L2-normalise (the 513-dim embedding already has L2-norm ≈ 1 except for the
    # appended price scalar — re-normalise here to ensure cosine sim = inner product)
    norms = np.linalg.norm(train_matrix, axis=1, keepdims=True).clip(min=1e-8)
    train_matrix = train_matrix / norms

    # Build FAISS index
    index = faiss.IndexFlatIP(dim)                              # inner product = cosine
    index.add(train_matrix)

    # Map from train position → external_code and vice-versa
    pos_to_code = {i: c for i, c in enumerate(train_codes)}
    code_to_pos = {c: i for i, c in enumerate(train_codes)}

    # Query each product
    fetch_k = k + 1 if exclude_self else k   # fetch one extra to drop self
    query_matrix = np.stack(
        [product_emb[c].numpy() for c in query_codes], axis=0
    ).astype(np.float32)
    query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True).clip(min=1e-8)
    query_matrix = query_matrix / query_norms

    _, indices = index.search(query_matrix, fetch_k)            # (N_query, fetch_k)

    ref_indices = {}
    for qi, code in enumerate(query_codes):
        neighbour_positions = indices[qi].tolist()
        neighbours = [pos_to_code[p] for p in neighbour_positions
                      if p >= 0]                                # -1 = not found
        if exclude_self and code in code_to_pos:
            neighbours = [n for n in neighbours if n != code]
        ref_indices[code] = neighbours[:k]

    return ref_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset}")
    print(f"k       : {args.k}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load embeddings
    # ------------------------------------------------------------------
    print("[1/5] Loading product embeddings ...")
    emb_path    = os.path.join(args.emb_dir, "product_embeddings.pt")
    product_emb = torch.load(emb_path, map_location="cpu", weights_only=True)
    print(f"  Loaded embeddings for {len(product_emb)} products")

    # ------------------------------------------------------------------
    # 2. Load splits and build mean trajectories
    # ------------------------------------------------------------------
    print("\n[2/5] Loading data splits ...")
    train_df, test_df, sales_cols, disc_cols = load_visuelle2_splits(args.data_root)

    train_codes = sorted(train_df["external_code"].unique().tolist())
    test_codes  = sorted(test_df["external_code"].unique().tolist())
    print(f"  Unique train products: {len(train_codes)}")
    print(f"  Unique test products : {len(test_codes)}")

    print("\n[3/5] Building mean trajectories for train products ...")
    mean_traj = build_mean_trajectories(train_df, sales_cols, disc_cols)

    # ------------------------------------------------------------------
    # 3. Retrieve neighbours
    # ------------------------------------------------------------------
    print("\n[4/5] Building FAISS index and retrieving neighbours ...")

    print("  Querying train products (exclude self) ...")
    train_ref_indices = retrieve_neighbours(
        query_codes=train_codes,
        train_codes=train_codes,
        product_emb=product_emb,
        k=args.k,
        exclude_self=True,
    )

    print("  Querying test products (no self to exclude) ...")
    test_ref_indices = retrieve_neighbours(
        query_codes=test_codes,
        train_codes=train_codes,
        product_emb=product_emb,
        k=args.k,
        exclude_self=False,
    )

    # ------------------------------------------------------------------
    # 4. Build and save reference tensors
    # ------------------------------------------------------------------
    print("\n[5/5] Building reference tensors and saving ...")

    train_references = build_reference_tensors(
        train_codes, train_ref_indices, mean_traj, args.k)
    test_references  = build_reference_tensors(
        test_codes,  test_ref_indices,  mean_traj, args.k)

    out = Path(args.out_dir)
    torch.save(train_ref_indices, out / "train_ref_indices.pt")
    torch.save(test_ref_indices,  out / "test_ref_indices.pt")
    torch.save(train_references,  out / "train_references.pt")
    torch.save(test_references,   out / "test_references.pt")

    # Sanity check
    sample_code = train_codes[0]
    sample_ref  = train_references[sample_code]
    print(f"\n  Sample product {sample_code}:")
    print(f"    neighbours : {train_ref_indices[sample_code]}")
    print(f"    reference  : shape {sample_ref.shape}, "
          f"dtype {sample_ref.dtype}")
    print(f"\n  Saved to {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
