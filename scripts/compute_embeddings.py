"""
scripts/compute_embeddings.py

Computes CLIP-based product attribute embeddings for the Visuelle 2.0 dataset.

For each unique product (external_code) the following are produced and saved:
  - image_embeddings.pt    : dict[int -> Tensor(512)]  CLIP image features, L2-normalised
  - tag_embeddings.pt      : dict[int -> Tensor(512)]  CLIP text features, L2-normalised
  - price_scalars.pt       : dict[int -> float]        mean price across stores (already in [0,1])
  - product_embeddings.pt  : dict[int -> Tensor(513)]  fused embedding used for retrieval + model conditioning
                                 = concat(L2_norm(image_emb + tag_emb), price_scalar)

Fusion rationale: CLIP image and text encoders share a joint 512-dim embedding space,
so element-wise sum followed by re-normalisation is the standard way to merge them
without introducing extra parameters at this stage.

Usage:
    conda run -n ML python scripts/compute_embeddings.py \
        --dataset visuelle2 \
        --data_root /home/shu_sho_bhit/BTP_2 \
        --out_dir /home/shu_sho_bhit/BTP_2/visuelle2_processed \
        --batch_size 64 \
        --device cuda
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute CLIP product embeddings")
    p.add_argument("--dataset",    type=str, default="visuelle2",
                   choices=["visuelle2"],
                   help="Which dataset to process")
    p.add_argument("--data_root",  type=str,
                   default="/home/shu_sho_bhit/BTP_2",
                   help="Root directory containing the raw dataset folders")
    p.add_argument("--out_dir",    type=str,
                   default="/home/shu_sho_bhit/BTP_2/visuelle2_processed",
                   help="Output directory for processed .pt files")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for CLIP inference")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--clip_model", type=str,
                   default="openai/clip-vit-base-patch32",
                   help="HuggingFace CLIP model identifier")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visuelle 2.0 helpers
# ---------------------------------------------------------------------------

def load_visuelle2_products(data_root: str) -> pd.DataFrame:
    """
    Loads all unique products from train + test splits.

    Returns a DataFrame with one row per unique external_code, containing:
    external_code, category, color, fabric, image_path, price (mean across stores)
    """
    vis_dir = os.path.join(data_root, "visuelle2")
    train = pd.read_csv(os.path.join(vis_dir, "stfore_train.csv"))
    test  = pd.read_csv(os.path.join(vis_dir, "stfore_test.csv"))
    all_rows = pd.concat([train, test], ignore_index=True)

    # One row per product (external_code); take first occurrence for metadata
    # (category/color/fabric/image_path are product-level, not store-level)
    products = (all_rows
                .groupby("external_code", as_index=False)
                .first()
                [["external_code", "category", "color", "fabric", "image_path"]])

    # Load mean price per product across all stores
    price_df = pd.read_csv(os.path.join(vis_dir, "price_discount_series.csv"))
    mean_price = (price_df
                  .groupby("external_code")["price"]
                  .mean()
                  .reset_index()
                  .rename(columns={"price": "mean_price"}))

    products = products.merge(mean_price, on="external_code", how="left")
    # Fill any missing prices with the global mean
    global_mean = products["mean_price"].mean()
    products["mean_price"] = products["mean_price"].fillna(global_mean)

    print(f"  Loaded {len(products)} unique products "
          f"({len(train['external_code'].unique())} train / "
          f"{len(test['external_code'].unique())} test)")
    return products


def build_tag_text(row: pd.Series) -> str:
    """
    Builds a natural-language description string from a product's three tags.

    Example: "a long sleeve top, grey color, acrylic fabric"
    """
    return f"a {row['category']}, {row['color']} color, {row['fabric']} fabric"


# ---------------------------------------------------------------------------
# CLIP encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_images(image_paths: list[str], processor: CLIPProcessor,
                  model: CLIPModel, device: str, batch_size: int) -> torch.Tensor:
    """
    Encodes a list of image file paths through CLIP's visual encoder.

    Args:
        image_paths: List of absolute paths to image files.
        processor:   CLIPProcessor (handles resizing, normalisation).
        model:       CLIPModel on the target device.
        device:      'cuda' or 'cpu'.
        batch_size:  Number of images per forward pass.

    Returns:
        Tensor of shape (N, 512), L2-normalised, on CPU.
    """
    all_feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="  Encoding images"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"    Warning: could not load {p} ({e}); using blank image")
                images.append(Image.new("RGB", (224, 224)))

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        vision_out = model.vision_model(pixel_values=pixel_values)
        feats = model.visual_projection(vision_out.pooler_output)  # (B, 512)
        feats = F.normalize(feats, dim=-1).cpu()
        all_feats.append(feats)

    return torch.cat(all_feats, dim=0)                      # (N, 512)


@torch.no_grad()
def encode_texts(texts: list[str], processor: CLIPProcessor,
                 model: CLIPModel, device: str, batch_size: int) -> torch.Tensor:
    """
    Encodes a list of text strings through CLIP's text encoder.

    Args:
        texts:       List of natural-language product description strings.
        processor:   CLIPProcessor (handles tokenisation).
        model:       CLIPModel on the target device.
        device:      'cuda' or 'cpu'.
        batch_size:  Number of strings per forward pass.

    Returns:
        Tensor of shape (N, 512), L2-normalised, on CPU.
    """
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding texts "):
        batch_texts = texts[i : i + batch_size]
        enc = processor(text=batch_texts, return_tensors="pt",
                        padding=True, truncation=True)
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        text_out = model.text_model(input_ids=input_ids,
                                    attention_mask=attention_mask)
        feats = model.text_projection(text_out.pooler_output)  # (B, 512)
        feats = F.normalize(feats, dim=-1).cpu()
        all_feats.append(feats)

    return torch.cat(all_feats, dim=0)                      # (N, 512)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset}")
    print(f"Device  : {args.device}")
    print(f"Out dir : {args.out_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load product metadata
    # ------------------------------------------------------------------
    print("[1/5] Loading product metadata ...")
    if args.dataset == "visuelle2":
        products = load_visuelle2_products(args.data_root)
        images_root = os.path.join(args.data_root, "visuelle2", "images")
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' not yet implemented")

    codes       = products["external_code"].tolist()
    image_paths = [os.path.join(images_root, p) for p in products["image_path"]]
    tag_texts   = [build_tag_text(row) for _, row in products.iterrows()]
    prices      = products["mean_price"].values.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Load CLIP model
    # ------------------------------------------------------------------
    print(f"\n[2/5] Loading CLIP model ({args.clip_model}) ...")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    # use_safetensors=True avoids the torch.load CVE check that blocks torch < 2.6
    model     = CLIPModel.from_pretrained(args.clip_model,
                                          use_safetensors=True).to(args.device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Encode images
    # ------------------------------------------------------------------
    print("\n[3/5] Encoding product images ...")
    image_embs = encode_images(image_paths, processor, model,
                               args.device, args.batch_size)   # (N, 512)

    # ------------------------------------------------------------------
    # 4. Encode tag text
    # ------------------------------------------------------------------
    print("\n[4/5] Encoding product tag text ...")
    tag_embs = encode_texts(tag_texts, processor, model,
                            args.device, args.batch_size)       # (N, 512)

    # ------------------------------------------------------------------
    # 5. Fuse and save
    # ------------------------------------------------------------------
    print("\n[5/5] Fusing embeddings and saving ...")

    # Fuse: element-wise sum then re-normalise (standard CLIP multimodal fusion)
    fused = F.normalize(image_embs + tag_embs, dim=-1)          # (N, 512)

    # Append price scalar
    price_tensor = torch.tensor(prices).unsqueeze(1)            # (N, 1)
    product_embs = torch.cat([fused, price_tensor], dim=1)      # (N, 513)

    # Build dicts keyed by external_code (int)
    image_emb_dict   = {int(c): image_embs[i]   for i, c in enumerate(codes)}
    tag_emb_dict     = {int(c): tag_embs[i]     for i, c in enumerate(codes)}
    price_dict       = {int(c): float(prices[i]) for i, c in enumerate(codes)}
    product_emb_dict = {int(c): product_embs[i] for i, c in enumerate(codes)}

    out = Path(args.out_dir)
    torch.save(image_emb_dict,   out / "image_embeddings.pt")
    torch.save(tag_emb_dict,     out / "tag_embeddings.pt")
    torch.save(price_dict,       out / "price_scalars.pt")
    torch.save(product_emb_dict, out / "product_embeddings.pt")

    print(f"\n  Saved {len(codes)} product embeddings to {args.out_dir}/")
    print(f"  image_embeddings.pt  : {image_embs.shape}")
    print(f"  tag_embeddings.pt    : {tag_embs.shape}")
    print(f"  product_embeddings.pt: {product_embs.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
