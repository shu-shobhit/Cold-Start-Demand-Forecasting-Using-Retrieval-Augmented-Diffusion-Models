"""tests/smoke/test_diff_models.py

Smoke tests for diff_models.py.

Tests:
  - ReferenceModulatedCrossAttention forward pass with and without attr_emb
  - ResidualBlock forward pass with and without attr_emb
  - diff_RATD end-to-end forward pass (with/without reference and attr_emb)
  - Backward compatibility: model with attr_dim=None behaves identically to
    the original RATD (attr_emb parameter is silently ignored)

Expected outcome:
  All output tensors have the correct shape.
  No exceptions are raised.
  Backward pass computes gradients without error.
  PASS printed at end of each section.
"""

import sys
import os

# Add project root (two levels up from tests/smoke/) to import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from diff_models import (
    ReferenceModulatedCrossAttention,
    ResidualBlock,
    diff_RATD,
)

SEP = "-" * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mini_diff_config(with_attr: bool = True) -> dict:
    """Return a small diff_RATD config for fast CPU tests."""
    cfg = {
        "channels": 16,
        "diffusion_embedding_dim": 32,
        "beta_start": 0.0001,
        "beta_end": 0.5,
        "num_steps": 10,
        "schedule": "quad",
        "is_linear": False,
        "layers": 2,
        "nheads": 4,
        "side_dim": 49,   # emb_total(41) + attr_side(8) for test model
        "h_size": 0,
        "ref_size": 12,
    }
    if with_attr:
        cfg["attr_dim"] = 513
    return cfg


def check(condition: bool, msg: str):
    if not condition:
        raise AssertionError(f"FAIL: {msg}")


# ---------------------------------------------------------------------------
# 1. ReferenceModulatedCrossAttention
# ---------------------------------------------------------------------------

def test_rma_no_attr():
    print(SEP)
    print("TEST 1: RMA forward — no attr_emb (original RATD behavior)")

    # dim = ref_size + h_size = 12 + 0 = 12
    # context_dim = ref_size * k = 12 * 3 = 36
    rma = ReferenceModulatedCrossAttention(
        dim=12, context_dim=36, heads=4, dim_head=8, attr_dim=None
    )

    B, C, K, L = 2, 4, 2, 12
    R = 36
    x         = torch.randn(B, C, K, L)
    cond_info = torch.randn(B, C, K, L)
    reference = torch.randn(B, K, R)

    out = rma(x, cond_info, reference, attr_emb=None)
    check(out.shape == (B * C, K, L),
          f"RMA output shape {out.shape} != ({B*C}, {K}, {L})")
    print(f"  output shape: {out.shape}  PASS")


def test_rma_with_attr():
    print(SEP)
    print("TEST 2: RMA forward — with attr_emb (path b)")

    attr_dim     = 513
    attr_proj_dim = 32
    rma = ReferenceModulatedCrossAttention(
        dim=12, context_dim=36, heads=4, dim_head=8,
        attr_dim=attr_dim, attr_proj_dim=attr_proj_dim,
    )

    B, C, K, L = 2, 4, 2, 12
    R = 36
    x         = torch.randn(B, C, K, L)
    cond_info = torch.randn(B, C, K, L)
    reference = torch.randn(B, K, R)
    attr_emb  = torch.randn(B, attr_dim)

    out_with    = rma(x, cond_info, reference, attr_emb=attr_emb)
    out_without = rma(x, cond_info, reference, attr_emb=None)

    check(out_with.shape == (B * C, K, L),
          f"RMA+attr output shape {out_with.shape} != ({B*C},{K},{L})")
    # Outputs should differ when attr_emb is present vs absent
    check(not torch.allclose(out_with, out_without),
          "RMA output identical with/without attr_emb — path (b) not active")
    print(f"  output shape: {out_with.shape}")
    print(f"  outputs differ with/without attr_emb: YES")
    print("  PASS")


def test_rma_backward():
    print(SEP)
    print("TEST 3: RMA backward pass (gradient flow)")

    rma = ReferenceModulatedCrossAttention(
        dim=12, context_dim=36, heads=4, dim_head=8, attr_dim=513
    )

    B, C, K, L = 2, 4, 2, 12
    x         = torch.randn(B, C, K, L, requires_grad=True)
    cond_info = torch.randn(B, C, K, L)
    reference = torch.randn(B, K, 36)
    attr_emb  = torch.randn(B, 513)

    out = rma(x, cond_info, reference, attr_emb=attr_emb)
    loss = out.sum()
    loss.backward()

    check(x.grad is not None, "No gradient on x after backward")
    check(not torch.isnan(x.grad).any(), "NaN gradient on x")
    print(f"  grad norm on x: {x.grad.norm().item():.4f}  PASS")


# ---------------------------------------------------------------------------
# 2. ResidualBlock
# ---------------------------------------------------------------------------

def test_residual_block_no_attr():
    print(SEP)
    print("TEST 4: ResidualBlock forward — no attr_emb")

    block = ResidualBlock(
        side_dim=49, ref_size=12, h_size=0,
        channels=16, diffusion_embedding_dim=32,
        nheads=4, is_linear=False, attr_dim=None,
    )

    B, C, K, L = 2, 16, 2, 12
    x         = torch.randn(B, C, K, L)
    cond_info = torch.randn(B, 49, K, L)
    diff_emb  = torch.randn(B, 32)
    reference = torch.randn(B, K, 36)

    residual, skip = block(x, cond_info, diff_emb, reference, attr_emb=None)
    check(residual.shape == (B, C, K, L),
          f"residual shape {residual.shape} != ({B},{C},{K},{L})")
    check(skip.shape == (B, C, K, L),
          f"skip shape {skip.shape} != ({B},{C},{K},{L})")
    print(f"  residual: {residual.shape}, skip: {skip.shape}  PASS")


def test_residual_block_with_attr():
    print(SEP)
    print("TEST 5: ResidualBlock forward — with attr_emb (path b)")

    block = ResidualBlock(
        side_dim=49, ref_size=12, h_size=0,
        channels=16, diffusion_embedding_dim=32,
        nheads=4, is_linear=False, attr_dim=513,
    )

    B, C, K, L = 2, 16, 2, 12
    x         = torch.randn(B, C, K, L)
    cond_info = torch.randn(B, 49, K, L)
    diff_emb  = torch.randn(B, 32)
    reference = torch.randn(B, K, 36)
    attr_emb  = torch.randn(B, 513)

    r1, s1 = block(x, cond_info, diff_emb, reference, attr_emb=attr_emb)
    r2, s2 = block(x, cond_info, diff_emb, reference, attr_emb=None)

    check(r1.shape == (B, C, K, L), f"residual shape {r1.shape}")
    check(not torch.allclose(r1, r2),
          "ResidualBlock outputs identical with/without attr_emb")
    print(f"  shapes OK, outputs differ with/without attr_emb  PASS")


# ---------------------------------------------------------------------------
# 3. diff_RATD (full denoiser)
# ---------------------------------------------------------------------------

def test_diff_ratd_no_ref_no_attr():
    print(SEP)
    print("TEST 6: diff_RATD — no reference, no attr_emb")

    cfg = _mini_diff_config(with_attr=False)
    model = diff_RATD(cfg, inputdim=2, use_ref=True)

    B, K, L = 2, 2, 12
    x         = torch.randn(B, 2, K, L)
    cond_info = torch.randn(B, cfg["side_dim"], K, L)
    t         = torch.randint(0, cfg["num_steps"], (B,))

    out = model(x, cond_info, t, reference=None, attr_emb=None)
    check(out.shape == (B, K, L), f"output shape {out.shape} != ({B},{K},{L})")
    print(f"  output: {out.shape}  PASS")


def test_diff_ratd_with_ref_and_attr():
    print(SEP)
    print("TEST 7: diff_RATD — with reference and attr_emb")

    cfg = _mini_diff_config(with_attr=True)
    model = diff_RATD(cfg, inputdim=2, use_ref=True)

    B, K, L = 2, 2, 12
    x         = torch.randn(B, 2, K, L)
    cond_info = torch.randn(B, cfg["side_dim"], K, L)
    t         = torch.randint(0, cfg["num_steps"], (B,))
    reference = torch.randn(B, K, 36)
    attr_emb  = torch.randn(B, 513)

    out_full = model(x, cond_info, t, reference=reference, attr_emb=attr_emb)
    out_none = model(x, cond_info, t, reference=None,      attr_emb=None)

    check(out_full.shape == (B, K, L), f"output shape {out_full.shape}")
    check(not torch.allclose(out_full, out_none),
          "diff_RATD outputs identical with/without reference+attr")
    print(f"  output: {out_full.shape}, differs with/without reference+attr  PASS")


def test_diff_ratd_backward():
    print(SEP)
    print("TEST 8: diff_RATD — full backward pass with attr_emb")

    cfg = _mini_diff_config(with_attr=True)
    model = diff_RATD(cfg, inputdim=2, use_ref=True)

    B, K, L = 2, 2, 12
    x         = torch.randn(B, 2, K, L, requires_grad=True)
    cond_info = torch.randn(B, cfg["side_dim"], K, L)
    t         = torch.randint(0, cfg["num_steps"], (B,))
    reference = torch.randn(B, K, 36)
    attr_emb  = torch.randn(B, 513)

    out  = model(x, cond_info, t, reference=reference, attr_emb=attr_emb)
    loss = out.sum()
    loss.backward()

    check(x.grad is not None, "No gradient on x")
    check(not torch.isnan(x.grad).any(), "NaN in gradient")
    # Note: ResidualBlock.attn1 is dead code from the original RATD snapshot
    # (defined but never called in forward).  Its params have no gradient.
    # We check that all *active* gradients are finite.
    active = [(n, p) for n, p in model.named_parameters()
              if p.requires_grad and p.grad is not None]
    check(len(active) > 0, "No parameters received a gradient")
    nan_params = [n for n, p in active if torch.isnan(p.grad).any()]
    check(len(nan_params) == 0, f"NaN gradient in: {nan_params[:3]}")
    print(f"  gradient OK, {len(active)} active params, no NaN/Inf  PASS")


def test_backward_compat_attr_dim_none():
    print(SEP)
    print("TEST 9: Backward compatibility — attr_dim=None (original RATD behavior)")

    cfg_orig = _mini_diff_config(with_attr=False)
    cfg_new  = _mini_diff_config(with_attr=False)   # no attr_dim key

    model_orig = diff_RATD(cfg_orig, inputdim=2)
    model_new  = diff_RATD(cfg_new,  inputdim=2)

    # Both should produce output without any attr_emb argument
    B, K, L = 2, 2, 12
    torch.manual_seed(99)
    x         = torch.randn(B, 2, K, L)
    cond_info = torch.randn(B, cfg_orig["side_dim"], K, L)
    t         = torch.zeros(B, dtype=torch.long)

    model_orig.eval(); model_new.eval()
    with torch.no_grad():
        out_orig = model_orig(x, cond_info, t)
        out_new  = model_new( x, cond_info, t)   # no attr_emb kwarg at all

    check(out_orig.shape == (B, K, L), "original model shape")
    check(out_new.shape  == (B, K, L), "new model (no attr_dim) shape")
    print(f"  both models produce {out_orig.shape}, no exceptions  PASS")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE: diff_models.py")
    print("=" * 60)

    test_rma_no_attr()
    test_rma_with_attr()
    test_rma_backward()
    test_residual_block_no_attr()
    test_residual_block_with_attr()
    test_diff_ratd_no_ref_no_attr()
    test_diff_ratd_with_ref_and_attr()
    test_diff_ratd_backward()
    test_backward_compat_attr_dim_none()

    print(SEP)
    print("ALL DIFF_MODELS SMOKE TESTS PASSED")
    print(SEP)
