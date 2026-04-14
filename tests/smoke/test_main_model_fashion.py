"""tests/smoke/test_main_model_fashion.py

Smoke tests for main_model_fashion.py (RATD_Fashion).

Tests:
  - Model construction from config: side_dim and attr_dim are correctly
    propagated to the diffmodel after the super().__init__() rebuild.
  - get_side_info: correct shape (no path-a channels; attrs enter via RMA).
  - Forward pass (is_train=1 and is_train=0): returns scalar loss with
    gradient.
  - evaluate(): returns (samples, c_target, target_mask, obs_mask, obs_tp)
    with correct shapes for any n_samples.
  - n_obs masking: gt_mask correctly reveals first n_obs weeks only.
  - process_data(): tensors are transposed to (B, K, L) convention.

Expected outcome: all assertions pass, PASS printed at the end.
No external data files are required — all inputs are synthetic.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import yaml

from main_model_fashion import RATD_Fashion

SEP = "-" * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mini_config() -> dict:
    """Minimal config for fast CPU tests (small embedding dims)."""
    return {
        "train": {
            "epochs": 2, "batch_size": 4, "lr": 1e-3, "itr_per_epoch": 1e8,
        },
        "diffusion": {
            "channels": 16,
            "diffusion_embedding_dim": 32,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": 10,
            "schedule": "quad",
            "is_linear": False,
            "layers": 2,
            "nheads": 4,
            "h_size": 0,
            "ref_size": 12,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 32,
            "featureemb": 8,
            "target_strategy": "test",
            "use_reference": True,
        },
    }


def _make_batch(B: int = 2, n_obs: int = 2) -> dict:
    """Create a synthetic batch matching Dataset_Visuelle2 output format."""
    L, K = 12, 2
    gt = torch.zeros(B, L, K)
    if n_obs > 0:
        gt[:, :n_obs, :] = 1.0
    return {
        "observed_data": torch.randn(B, L, K),
        "observed_mask": torch.ones(B, L, K),
        "gt_mask":       gt,
        "timepoints":    torch.arange(L).float().unsqueeze(0).expand(B, -1),
        "reference":     torch.randn(B, 36, K),
        "product_emb":   torch.randn(B, 513),
    }


def check(condition: bool, msg: str):
    if not condition:
        raise AssertionError(f"FAIL: {msg}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_model_construction():
    print(SEP)
    print("TEST 1: RATD_Fashion construction — side_dim and attr_dim")

    config = _mini_config()
    model  = RATD_Fashion(config, device="cpu")

    # emb_total_dim = timeemb(32) + featureemb(8) + 1(cond_mask) = 41
    # side_dim equals emb_total_dim exactly — no path-a channels added
    expected_emb_total = 32 + 8 + 1
    check(model.emb_total_dim == expected_emb_total,
          f"emb_total_dim {model.emb_total_dim} != {expected_emb_total}")

    actual_side_dim = config["diffusion"]["side_dim"]
    check(actual_side_dim == expected_emb_total,
          f"config side_dim {actual_side_dim} != {expected_emb_total}")

    check(config["diffusion"]["attr_dim"] == 513,
          f"config attr_dim {config['diffusion']['attr_dim']} != 513")

    rma = model.diffmodel.residual_layers[0].RMA
    check(rma.attr_dim == 513,
          f"RMA.attr_dim {rma.attr_dim} != 513")

    print(f"  emb_total_dim={model.emb_total_dim}, "
          f"side_dim={actual_side_dim}, RMA.attr_dim={rma.attr_dim}  PASS")


def test_side_info_shape():
    print(SEP)
    print("TEST 2: get_side_info — shape equals emb_total_dim (no path-a channels)")

    config = _mini_config()
    model  = RATD_Fashion(config, device="cpu")

    B, K, L = 2, 2, 12
    cond_mask   = torch.zeros(B, K, L)
    observed_tp = torch.arange(L).float().unsqueeze(0).expand(B, -1)

    side_info = model.get_side_info(observed_tp, cond_mask)

    expected_C = model.emb_total_dim   # timeemb + featureemb + 1 = 41
    check(side_info.shape == (B, expected_C, K, L),
          f"side_info shape {side_info.shape} != ({B},{expected_C},{K},{L})")

    print(f"  side_info: {side_info.shape}  (C={expected_C} = emb_total_dim)  PASS")


def test_process_data():
    print(SEP)
    print("TEST 3: process_data — transpose (B,L,K) -> (B,K,L)")

    config = _mini_config()
    model  = RATD_Fashion(config, device="cpu")
    batch  = _make_batch(B=3, n_obs=2)

    (obs_d, obs_m, obs_tp, gt_m, _, _, ref, pemb) = model.process_data(batch)

    check(obs_d.shape  == (3, 2, 12), f"obs_data {obs_d.shape}")
    check(obs_m.shape  == (3, 2, 12), f"obs_mask {obs_m.shape}")
    check(gt_m.shape   == (3, 2, 12), f"gt_mask  {gt_m.shape}")
    check(obs_tp.shape == (3, 12),    f"obs_tp   {obs_tp.shape}")
    check(ref.shape    == (3, 2, 36), f"ref      {ref.shape}")
    check(pemb.shape   == (3, 513),   f"pemb     {pemb.shape}")
    print(f"  all shapes correct  PASS")


def test_gt_mask_nobs():
    print(SEP)
    print("TEST 4: gt_mask — first n_obs weeks revealed, rest zero")

    config = _mini_config()
    model  = RATD_Fashion(config, device="cpu")

    for n_obs in [0, 1, 2, 4]:
        batch = _make_batch(B=2, n_obs=n_obs)
        (_, _, _, gt_m, _, _, _, _) = model.process_data(batch)
        # gt_m is (B, K, L) in (B, K, L) order after permute
        # First n_obs time steps should be 1, rest 0
        check(gt_m[:, :, :n_obs].sum().item() == 2 * 2 * n_obs,
              f"n_obs={n_obs}: expected {2*2*n_obs} revealed, got {gt_m[:,:,:n_obs].sum().item()}")
        check(gt_m[:, :, n_obs:].sum().item() == 0,
              f"n_obs={n_obs}: future steps not fully masked")

    print(f"  n_obs ∈ {{0,1,2,4}} masking all correct  PASS")


def test_forward_train():
    print(SEP)
    print("TEST 5: forward (is_train=1) — scalar loss with gradient")

    config = _mini_config()
    model  = RATD_Fashion(config, device="cpu")
    model.train()
    batch  = _make_batch(B=2, n_obs=0)

    loss = model(batch, is_train=1)

    check(loss.dim() == 0, f"loss not scalar: shape {loss.shape}")
    check(not torch.isnan(loss), "loss is NaN")
    loss.backward()

    # Note: ResidualBlock.attn1 is defined in the original RATD code but is
    # never called in the forward pass (dead code from the research snapshot).
    # Its parameters correctly have no gradient.  We only verify that all
    # *active* parameters (those with .grad set) are free of NaN/Inf.
    active_grads = [p.grad for p in model.parameters()
                    if p.requires_grad and p.grad is not None]
    check(len(active_grads) > 0, "No parameters have gradients at all")
    nan_grads = [g for g in active_grads if torch.isnan(g).any()]
    inf_grads = [g for g in active_grads if torch.isinf(g).any()]
    check(len(nan_grads) == 0, f"{len(nan_grads)} params have NaN gradient")
    check(len(inf_grads) == 0, f"{len(inf_grads)} params have Inf gradient")
    print(f"  loss={loss.item():.6f}, {len(active_grads)} params have valid grads  PASS")


def test_forward_val():
    print(SEP)
    print("TEST 6: forward (is_train=0) — validation loss, no NaN")

    config = _mini_config()
    model  = RATD_Fashion(config, device="cpu")
    model.eval()
    batch  = _make_batch(B=2, n_obs=2)

    with torch.no_grad():
        loss = model(batch, is_train=0)

    check(loss.dim() == 0, f"val loss not scalar: {loss.shape}")
    check(not torch.isnan(loss), "val loss is NaN")
    print(f"  val loss={loss.item():.6f}  PASS")


def test_evaluate_shapes():
    print(SEP)
    print("TEST 7: evaluate — output shapes for n_samples=3")

    config   = _mini_config()
    # Use fewer diffusion steps to keep test fast
    config["diffusion"]["num_steps"] = 3
    model    = RATD_Fashion(config, device="cpu")
    model.eval()
    batch    = _make_batch(B=2, n_obs=0)
    n_samples = 3

    with torch.no_grad():
        samples, c_target, target_mask, obs_mask, obs_tp = model.evaluate(batch, n_samples)

    check(samples.shape    == (2, n_samples, 2, 12), f"samples    {samples.shape}")
    check(c_target.shape   == (2, 2, 12),            f"c_target   {c_target.shape}")
    check(target_mask.shape == (2, 2, 12),            f"target_mask{target_mask.shape}")
    check(obs_mask.shape   == (2, 2, 12),            f"obs_mask   {obs_mask.shape}")
    check(obs_tp.shape     == (2, 12),               f"obs_tp     {obs_tp.shape}")
    print(f"  samples={samples.shape}  PASS")


def test_cold_start_target_mask():
    print(SEP)
    print("TEST 8: cold-start (n_obs=0) — entire horizon is target")

    config = _mini_config()
    config["diffusion"]["num_steps"] = 3
    model  = RATD_Fashion(config, device="cpu")
    model.eval()
    batch  = _make_batch(B=2, n_obs=0)

    with torch.no_grad():
        _, _, target_mask, obs_mask, _ = model.evaluate(batch, n_samples=1)

    # n_obs=0 → gt_mask is all zeros → target_mask = obs_mask - gt_mask = obs_mask = all ones
    check(target_mask.sum().item() == 2 * 2 * 12,
          f"cold-start target_mask sum {target_mask.sum().item()} != {2*2*12}")
    print(f"  all {int(target_mask.sum())} positions are forecasting targets  PASS")


def test_config_from_yaml():
    print(SEP)
    print("TEST 9: build from real YAML — config/visuelle2.yaml")

    yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "visuelle2.yaml"
    )
    if not os.path.exists(yaml_path):
        print(f"  SKIP: {yaml_path} not found")
        return

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    model = RATD_Fashion(config, device="cpu")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Built from visuelle2.yaml — params: {n:,}  PASS")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE: main_model_fashion.py")
    print("=" * 60)

    test_model_construction()
    test_side_info_shape()
    test_process_data()
    test_gt_mask_nobs()
    test_forward_train()
    test_forward_val()
    test_evaluate_shapes()
    test_cold_start_target_mask()
    test_config_from_yaml()

    print(SEP)
    print("ALL MAIN_MODEL_FASHION SMOKE TESTS PASSED")
    print(SEP)
