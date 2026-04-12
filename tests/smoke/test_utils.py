"""tests/smoke/test_utils.py

Smoke tests for utils.py.

Tests:
  - quantile_loss: known analytical values.
  - calc_denominator: sum of absolute target values.
  - calc_wape: known WAPE fractions (perfect forecast → 0, off-by-factor → known).
  - calc_quantile_CRPS: monotonicity and non-negativity.
  - calc_quantile_CRPS_sum: shape and non-negativity.
  - train loop: runs for one mini-epoch on a synthetic model without error.
  - evaluate: runs end-to-end on a synthetic model and produces metric output.

Expected outcome: all assertions pass, PASS printed at the end.
No external data files required.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import tempfile
import torch
import numpy as np

from utils import (
    quantile_loss,
    calc_denominator,
    calc_wape,
    calc_quantile_CRPS,
    calc_quantile_CRPS_sum,
    train,
    evaluate,
)

SEP = "-" * 60


def check(condition: bool, msg: str):
    if not condition:
        raise AssertionError(f"FAIL: {msg}")


# ---------------------------------------------------------------------------
# Synthetic model for train / evaluate smoke tests
# ---------------------------------------------------------------------------

class _TrivialModel(torch.nn.Module):
    """Minimal model that satisfies the train/evaluate interface."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Linear(1, 1)

    def forward(self, batch, is_train=1):
        return self.p.weight.sum() * 0 + torch.tensor(1.0, requires_grad=True)

    def evaluate(self, batch, n_samples):
        B, K, L = 2, 2, 12
        samples      = torch.randn(B, n_samples, K, L)
        c_target     = torch.randn(B, K, L)
        target_mask  = torch.ones(B, K, L)
        obs_mask     = torch.ones(B, K, L)
        obs_tp       = torch.arange(L).float().unsqueeze(0).expand(B, -1)
        return samples, c_target, target_mask, obs_mask, obs_tp

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        super().eval()
        return self


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_quantile_loss_perfect():
    print(SEP)
    print("TEST 1: quantile_loss — perfect forecast gives 0")

    target   = torch.tensor([[1.0, 2.0, 3.0]])
    forecast = torch.tensor([[1.0, 2.0, 3.0]])
    mask     = torch.ones_like(target)

    loss = quantile_loss(target, forecast, q=0.5, eval_points=mask)
    check(loss.item() == 0.0, f"perfect forecast quantile_loss={loss.item()} != 0")
    print(f"  quantile_loss(perfect, q=0.5) = {loss.item()}  PASS")


def test_quantile_loss_known():
    print(SEP)
    print("TEST 2: quantile_loss — analytical check (q=0.5, median loss = MAE)")

    target   = torch.tensor([[0.0, 2.0]])
    forecast = torch.tensor([[1.0, 1.0]])  # forecast > target[0], < target[1]
    mask     = torch.ones_like(target)

    # q=0.5: loss = 2 * sum(|f-t| * mask * (t<=f ? 0.5 : -0.5+1))
    # For element 0: f=1 > t=0 → contrib = |1-0| * (1-0.5) = 0.5
    # For element 1: f=1 < t=2 → contrib = |1-2| * (1-0.5) = 0.5
    # total = 2 * (0.5+0.5) = 2.0
    loss = quantile_loss(target, forecast, q=0.5, eval_points=mask)
    expected = 2.0
    check(abs(loss.item() - expected) < 1e-5,
          f"quantile_loss={loss.item():.6f} != {expected}")
    print(f"  quantile_loss(q=0.5) = {loss.item():.4f} (expected {expected})  PASS")


def test_calc_denominator():
    print(SEP)
    print("TEST 3: calc_denominator — sum of |target * eval_points|")

    target     = torch.tensor([[1.0, -2.0, 3.0]])
    eval_points = torch.tensor([[1.0,  1.0, 0.0]])   # last point excluded

    denom = calc_denominator(target, eval_points)
    check(denom.item() == 3.0,   # |1| + |-2| + |3|*0
          f"denominator {denom.item()} != 3.0")
    print(f"  denominator = {denom.item()}  PASS")


def test_calc_wape_perfect():
    print(SEP)
    print("TEST 4: calc_wape — perfect forecast → WAPE = 0")

    target   = torch.ones(4, 12, 1)
    forecast = torch.ones(4, 12, 1)
    mask     = torch.ones(4, 12, 1)

    wape = calc_wape(target, forecast, mask)
    check(wape == 0.0, f"WAPE for perfect forecast = {wape} != 0")
    print(f"  WAPE(perfect) = {wape:.4f}  PASS")


def test_calc_wape_known():
    print(SEP)
    print("TEST 5: calc_wape — predict constant 0 → WAPE = 1.0")

    target   = torch.ones(4, 12, 1)     # all 1s
    forecast = torch.zeros(4, 12, 1)    # all 0s
    mask     = torch.ones(4, 12, 1)

    wape = calc_wape(target, forecast, mask)
    # WAPE = sum(|0-1|) / sum(|1|) = 1.0
    check(abs(wape - 1.0) < 1e-5, f"WAPE(predict-zero) = {wape:.6f} != 1.0")
    print(f"  WAPE(predict zeros against ones) = {wape:.4f}  PASS")


def test_calc_wape_scaler():
    print(SEP)
    print("TEST 6: calc_wape — scaler de-normalisation is neutral for ratio")

    target   = torch.ones(4, 12, 1)
    forecast = torch.zeros(4, 12, 1)
    mask     = torch.ones(4, 12, 1)

    # WAPE is scale-invariant (ratio), so scaler should not affect result
    wape_raw    = calc_wape(target, forecast, mask, scaler=1.0)
    wape_scaled = calc_wape(target, forecast, mask, scaler=53.0)
    check(abs(wape_raw - wape_scaled) < 1e-5,
          f"WAPE changes with scaler: {wape_raw:.6f} vs {wape_scaled:.6f}")
    print(f"  WAPE scale-invariant: {wape_raw:.4f} == {wape_scaled:.4f}  PASS")


def test_crps_nonnegative():
    print(SEP)
    print("TEST 7: calc_quantile_CRPS — non-negative and finite")

    N, L, K    = 4, 12, 1
    n_samples  = 10
    target     = torch.randn(N, L, K)
    samples    = torch.randn(N, n_samples, L, K)
    eval_pts   = torch.ones(N, L, K)

    crps = calc_quantile_CRPS(target, samples, eval_pts, 0, 1)
    check(crps >= 0, f"CRPS negative: {crps}")
    check(math.isfinite(crps), f"CRPS not finite: {crps}")
    print(f"  CRPS = {crps:.6f} (≥0, finite)  PASS")


def test_crps_sum_nonnegative():
    print(SEP)
    print("TEST 8: calc_quantile_CRPS_sum — non-negative")

    N, L, K    = 4, 12, 2
    n_samples  = 10
    target     = torch.randn(N, L, K)
    samples    = torch.randn(N, n_samples, L, K)
    eval_pts   = torch.ones(N, L, K)

    crps_sum = calc_quantile_CRPS_sum(target, samples, eval_pts, 0, 1)
    check(crps_sum >= 0, f"CRPS_sum negative: {crps_sum}")
    print(f"  CRPS_sum = {crps_sum:.6f}  PASS")


def test_train_one_epoch():
    print(SEP)
    print("TEST 9: train — runs one epoch without error")

    model  = _TrivialModel()
    config = {"epochs": 1, "lr": 1e-3, "itr_per_epoch": 2}

    # Minimal dataloader: two batches of empty dicts
    loader = [{"dummy": 0}, {"dummy": 1}]

    with tempfile.TemporaryDirectory() as tmpdir:
        train(model, config, loader, foldername=tmpdir)
        ckpt = os.path.join(tmpdir, "model.pth")
        check(os.path.exists(ckpt), f"model.pth not saved to {tmpdir}")

    print("  1 epoch ran, checkpoint saved  PASS")


def test_evaluate_runs():
    print(SEP)
    print("TEST 10: evaluate — runs end-to-end, prints metrics")

    model  = _TrivialModel()
    loader = [{"dummy": i} for i in range(3)]

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluate(model, loader, nsample=2, scaler=1, mean_scaler=0, foldername=tmpdir)
        results_pk = os.path.join(tmpdir, "result_nsample2.pk")
        check(os.path.exists(results_pk), "result pickle not written")
        gen_pk = os.path.join(tmpdir, "generated_outputs_nsample2.pk")
        check(os.path.exists(gen_pk), "generated_outputs pickle not written")

    print("  evaluate ran, output files written  PASS")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE: utils.py")
    print("=" * 60)

    test_quantile_loss_perfect()
    test_quantile_loss_known()
    test_calc_denominator()
    test_calc_wape_perfect()
    test_calc_wape_known()
    test_calc_wape_scaler()
    test_crps_nonnegative()
    test_crps_sum_nonnegative()
    test_train_one_epoch()
    test_evaluate_runs()

    print(SEP)
    print("ALL UTILS SMOKE TESTS PASSED")
    print(SEP)
