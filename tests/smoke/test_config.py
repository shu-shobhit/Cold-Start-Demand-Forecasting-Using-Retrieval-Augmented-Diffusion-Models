"""tests/smoke/test_config.py

Smoke tests for YAML configuration files in config/.

Tests:
  - All three YAML files (base.yaml, base_forecasting.yaml, visuelle2.yaml)
    load without error and parse to Python dicts.
  - visuelle2.yaml contains all required keys for RATD_Fashion construction.
  - RATD_Fashion can be built directly from visuelle2.yaml without any
    additional key injection (h_size, ref_size already in diffusion block).
  - config values have expected types and ranges (e.g. num_steps > 0,
    batch_size > 0, attr_emb_dim > 0).

Expected outcome: all assertions pass, PASS printed at the end.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import yaml

SEP = "-" * 60
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")


def check(condition: bool, msg: str):
    if not condition:
        raise AssertionError(f"FAIL: {msg}")


def _load(filename: str) -> dict:
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_base_yaml_loads():
    print(SEP)
    print("TEST 1: base.yaml — loads without error")
    cfg = _load("base.yaml")
    check(isinstance(cfg, dict), "base.yaml did not parse to dict")
    check("train" in cfg,      "base.yaml missing 'train' key")
    check("diffusion" in cfg,  "base.yaml missing 'diffusion' key")
    check("model" in cfg,      "base.yaml missing 'model' key")
    print(f"  keys: {list(cfg.keys())}  PASS")


def test_base_forecasting_yaml_loads():
    print(SEP)
    print("TEST 2: base_forecasting.yaml — loads without error")
    cfg = _load("base_forecasting.yaml")
    check(isinstance(cfg, dict), "base_forecasting.yaml did not parse to dict")
    check("train" in cfg,      "missing 'train'")
    check("diffusion" in cfg,  "missing 'diffusion'")
    check("model" in cfg,      "missing 'model'")
    print(f"  keys: {list(cfg.keys())}  PASS")


def test_visuelle2_required_train_keys():
    print(SEP)
    print("TEST 3: visuelle2.yaml — required train keys present and valid")
    cfg = _load("visuelle2.yaml")
    t = cfg["train"]
    for key in ("epochs", "batch_size", "lr", "itr_per_epoch"):
        check(key in t, f"train.{key} missing")
    check(t["epochs"] > 0,      f"train.epochs={t['epochs']} must be > 0")
    check(t["batch_size"] > 0,  f"train.batch_size={t['batch_size']} must be > 0")
    check(t["lr"] > 0,          f"train.lr={t['lr']} must be > 0")
    print(f"  train block OK  PASS")


def test_visuelle2_required_diffusion_keys():
    print(SEP)
    print("TEST 4: visuelle2.yaml — required diffusion keys present and valid")
    cfg = _load("visuelle2.yaml")
    d = cfg["diffusion"]
    required = ("layers", "channels", "nheads", "diffusion_embedding_dim",
                "beta_start", "beta_end", "num_steps", "schedule",
                "is_linear", "h_size", "ref_size")
    for key in required:
        check(key in d, f"diffusion.{key} missing")

    check(d["num_steps"] > 0,       f"num_steps={d['num_steps']} must be > 0")
    check(d["layers"] > 0,          f"layers={d['layers']} must be > 0")
    check(d["channels"] > 0,        f"channels={d['channels']} must be > 0")
    check(d["h_size"] >= 0,         f"h_size={d['h_size']} must be >= 0")
    check(d["ref_size"] > 0,        f"ref_size={d['ref_size']} must be > 0")
    check(d["ref_size"] == 12,      f"ref_size={d['ref_size']} should equal pred_len=12")
    check(d["schedule"] in ("quad", "linear"),
          f"schedule={d['schedule']} unknown")
    check(0 < d["beta_start"] < d["beta_end"],
          f"beta schedule invalid: {d['beta_start']} >= {d['beta_end']}")
    print(f"  diffusion block OK  PASS")


def test_visuelle2_required_model_keys():
    print(SEP)
    print("TEST 5: visuelle2.yaml — required model keys present and valid")
    cfg = _load("visuelle2.yaml")
    m = cfg["model"]
    required = ("is_unconditional", "timeemb", "featureemb",
                "target_strategy", "use_reference", "attr_emb_dim")
    for key in required:
        check(key in m, f"model.{key} missing")

    check(m["is_unconditional"] in (0, 1, True, False),
          f"is_unconditional invalid: {m['is_unconditional']}")
    check(m["timeemb"] > 0,     f"timeemb={m['timeemb']} must be > 0")
    check(m["featureemb"] > 0,  f"featureemb={m['featureemb']} must be > 0")
    check(m["attr_emb_dim"] > 0, f"attr_emb_dim={m['attr_emb_dim']} must be > 0")
    check(m["use_reference"] in (True, False, 0, 1),
          f"use_reference invalid: {m['use_reference']}")
    print(f"  model block OK  PASS")


def test_visuelle2_builds_ratd_fashion():
    print(SEP)
    print("TEST 6: visuelle2.yaml — RATD_Fashion builds cleanly from this config")

    from main_model_fashion import RATD_Fashion

    cfg = _load("visuelle2.yaml")
    model = RATD_Fashion(cfg, device="cpu")

    # After construction, diffusion config should be mutated with side_dim + attr_dim
    check("side_dim" in cfg["diffusion"],
          "config missing diffusion.side_dim after construction")
    check("attr_dim" in cfg["diffusion"],
          "config missing diffusion.attr_dim after construction")
    check(cfg["diffusion"]["attr_dim"] == 513,
          f"attr_dim={cfg['diffusion']['attr_dim']} != 513")

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(n > 0, "model has no trainable parameters")
    print(f"  RATD_Fashion built: {n:,} parameters  PASS")


def test_visuelle2_reference_size_consistency():
    print(SEP)
    print("TEST 7: visuelle2.yaml — ref_size * 3 = 36 (k=3 retrieved references)")

    cfg = _load("visuelle2.yaml")
    ref_size = cfg["diffusion"]["ref_size"]
    K_REFS   = 3
    expected_ref_len = ref_size * K_REFS

    check(expected_ref_len == 36,
          f"ref_size*k = {ref_size}*{K_REFS}={expected_ref_len} should be 36")
    print(f"  ref_size={ref_size}, k=3 → ref_tensor_len={expected_ref_len}  PASS")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE: config/*.yaml")
    print("=" * 60)

    test_base_yaml_loads()
    test_base_forecasting_yaml_loads()
    test_visuelle2_required_train_keys()
    test_visuelle2_required_diffusion_keys()
    test_visuelle2_required_model_keys()
    test_visuelle2_builds_ratd_fashion()
    test_visuelle2_reference_size_consistency()

    print(SEP)
    print("ALL CONFIG SMOKE TESTS PASSED")
    print(SEP)
