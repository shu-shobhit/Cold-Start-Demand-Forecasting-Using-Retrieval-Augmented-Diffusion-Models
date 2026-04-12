"""tests/smoke/run_all.py

Master runner for all smoke tests.

Executes each test module in a subprocess so that one failure does not
prevent other tests from running.  Prints a summary table at the end.

Usage:
    # From project root:
    conda run -n ML python tests/smoke/run_all.py

    # Or from tests/smoke/:
    conda run -n ML python run_all.py
"""

import subprocess
import sys
import os

# Modules to run, in dependency order (data-free first)
TEST_MODULES = [
    "test_diff_models.py",
    "test_main_model_fashion.py",
    "test_utils.py",
    "test_config.py",
    "test_dataset_visuelle2.py",   # auto-skips if data absent
]

SMOKE_DIR = os.path.dirname(os.path.abspath(__file__))
SEP       = "=" * 60


def run_module(module: str) -> tuple:
    """Run one test module as a subprocess.

    Returns:
        (module, passed: bool, stdout: str, stderr: str)
    """
    path = os.path.join(SMOKE_DIR, module)
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        cwd=os.path.join(SMOKE_DIR, "..", ".."),  # project root as cwd
    )
    passed = result.returncode == 0
    return module, passed, result.stdout, result.stderr


def main():
    print(SEP)
    print("RATD_Fashion — Smoke Test Suite")
    print(SEP)

    results = []
    for module in TEST_MODULES:
        print(f"\nRunning {module} ...")
        module, passed, stdout, stderr = run_module(module)
        results.append((module, passed))

        # Always show output so failures are diagnosable
        if stdout.strip():
            print(stdout)
        if stderr.strip():
            # Filter the known benign transformer UserWarning
            stderr_lines = [
                l for l in stderr.splitlines()
                if "enable_nested_tensor" not in l
                and "UserWarning" not in l
                and "warnings.warn" not in l
            ]
            if stderr_lines:
                print("STDERR:", "\n".join(stderr_lines))

    # Summary
    print("\n" + SEP)
    print("SUMMARY")
    print(SEP)
    n_pass = n_fail = n_skip = 0
    for module, passed in results:
        if not passed:
            status = "FAIL"
            n_fail += 1
        else:
            # A module that prints SKIP but exits 0 counts as skip
            status = "PASS"
            n_pass += 1
        print(f"  {status}  {module}")

    print(SEP)
    print(f"  {n_pass} passed  |  {n_fail} failed")
    print(SEP)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
