"""X5 unit tests: run everything with one command.

  python modeling/tests/run_all.py --scratch <empty scratch dir>

Synthetic data only, except the feature rebuild test (reruns build_features.py on the raw CSVs into the scratch
dir; the script never reads target). No target, no holdout, no real-data training.
Outputs: checks/unit_tests.json and unit_tests.out.txt (an existing file is not overwritten).
Options: --out-dir (default checks), --skip-real-rebuild (synthetic tests only).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "modeling")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import test_kl  # noqa: E402
import test_learners  # noqa: E402
import test_qini  # noqa: E402
import test_build_features  # noqa: E402
import test_tune  # noqa: E402
import tune as TU  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "checks")
    ap.add_argument("--skip-real-rebuild", action="store_true")
    a = ap.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=True)
    assert not any(a.scratch.iterdir()), f"scratch dir must be empty: {a.scratch}"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    import lightgbm, optuna, sklearn, sklift, xgboost  # noqa: E401
    started = time.time()
    env = {"python": platform.python_version(), "platform": platform.platform(),
           **{m.__name__: m.__version__ for m in (np, pd, xgboost, lightgbm, optuna, sklift, sklearn)}}
    log(f"X5 unit tests  {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    log(f"env {env}")
    log(f"code fingerprint {TU.code_fingerprint()}")
    results, timing = {}, {}
    for name, fn in (("qini", lambda: test_qini.run(log)),
                     ("kl_filter", lambda: test_kl.run(log)),
                     ("learners", lambda: test_learners.run(log)),
                     ("tuning_frame", lambda: test_tune.run(log, a.scratch)),
                     ("build_features", lambda: test_build_features.run(log, a.scratch, real=not a.skip_real_rebuild))):
        log(f"\n== {name}")
        t0 = time.time()
        results[name] = fn()
        timing[name] = round(time.time() - t0, 1)
        log(f"== {name}: {'PASS' if results[name]['pass'] else 'FAIL'} ({timing[name]}s)")
    overall = all(r["pass"] for r in results.values())
    log(f"\nOVERALL {'PASS' if overall else 'FAIL'}  total {time.time() - started:.1f}s  timing {timing}")
    report = {"step": "unit tests (synthetic only; no target/holdout read)",
              "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "env": env,
              "code_fingerprint": TU.code_fingerprint(),
              "code_sha256": {p.name: TU.sha256_file(p) for p in TU.CODE_FILES},
              "test_sha256": {p.name: TU.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
              "real_rebuild_skipped": a.skip_real_rebuild, "timing_seconds": timing,
              "total_seconds": round(time.time() - started, 1), "overall_pass": overall, "results": results}
    jpath, tpath = a.out_dir / "unit_tests.json", a.out_dir / "unit_tests.out.txt"
    for p in (jpath, tpath):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    jpath.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    tpath.write_text("\n".join(lines) + "\n")
    print(f"wrote {jpath} and {tpath}")


if __name__ == "__main__":
    main()
