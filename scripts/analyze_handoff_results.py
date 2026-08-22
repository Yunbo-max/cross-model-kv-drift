"""Add paired bootstrap intervals to a handoff evaluation result."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text())
    cases = payload["cases"]
    rng = np.random.default_rng(args.seed)
    report = {"examples": len(cases), "bootstrap_samples": args.samples, "seed": args.seed}
    for suffix in ("", "_norm"):
        native = np.array([case[f"native{suffix}_correct"] for case in cases], dtype=float)
        transfer = np.array([case[f"transfer{suffix}_correct"] for case in cases], dtype=float)
        indices = rng.integers(0, len(cases), size=(args.samples, len(cases)))
        deltas = (transfer[indices] - native[indices]).mean(axis=1)
        native_means = native[indices].mean(axis=1)
        retention = transfer[indices].mean(axis=1) / native_means
        name = "raw" if not suffix else "normalized"
        report[name] = {
            "accuracy_delta": float((transfer - native).mean()),
            "accuracy_delta_ci95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
            "retention": float(transfer.mean() / native.mean()),
            "retention_ci95": [float(x) for x in np.quantile(retention, [0.025, 0.975])],
            "native_only_correct": int(np.sum((native == 1) & (transfer == 0))),
            "transfer_only_correct": int(np.sum((native == 0) & (transfer == 1))),
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
