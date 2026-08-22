"""Create KVBridge's fail-closed evidence manifest for sequential capture."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open

from kvbridge.config import ModelSignature
from kvbridge.evidence import calibration_contract_sha256, sha256_file
from kvbridge.io import atomic_write_text
from kvbridge.provenance import code_revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("calibration_dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = ModelSignature.from_dict(config["source"])
    target = ModelSignature.from_dict(config["target"])
    paths = sorted(args.calibration_dir.glob("*.safetensors"))
    records = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as stream:
            sequence_id = (stream.metadata() or {}).get("sequence_id")
        records.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "sequence_id": sequence_id,
        })
    manifest = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "calibration_contract_sha256": calibration_contract_sha256(args.config),
        "code_revision": code_revision(),
        "dataset": config["calibration"]["dataset"],
        "dataset_revision": config["calibration"]["dataset_revision"],
        "split": config["calibration"]["split"],
        "sequences": len(paths),
        "tokens": config["calibration"]["tokens"],
        "source_fingerprint": source.fingerprint,
        "target_fingerprint": target.fingerprint,
        "source_signature": source.to_dict(),
        "target_signature": target.to_dict(),
        "tokenizer_hash": source.tokenizer_hash,
        "capture_mode": "sequential-model-loading",
        "persisted_token_stride": int(config.get("persisted_token_stride", 1)),
        "persisted_keys_are_content": bool(config.get("persisted_token_stride", 1) > 1),
        "shards": records,
    }
    atomic_write_text(
        args.calibration_dir / "capture_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
