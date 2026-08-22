"""Capture aligned Ministral caches without co-resident source/target weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.fit import CalibrationPair
from kvbridge.huggingface import capture_cache, tokenizer_fingerprint
from kvbridge.io import save_calibration_shard


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_tensors(cache: KVCache) -> dict[str, torch.Tensor]:
    result = {
        "keys": torch.stack(cache.keys).cpu().contiguous(),
        "values": torch.stack(cache.values).cpu().contiguous(),
    }
    if cache.rotary is None:
        raise RuntimeError("captured cache has no rotary factors")
    result["rope_cos"] = cache.rotary.cos.cpu().contiguous()
    result["rope_sin"] = cache.rotary.sin.cpu().contiguous()
    return result


def save_source(
    path: Path, cache: KVCache, input_ids: torch.Tensor, sequence_id: str,
    *, persisted_token_stride: int,
) -> None:
    tensors = {**cache_tensors(cache), "input_ids": input_ids.cpu().contiguous()}
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    save_file(tensors, temporary, metadata={
        "sequence_id": sequence_id,
        "format": "source-kv-v2",
        "keys_are_content": str(cache.keys_are_content).lower(),
        "persisted_token_stride": str(persisted_token_stride),
    })
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_source(path: Path) -> tuple[KVCache, torch.Tensor, str, int]:
    with safe_open(path, framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
        tensors = {name: stream.get_tensor(name) for name in stream.keys()}
    if metadata.get("format") not in {"source-kv-v1", "source-kv-v2"}:
        raise RuntimeError(f"bad source cache format: {path}")
    rotary = RotaryFactors(tensors["rope_cos"], tensors["rope_sin"], False)
    cache = KVCache(
        tensors["keys"], tensors["values"], rotary,
        metadata.get("keys_are_content") == "true",
    )
    return (
        cache, tensors["input_ids"], metadata["sequence_id"],
        int(metadata.get("persisted_token_stride", "1")),
    )


def load_model(local_dir: str):
    architecture = (AutoConfig.from_pretrained(local_dir).architectures or [""])[0]
    common = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if architecture == "Mistral3ForConditionalGeneration":
        return AutoModelForImageTextToText.from_pretrained(
            local_dir,
            device_map={
                "model.vision_tower": "cpu",
                "model.multi_modal_projector": "cpu",
                "model.language_model": 0,
                "lm_head": 0,
            },
            **common,
        ).eval()
    return AutoModelForCausalLM.from_pretrained(
        local_dir, device_map={"": 0}, **common
    ).eval()


def source_phase(
    config: dict, output: Path, *, skip: int = 0, sequences: int | None = None,
    holdback: int = 0,
) -> None:
    spec = config["source"]
    tokenizer_dir = config.get("tokenizer_local_dir", spec["local_dir"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, fix_mistral_regex=True)
    model = load_model(spec["local_dir"])
    dataset = load_dataset(
        config["dataset"]["id"],
        revision=config["dataset"]["revision"],
        split=config["dataset"]["split"],
        streaming=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    records = []
    desired = sequences or config["sequences"]
    persisted_stride = int(config.get("persisted_token_stride", 1))
    for row_index, row in enumerate(dataset):
        if row_index < skip:
            continue
        ids = tokenizer(
            row["text"], return_tensors="pt", truncation=True,
            max_length=config["tokens"], add_special_tokens=False,
        )["input_ids"]
        if ids.shape[1] != config["tokens"]:
            continue
        if holdback < 0 or holdback >= ids.shape[1]:
            raise ValueError("holdback must be smaller than the token sequence")
        cache_ids = ids[:, :-holdback] if holdback else ids
        cache = capture_cache(model, cache_ids).detach()
        if cache.shape[0] != spec["num_layers"]:
            raise RuntimeError(f"source layer mismatch: {cache.shape}")
        if persisted_stride > 1:
            cache = cache.to_content_space().sample_tokens(persisted_stride)
        path = output / f"{len(records):05}.safetensors"
        sequence_id = f"{config['dataset']['id']}:{config['dataset']['split']}:{row_index}"
        save_source(
            path, cache, ids, sequence_id,
            persisted_token_stride=persisted_stride,
        )
        records.append({"name": path.name, "sha256": sha256(path), "sequence_id": sequence_id})
        print(f"source {len(records)}/{desired}", flush=True)
        if len(records) == desired:
            break
    if len(records) != desired:
        raise RuntimeError("not enough usable source sequences")
    (output / "manifest.json").write_text(json.dumps({
        "model": spec, "tokenizer_hash": tokenizer_fingerprint(tokenizer), "records": records,
        "persisted_token_stride": persisted_stride,
        "persisted_keys_are_content": persisted_stride > 1,
    }, indent=2) + "\n")


def target_phase(config: dict, source_dir: Path, output: Path) -> None:
    spec = config["target"]
    tokenizer_dir = config.get("tokenizer_local_dir", spec["local_dir"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, fix_mistral_regex=True)
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if tokenizer_fingerprint(tokenizer) != source_manifest["tokenizer_hash"]:
        raise RuntimeError("source and target tokenizer fingerprints differ")
    model = load_model(spec["local_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, record in enumerate(source_manifest["records"]):
        source_path = source_dir / record["name"]
        if sha256(source_path) != record["sha256"]:
            raise RuntimeError(f"source hash mismatch: {source_path}")
        source, ids, sequence_id, persisted_stride = load_source(source_path)
        if persisted_stride != source_manifest.get("persisted_token_stride", 1):
            raise RuntimeError(f"source persisted stride mismatch: {source_path}")
        target = capture_cache(model, ids).detach()
        if target.shape[0] != spec["num_layers"]:
            raise RuntimeError(f"target layer mismatch: {target.shape}")
        if persisted_stride > 1:
            target = target.to_content_space().sample_tokens(persisted_stride)
        if source.shape[3] != target.shape[3]:
            raise RuntimeError("source and target persisted token axes do not align")
        path = save_calibration_shard(
            output / f"{index:05}.safetensors",
            CalibrationPair(source, target), sequence_id=sequence_id,
        )
        records.append({"name": path.name, "sha256": sha256(path), "sequence_id": sequence_id})
        print(f"target {len(records)}/{len(source_manifest['records'])}", flush=True)
    (output / "manifest.json").write_text(json.dumps({
        "source": config["source"], "target": spec,
        "tokenizer_hash": source_manifest["tokenizer_hash"], "records": records,
        "persisted_token_stride": source_manifest.get("persisted_token_stride", 1),
        "persisted_keys_are_content": source_manifest.get("persisted_keys_are_content", False),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("source", "target"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--sequences", type=int)
    parser.add_argument("--holdback", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase == "source":
        source_phase(
            config, args.source_dir, skip=args.skip, sequences=args.sequences,
            holdback=args.holdback,
        )
    else:
        if args.output_dir is None:
            parser.error("target phase requires --output-dir")
        target_phase(config, args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
    # torch 2.13 + streaming datasets can abort while background HTTP thread
    # state is finalized even after every artifact has been durably closed.
    # Exit directly only after the successful main path; exceptions still fail.
    os._exit(0)
