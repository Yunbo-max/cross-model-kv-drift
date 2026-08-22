"""Validate one tokenizer's IDs against both model embedding ranges."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from kvbridge.huggingface import tokenizer_fingerprint
from kvbridge.io import atomic_write_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--texts", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source = AutoConfig.from_pretrained(args.source_config)
    target = AutoConfig.from_pretrained(args.target_config)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        split="train",
        streaming=True,
    )
    maximum_id = -1
    token_count = 0
    texts = 0
    for row in dataset:
        ids = tokenizer(
            row["text"], truncation=True, max_length=args.max_length,
            add_special_tokens=False,
        )["input_ids"]
        if not ids:
            continue
        maximum_id = max(maximum_id, max(ids))
        token_count += len(ids)
        texts += 1
        if texts == args.texts:
            break
    if texts != args.texts:
        raise RuntimeError(f"only found {texts} non-empty texts")
    source_ok = maximum_id < source.vocab_size
    target_ok = maximum_id < target.vocab_size
    report = {
        "schema_version": 1,
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "texts": texts,
        "tokens": token_count,
        "maximum_token_id": maximum_id,
        "source_vocab_size": source.vocab_size,
        "target_vocab_size": target.vocab_size,
        "source_ids_in_range": source_ok,
        "target_ids_in_range": target_ok,
        "shared_ids_by_construction": True,
        "shared_lengths_by_construction": True,
    }
    if not source_ok or not target_ok:
        raise RuntimeError(json.dumps(report))
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
