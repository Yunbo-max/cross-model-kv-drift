"""Sequential, lm-eval-compatible HellaSwag handoff evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from capture_ministral_sequential import load_model, load_source, save_source, sha256
from kvbridge.huggingface import capture_cache, suffix_logits_from_cache
from kvbridge.mapper import CrossModelKVMapper


DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"


def preprocess(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation, context = context[-spaces:] + continuation, context[:-spaces]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    whole_ids = tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
    return context_ids, whole_ids[len(context_ids):]


def capture_phase(config: dict, output: Path, limit: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_local_dir"])
    model = load_model(config["source"]["local_dir"])
    dataset = load_dataset(
        "Rowan/hellaswag", revision=DATASET_REVISION, split="validation"
    )
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(dataset.select(range(limit))):
        context = preprocess(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize())
        choices = [preprocess(item) for item in row["endings"]]
        encoded = [encode_pair(tokenizer, context, " " + choice) for choice in choices]
        context_ids = encoded[0][0]
        if any(pair[0] != context_ids for pair in encoded):
            raise RuntimeError("choice-dependent context tokenization")
        if len(context_ids) < 2 or any(not pair[1] for pair in encoded):
            raise RuntimeError("empty HellaSwag context or continuation")
        ids = torch.tensor([context_ids], dtype=torch.long)
        cache = capture_cache(model, ids[:, :-1]).detach()
        path = output / f"{index:05}.safetensors"
        sequence_id = f"Rowan/hellaswag:validation:{index}"
        save_source(path, cache, ids, sequence_id, persisted_token_stride=1)
        records.append({
            "name": path.name,
            "sha256": sha256(path),
            "sequence_id": sequence_id,
            "continuations": [pair[1] for pair in encoded],
            "choice_char_lengths": [len(choice) for choice in choices],
            "gold": int(row["label"]),
        })
        print(f"captured {index + 1}/{limit}", flush=True)
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "dataset": "Rowan/hellaswag",
        "dataset_revision": DATASET_REVISION,
        "split": "validation",
        "limit": limit,
        "tokenizer_local_dir": config["tokenizer_local_dir"],
        "records": records,
    }, indent=2) + "\n")


def continuation_ll(model, cache, context_last: torch.Tensor, continuation: list[int]) -> float:
    device = model.get_input_embeddings().weight.device
    continuation_tensor = torch.tensor([continuation], dtype=torch.long, device=device)
    suffix = torch.cat((context_last.to(device), continuation_tensor), dim=1)
    logits = suffix_logits_from_cache(model, cache, suffix)[:, :len(continuation)]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return float(log_probs.gather(-1, continuation_tensor.unsqueeze(-1)).sum().item())


def evaluate_phase(config: dict, source_dir: Path, mapper_dir: Path, output: Path) -> None:
    manifest = json.loads((source_dir / "manifest.json").read_text())
    model = load_model(config["target"]["local_dir"])
    device = model.get_input_embeddings().weight.device
    mapper = CrossModelKVMapper.load(mapper_dir).to(device, dtype=torch.float32)
    cases = []
    for index, record in enumerate(manifest["records"]):
        path = source_dir / record["name"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"source checksum mismatch: {path}")
        source, context_ids, sequence_id, stride = load_source(path)
        if stride != 1 or sequence_id != record["sequence_id"]:
            raise RuntimeError("source cache metadata mismatch")
        prefix, context_last = context_ids[:, :-1], context_ids[:, -1:]
        native = capture_cache(model, prefix)
        mapped = mapper.map(source, target_rotary=native.rotary).to(device)
        native_ll = [
            continuation_ll(model, native, context_last, choice)
            for choice in record["continuations"]
        ]
        transfer_ll = [
            continuation_ll(model, mapped, context_last, choice)
            for choice in record["continuations"]
        ]
        lengths = record["choice_char_lengths"]
        native_pred = max(range(4), key=native_ll.__getitem__)
        transfer_pred = max(range(4), key=transfer_ll.__getitem__)
        native_norm = max(range(4), key=lambda i: native_ll[i] / lengths[i])
        transfer_norm = max(range(4), key=lambda i: transfer_ll[i] / lengths[i])
        cases.append({
            "sequence_id": sequence_id,
            "gold": record["gold"],
            "native_ll": native_ll,
            "transfer_ll": transfer_ll,
            "native_correct": native_pred == record["gold"],
            "transfer_correct": transfer_pred == record["gold"],
            "native_norm_correct": native_norm == record["gold"],
            "transfer_norm_correct": transfer_norm == record["gold"],
        })
        print(f"evaluated {index + 1}/{len(manifest['records'])}", flush=True)
    n = len(cases)
    summary = {
        "examples": n,
        "native_acc": sum(x["native_correct"] for x in cases) / n,
        "transfer_acc": sum(x["transfer_correct"] for x in cases) / n,
        "native_acc_norm": sum(x["native_norm_correct"] for x in cases) / n,
        "transfer_acc_norm": sum(x["transfer_norm_correct"] for x in cases) / n,
    }
    summary["retention"] = summary["transfer_acc"] / summary["native_acc"]
    summary["retention_norm"] = summary["transfer_acc_norm"] / summary["native_acc_norm"]
    output.write_text(json.dumps({"summary": summary, "cases": cases}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("capture", "evaluate"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--mapper-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase == "capture":
        capture_phase(config, args.source_dir, args.limit)
    else:
        if args.mapper_dir is None or args.output is None:
            parser.error("evaluate requires --mapper-dir and --output")
        evaluate_phase(config, args.source_dir, args.mapper_dir, args.output)


if __name__ == "__main__":
    main()
    os._exit(0)
