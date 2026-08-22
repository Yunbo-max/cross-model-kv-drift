"""Evaluate a mapped cache from pre-captured source-only caches."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from capture_ministral_sequential import load_model, load_source
from kvbridge.huggingface import (
    capture_cache,
    suffix_logits_and_queries_from_cache,
    suffix_logits_from_cache,
)
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.metrics import attention_output_cosine, logit_kl_divergence
from kvbridge.synthetic import cache_r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("mapper_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    target_dir = config["target"]["local_dir"]
    tokenizer_dir = config.get("tokenizer_local_dir", target_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, fix_mistral_regex=True)
    model = load_model(target_dir)
    device = model.get_input_embeddings().weight.device
    mapper = CrossModelKVMapper.load(args.mapper_dir).to(device, dtype=torch.float32)
    cases = []
    for path in sorted(args.source_dir.glob("*.safetensors")):
        source, ids, sequence_id, _ = load_source(path)
        prefix, suffix = ids[:, :-1], ids[:, -1:]
        # Source cache was captured for the full sequence; recapture files for
        # evaluation therefore store the held-back-token prefix explicitly.
        if source.shape[3] != prefix.shape[1]:
            raise RuntimeError("evaluation source cache does not match prefix length")
        reference = capture_cache(model, prefix)
        mapped = mapper.map(source, target_rotary=reference.rotary).to(device)
        reference_logits_from_cache, probe_queries = suffix_logits_and_queries_from_cache(
            model, reference, suffix
        )
        attention = attention_output_cosine(
            probe_queries, mapped, reference, causal=False
        )
        reference_logits = reference_logits_from_cache
        candidate_logits = suffix_logits_from_cache(model, mapped, suffix)
        case = {
            "sequence_id": sequence_id,
            "cache_r2": cache_r2(mapped, reference),
            "attention_cosine_mean": attention.mean,
            "attention_cosine_min": attention.minimum,
            "logit_kl": logit_kl_divergence(candidate_logits, reference_logits),
            "next_token_agreement": bool(torch.equal(
                candidate_logits[:, -1].argmax(-1), reference_logits[:, -1].argmax(-1)
            )),
        }
        if not all(math.isfinite(float(case[k])) for k in (
            "cache_r2", "attention_cosine_mean", "attention_cosine_min", "logit_kl"
        )):
            raise RuntimeError("non-finite evaluation metric")
        cases.append(case)
        print(f"evaluated {len(cases)}: attention={attention.mean:.6f}, KL={case['logit_kl']:.6f}", flush=True)
        del source, reference, mapped, reference_logits, reference_logits_from_cache
        del candidate_logits, probe_queries, attention
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "sequences": len(cases),
        "cache_r2_mean": sum(x["cache_r2"] for x in cases) / len(cases),
        "attention_cosine_mean": sum(x["attention_cosine_mean"] for x in cases) / len(cases),
        "attention_cosine_min": min(x["attention_cosine_min"] for x in cases),
        "logit_kl_mean": sum(x["logit_kl"] for x in cases) / len(cases),
        "next_token_agreement": sum(x["next_token_agreement"] for x in cases) / len(cases),
    }
    args.output.write_text(json.dumps({"summary": summary, "cases": cases}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
