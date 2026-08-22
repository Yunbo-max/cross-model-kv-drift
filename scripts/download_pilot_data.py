"""Materialize small, revision-aware datasets used by the local pilot."""

from pathlib import Path

from datasets import Dataset, load_dataset


ROOT = Path(__file__).resolve().parents[1] / "data"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def save_stream_prefix(dataset_id: str, revision: str, count: int, destination: Path) -> None:
    stream = load_dataset(dataset_id, revision=revision, split="train", streaming=True)
    rows = []
    for row in stream:
        if isinstance(row.get("text"), str) and row["text"].strip():
            rows.append(row)
        if len(rows) == count:
            break
    Dataset.from_list(rows).save_to_disk(destination)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    # Covers the smoke calibration (16 rows) and the evaluator's skip=2048 window.
    save_stream_prefix(
        "HuggingFaceFW/fineweb-edu",
        FINEWEB_REVISION,
        2_100,
        ROOT / "fineweb-edu-pilot-2100",
    )
    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1").save_to_disk(
        ROOT / "wikitext-2-raw-v1"
    )
    load_dataset("stanfordnlp/coqa", trust_remote_code=True).save_to_disk(ROOT / "coqa")


if __name__ == "__main__":
    main()
