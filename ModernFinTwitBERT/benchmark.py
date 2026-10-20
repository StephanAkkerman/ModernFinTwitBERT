"""
Benchmark models on the frozen test set.

Two kinds of entries in benchmark.json:
- "base": a pretrained model (HF id or local path) that is fine-tuned on our sentiment
  data with the recipe in config.json, once per seed, then scored. Use this to compare
  starting points, e.g. ModernBERT-base vs a finance-pretrained ModernBERT.
- "classifier": an existing sentiment model that is scored as-is, e.g. our released
  model or other people's. "label_map" maps its labels onto neutral/bullish/bearish,
  and "tokenizer" names one for models uploaded without tokenizer files.

Runs already in the results file are skipped, so adding a model to benchmark.json and
rerunning only runs the new one. Run from the repository root:

    python ModernFinTwitBERT/benchmark.py
    python ModernFinTwitBERT/benchmark.py --only fin-modernbert finbert
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from data import LABELS, load_finetuning_data
from eval import Evaluate, score
from model import DEVICE_ARGS, compute_metrics
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

OUTPUT_DIR = Path("output/benchmark")
SUMMARY_COLUMNS = [
    "test/macro_f1",
    "test/macro_f1_low",
    "test/macro_f1_high",
    "test/precision_bullish",
    "test/precision_bearish",
    "test/recall_neutral",
    "test/ece",
]


def load_base_model(base: str):
    labels = [label.upper() for label in LABELS]
    kwargs = dict(
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        dtype="auto",
        cache_dir="models",
        # Allows starting from a model that already has a classifier head
        ignore_mismatched_sizes=True,
    )
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            base, **kwargs, **DEVICE_ARGS
        )
    except (ValueError, ImportError):
        # Architectures without flash-attn support (e.g. BERT) use the default attention
        model = AutoModelForSequenceClassification.from_pretrained(
            base, **kwargs, device_map="cuda" if torch.cuda.is_available() else None
        )
    model.config.problem_type = "single_label_classification"
    return model


def finetune(base: str, seed: int, out: Path, data: tuple, config: dict, overrides: dict):
    """Fine-tune `base` with the config.json recipe and save the best checkpoint to `out`."""
    tokenizer = AutoTokenizer.from_pretrained(base, cache_dir="models")
    max_length = config["finetune"].get("max_length", 128)

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train, val = (split.map(encode, batched=True, remove_columns=["text"]) for split in data)

    args = {
        **config["base_args"],
        **config["finetune"]["finetune_args"],
        "report_to": "none",
        **overrides,
        "seed": seed,
        "output_dir": str(out),
    }
    trainer = Trainer(
        model=load_base_model(base),
        args=TrainingArguments(**args),
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    # The best model is saved above; the checkpoints only take up disk space
    for checkpoint in out.glob("checkpoint-*"):
        shutil.rmtree(checkpoint)


def test_scores(model_path: str, label_map: dict | None, tokenizer: str | None) -> dict:
    evaluator = Evaluate(model_path, label_map, tokenizer)
    df = evaluator.predict()
    unknown = set(df["pred"]) - set(LABELS)
    if unknown:
        raise ValueError(
            f"{model_path} predicts labels {unknown}; add a label_map for it in benchmark.json"
        )
    output = score(df)
    del evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def summarize(results: pd.DataFrame) -> str:
    """Mean (and std over seeds) per model, best macro F1 first, as a Markdown table."""
    metrics = [c for c in SUMMARY_COLUMNS if c in results]
    grouped = results.groupby(["name", "kind", "model"], sort=False)
    mean, std = grouped[metrics].mean(), grouped[metrics].std()
    runs = grouped.size()

    rows = []
    for key in mean.sort_values("test/macro_f1", ascending=False).index:
        name, kind, model = key
        row = {"model": name, "kind": kind, "runs": runs[key]}
        for metric in metrics:
            value = f"{mean.loc[key, metric]:.3f}"
            if runs[key] > 1 and not metric.startswith("test/macro_f1_"):
                value += f" ± {std.loc[key, metric]:.3f}"
            row[metric.removeprefix("test/")] = value
        rows.append(row)
    lines = [" | ".join(rows[0]), " | ".join("---" for _ in rows[0])]
    lines += [" | ".join(str(value) for value in row.values()) for row in rows]
    return "\n".join(f"| {line} |" for line in lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark models on the frozen test set")
    parser.add_argument("--config", default="benchmark.json")
    parser.add_argument("--only", nargs="+", help="Only run these entries, by name")
    args = parser.parse_args()

    with open(args.config) as file:
        benchmark = json.load(file)
    with open("config.json") as file:
        config = json.load(file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    done = set(zip(results["name"], results["seed"])) if len(results) else set()

    seeds = benchmark.get("seeds", [42])
    data = None
    for entry in benchmark["models"]:
        name = entry["name"]
        if args.only and name not in args.only:
            continue

        if "classifier" in entry:
            kind, model = "classifier", entry["classifier"]
            runs = [(-1, model)]
        else:
            kind, model = "base", entry["base"]
            runs = [(seed, OUTPUT_DIR / name / f"seed{seed}") for seed in seeds]

        for seed, model_path in runs:
            if (name, seed) in done:
                print(f"Skipping {name} seed {seed}: already in {results_path}")
                continue
            print(f"=== {name} ({kind}: {model}) seed {seed}")

            if kind == "base":
                if data is None:
                    data = load_finetuning_data(
                        drop_sources=config["finetune"].get("drop_sources"),
                        relabel=config["finetune"].get("relabel", True),
                    )
                overrides = {
                    **benchmark.get("finetune_args", {}),
                    **entry.get("finetune_args", {}),
                }
                finetune(model, seed, model_path, data, config, overrides)

            row = {"name": name, "kind": kind, "model": model, "seed": seed}
            row.update(
                test_scores(str(model_path), entry.get("label_map"), entry.get("tokenizer"))
            )
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            results.to_csv(results_path, index=False)
            print(f"macro F1 {row['test/macro_f1']:.3f}")

    summary = summarize(results)
    (OUTPUT_DIR / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
