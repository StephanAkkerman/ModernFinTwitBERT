"""
Confidence thresholds for the bot: call a tweet bullish or bearish only when the model's
calibrated confidence in that class reaches its threshold, otherwise neutral. Raising a
threshold trades directional recall for precision, without retraining.

The thresholds are tuned on the dev set (same feed as the bot reads) and confirmed on the
test set. For each class the script picks the lowest threshold whose dev precision
reaches --target-precision, so it keeps as much recall as that precision allows. Run it
again after every retrain, since a new model has different confidence scores.

Run from the repository root:

    python ModernFinTwitBERT/thresholds.py
    python ModernFinTwitBERT/thresholds.py --model output/experiments/models/llrd-0.9-seed42
    python ModernFinTwitBERT/thresholds.py --target-precision 0.9 --save

--save writes the thresholds to the model's config as `decision_thresholds`
({"bullish": t, "bearish": t}) and, for a local model with a README.md (from
experiments.py --save), adds them to its model card; --hub-repo also uploads the config
and card to Hugging Face.
The tradeoff tables and probabilities go to output/thresholds/<model>/; the probabilities
are reused on the next run, so pass --recompute after retraining a local model.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import LABELS, load_dev_data, load_test_data

DIRECTIONAL = ["bullish", "bearish"]
GRID = np.round(np.arange(0.34, 1.0, 0.01), 2)


def predict_probs(
    model, tokenizer, texts: list[str], batch_size: int = 32
) -> np.ndarray:
    """Calibrated class probabilities (logits / config.calibration_temperature)."""
    temperature = getattr(model.config, "calibration_temperature", 1.0) or 1.0
    probs = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs.append((logits / temperature).softmax(-1).numpy())
    return np.concatenate(probs)


def apply_thresholds(probs: np.ndarray, thresholds: dict) -> np.ndarray:
    """Argmax prediction, demoted to neutral when a directional call is below its threshold."""
    pred = probs.argmax(1)
    for label in DIRECTIONAL:
        i = LABELS.index(label)
        pred[(pred == i) & (probs[:, i] < thresholds.get(label, 0.0))] = LABELS.index(
            "neutral"
        )
    return pred


def metrics(labels: np.ndarray, pred: np.ndarray) -> dict:
    output = {"macro_f1": f1_score(labels, pred, labels=[0, 1, 2], average="macro")}
    for label in DIRECTIONAL:
        i = LABELS.index(label)
        called, actual = pred == i, labels == i
        hits = (called & actual).sum()
        output[f"{label}_calls"] = int(called.sum())
        output[f"{label}_precision"] = hits / called.sum() if called.any() else np.nan
        output[f"{label}_recall"] = hits / actual.sum()
    return output


def sweep(probs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Precision, recall and calls per class for each threshold on that class alone."""
    rows = []
    for t in GRID:
        pred = apply_thresholds(probs, {label: t for label in DIRECTIONAL})
        rows.append({"threshold": t, **metrics(labels, pred)})
    return pd.DataFrame(rows)


def pick(table: pd.DataFrame, label: str, target: float) -> float:
    """Lowest threshold whose precision reaches the target from there on up."""
    precision = table[f"{label}_precision"].fillna(1.0).to_numpy()
    # Precision is noisy with few calls, so require it to hold for all higher thresholds
    holds = np.minimum.accumulate(precision[::-1])[::-1] >= target
    return float(table["threshold"][holds].iloc[0]) if holds.any() else float(GRID[-1])


def update_card(card: str, thresholds: dict, report: pd.DataFrame, target: float) -> str:
    """The model card with a section on the thresholds, replacing an earlier one."""
    rows = [list(report.columns)] + report.astype(str).values.tolist()
    table = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    table = table.replace("\n", "\n| " + " | ".join("---" for _ in rows[0]) + " |\n", 1)
    section = f"""## Decision thresholds

`config.decision_thresholds` = `{json.dumps(thresholds)}`, tuned on the dev set for
precision >= {target}. Call a tweet bullish or bearish only when its calibrated
probability for that class reaches the threshold, otherwise neutral:

```python
label = model.config.id2label[int(probs[0].argmax())].lower()
if label != "neutral" and probs[0].max() < model.config.decision_thresholds[label]:
    label = "neutral"
```

{table}
"""
    card = re.sub(r"## Decision thresholds\n.*?(?=\n## |\Z)", "", card, flags=re.DOTALL)
    return card.rstrip() + "\n\n" + section


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="StephanAkkerman/FinTwitRoBERTa-sentiment")
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--save", action="store_true", help="store in the model config")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="ignore cached probabilities (retrained model)",
    )
    parser.add_argument("--hub-repo", help="also upload the config to this HF repo")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    out = Path("output/thresholds") / args.model.replace("/", "--")
    out.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name, dataset in (("dev", load_dev_data()), ("test", load_test_data())):
        df = dataset.to_pandas()
        cache = out / f"probs_{name}.npy"
        if cache.exists() and not args.recompute:
            probs = np.load(cache)
        else:
            probs = predict_probs(model, tokenizer, df["text"].tolist())
            np.save(cache, probs)
        splits[name] = (probs, df["label"].to_numpy())
        sweep(probs, splits[name][1]).to_csv(out / f"sweep_{name}.csv", index=False)

    dev_table = sweep(*splits["dev"])
    thresholds = {
        label: pick(dev_table, label, args.target_precision) for label in DIRECTIONAL
    }
    print(f"Thresholds for dev precision >= {args.target_precision}: {thresholds}\n")

    report = []
    for name, (probs, labels) in splits.items():
        for setting, t in (("argmax", {}), ("thresholds", thresholds)):
            report.append(
                {
                    "split": name,
                    "setting": setting,
                    **metrics(labels, apply_thresholds(probs, t)),
                }
            )
    report = pd.DataFrame(report).round(3)
    report.to_csv(out / "report.csv", index=False)
    print(report.to_string(index=False))

    if args.save or args.hub_repo:
        model.config.decision_thresholds = thresholds
        readme = Path(args.model) / "README.md"
        if Path(args.model).is_dir():
            model.config.save_pretrained(args.model)
            print(f"\nSaved decision_thresholds to {args.model}/config.json")
            if readme.exists():
                text = update_card(readme.read_text(), thresholds, report, args.target_precision)
                if args.hub_repo:
                    text = text.replace("<this repo>", args.hub_repo)
                readme.write_text(text)
                print(f"Added the thresholds to {readme}")
        if args.hub_repo:
            message = f"Add decision_thresholds (dev precision >= {args.target_precision})"
            model.config.push_to_hub(args.hub_repo, commit_message=message)
            if readme.exists():
                HfApi().upload_file(
                    path_or_fileobj=str(readme),
                    path_in_repo="README.md",
                    repo_id=args.hub_repo,
                    commit_message=message,
                )
            print(f"Uploaded config with decision_thresholds to {args.hub_repo}")
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
