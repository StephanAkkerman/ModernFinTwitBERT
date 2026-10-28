"""
Fine-tuning experiments: hyperparameter grids and training methods, tracked in wandb.

Each experiment in experiments.json fine-tunes the base model once per seed and records
validation, dev and test scores. Pick the best experiment by dev macro F1
(`dev/macro_f1`) and use the test set only to confirm it: choosing by test score would
tune on the test set. The dev set (data/test/dev_v0.csv) comes from the same feed as the
test set; the validation split comes from the training sources, which are labeled and
balanced differently, and ranked experiments almost in reverse order of test.

Experiment options, on top of TrainingArguments (finetune_args):
- "grid": {"learning_rate": [1e-5, 2e-5]} runs every combination as its own experiment
- "llrd": 0.9 gives each lower layer 0.9x the learning rate of the layer above it
- "class_weights": "balanced" weighs the loss by inverse class frequency, which trades
  some precision for bullish/bearish recall
- label_smoothing_factor, warmup_steps, weight_decay, ... are TrainingArguments
- "extra_train": [{"path": "data/train/feed_v1.csv", "repeat": 3}] adds labeled CSV files
  to the training split (not validation); without it, finetune "extra_train" in config.json

After training, a temperature is fitted on the dev set, so `test/ece_calibrated`
shows how reliable the confidence scores can be made. Test probabilities are saved per
run, and the summary also scores the average of each experiment's seeds (an ensemble).

Runs already in the results file are skipped. Run from the repository root:

    wandb login   # once
    python ModernFinTwitBERT/experiments.py
    python ModernFinTwitBERT/experiments.py --only baseline llrd-0.9

To keep the final model, train it again with --save: the best checkpoint is saved to
output/experiments/models/<name>-seed<seed> with the dev-fitted temperature in its config
(`calibration_temperature`) and a model card. --hub-repo also uploads it to Hugging Face
(private unless --public; log in first with `hf auth login`):

    python ModernFinTwitBERT/experiments.py --only llrd-0.9 --seeds 42 --save \
        --hub-repo StephanAkkerman/FinTwitRoBERTa-sentiment
"""

import argparse
import itertools
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from benchmark import load_base_model
from eval import score
from model import compute_metrics
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data import LABELS, load_dev_data, load_finetuning_data, load_test_data

OUTPUT_DIR = Path("output/experiments")
SUMMARY_COLUMNS = [
    "dev/macro_f1",
    "val/f1",
    "test/macro_f1",
    "test/precision_bullish",
    "test/precision_bearish",
    "test/recall_bullish",
    "test/recall_bearish",
    "test/ece",
    "test/ece_calibrated",
]
# Short names for grid values in experiment names, e.g. grid-lr2e-05-ep3
SHORT = {"learning_rate": "lr", "num_train_epochs": "ep", "weight_decay": "wd"}


def wandb_enabled() -> bool:
    """Logged in to wandb by `wandb login` or WANDB_API_KEY."""
    import wandb

    # Let wandb resolve credentials: any host, and _netrc on Windows
    return wandb.login(prompt=False)


def expand(experiments: list[dict]) -> list[dict]:
    """One experiment per combination of each experiment's grid values."""
    expanded = []
    for experiment in experiments:
        grid = experiment.get("grid", {})
        for values in itertools.product(*grid.values()):
            setting = dict(zip(grid, values))
            name = "-".join(
                [experiment["name"], *(f"{SHORT.get(k, k)}{v}" for k, v in setting.items())]
            )
            args = {**experiment.get("finetune_args", {}), **setting}
            expanded.append({**experiment, "name": name, "finetune_args": args})
    return expanded


def llrd_optimizer(model, args: TrainingArguments, decay: float):
    """AdamW where layer i of n gets lr * decay ** (n - i); embeddings get the lowest."""
    layers = model.config.num_hidden_layers
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match = re.search(r"\.layers?\.(\d+)\.", name)
        if match:
            depth = int(match.group(1)) + 1
        elif "embeddings" in name:
            depth = 0
        else:
            # Classifier head and pooler
            depth = layers + 1
        no_decay = param.ndim < 2 or "norm" in name.lower()
        key = (depth, no_decay)
        groups.setdefault(key, []).append(param)

    param_groups = [
        {
            "params": params,
            "lr": args.learning_rate * decay ** (layers + 1 - depth),
            "weight_decay": 0.0 if no_decay else args.weight_decay,
        }
        for (depth, no_decay), params in groups.items()
    ]
    return torch.optim.AdamW(param_groups, lr=args.learning_rate, fused=torch.cuda.is_available())


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights (and the label smoothing from the arguments)."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits.float(),
            labels,
            weight=self.class_weights.to(outputs.logits.device),
            label_smoothing=self.args.label_smoothing_factor,
        )
        return (loss, outputs) if return_outputs else loss


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """The temperature that minimizes the validation negative log-likelihood."""
    logits, labels = torch.tensor(logits, dtype=torch.float32), torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp())


def test_metrics(probs: np.ndarray, test: pd.DataFrame, prefix: str = "test") -> dict:
    df = test.assign(
        pred=[LABELS[i] for i in probs.argmax(1)], confidence=probs.max(1)
    )
    output = score(df)
    recall = (df["pred"] == df["label"]).groupby(df["label"]).mean()
    output["test/recall_bullish"] = recall.get("bullish", 0.0)
    output["test/recall_bearish"] = recall.get("bearish", 0.0)
    return {k.replace("test/", f"{prefix}/"): float(v) for k, v in output.items()}


def run(
    experiment: dict,
    seed: int,
    base: str,
    data: tuple,
    dev,
    test,
    config: dict,
    out: Path,
    save_dir: Path | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(base, cache_dir="models")
    max_length = config["finetune"].get("max_length", 128)

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train, val = (split.map(encode, batched=True, remove_columns=["text"]) for split in data)
    dev_encoded = dev.map(encode, batched=True, remove_columns=["text", "file"])
    test_encoded = test.map(encode, batched=True, remove_columns=["text", "file"])

    use_wandb = wandb_enabled()
    args = TrainingArguments(
        **{
            **config["base_args"],
            **config["finetune"]["finetune_args"],
            **experiment.get("finetune_args", {}),
            "report_to": "wandb" if use_wandb else "none",
            "run_name": f"{experiment['name']}-seed{seed}",
            "seed": seed,
            "output_dir": str(out),
            # Keep only the best checkpoint on disk
            "save_total_limit": 1,
        }
    )
    if use_wandb:
        import wandb

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "FinTwitRoBERTa-finetune"),
            group=experiment["name"],
            name=args.run_name,
            config={"base": base, "seed": seed, **experiment},
            reinit="finish_previous",
        )

    model = load_base_model(base)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    if "llrd" in experiment:
        trainer_kwargs["optimizers"] = (llrd_optimizer(model, args, experiment["llrd"]), None)
    if experiment.get("class_weights") == "balanced":
        counts = np.bincount(train["label"], minlength=len(LABELS))
        weights = torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32)
        trainer = WeightedTrainer(**trainer_kwargs, class_weights=weights)
    else:
        trainer = Trainer(**trainer_kwargs)
    # Our models compute a plain mean loss; let Trainer average it over accumulation
    trainer.model_accepts_loss_kwargs = False
    trainer.train()

    # The best checkpoint is loaded at the end of training
    val_output = trainer.predict(val)
    dev_output = trainer.predict(dev_encoded)
    temperature = fit_temperature(dev_output.predictions, dev_output.label_ids)
    logits = trainer.predict(test_encoded).predictions
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1).numpy()
    calibrated = torch.softmax(torch.tensor(logits / temperature, dtype=torch.float32), -1).numpy()
    dev_probs = torch.softmax(torch.tensor(dev_output.predictions, dtype=torch.float32), -1)

    row = {
        **test_metrics(dev_probs.numpy(), labeled(dev), prefix="dev"),
        "val/f1": val_output.metrics["test_f1"],
        "temperature": temperature,
        **test_metrics(probs, labeled(test)),
        "test/ece_calibrated": test_metrics(calibrated, labeled(test))["test/ece"],
    }
    if use_wandb:
        wandb.log({k: v for k, v in row.items()})
        wandb.finish()

    np.save(OUTPUT_DIR / "predictions" / f"{experiment['name']}-seed{seed}.npy", calibrated)
    if save_dir:
        # Not `temperature`: transformers reserves that name for generation
        trainer.model.config.calibration_temperature = temperature
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(save_dir)
        (save_dir / "README.md").write_text(model_card(experiment, seed, base, row))
        print(f"Saved model to {save_dir}")
    shutil.rmtree(out, ignore_errors=True)
    return row


def model_card(experiment: dict, seed: int, base: str, row: dict) -> str:
    """A Hugging Face README with the recipe, scores and how to use the temperature."""
    scores = "\n".join(
        f"| {metric} | {row[metric]:.3f} |" for metric in SUMMARY_COLUMNS if metric in row
    )
    recipe = json.dumps({**experiment, "seed": seed}, indent=2)
    return f"""---
base_model: {base}
library_name: transformers
pipeline_tag: text-classification
language: en
tags:
- finance
- twitter
- sentiment
---

# FinTwitRoBERTa sentiment

Classifies financial tweets as NEUTRAL, BULLISH or BEARISH. Fine-tuned from
[{base}](https://huggingface.co/{base}) with ModernFinTwitBERT's `experiments.py`.

## Usage

Divide the logits by `config.calibration_temperature` (fitted on the dev set) before the
softmax to get calibrated confidence scores:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "<this repo>"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

inputs = tokenizer(["$NVDA breaking out, loading calls"], return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
probs = (logits / model.config.calibration_temperature).softmax(-1)
print({{model.config.id2label[i]: p for i, p in enumerate(probs[0].tolist())}})
```

## Scores

dev = 749 tweets from a real feed (2024+), used to choose the recipe and fit the temperature;
test = 1,237 held-out tweets from the same feed. `test/ece_calibrated` uses the temperature.

| metric | value |
| --- | --- |
{scores}
| temperature | {row["temperature"]:.3f} |

## Recipe

```json
{recipe}
```
"""


def labeled(dataset) -> pd.DataFrame:
    """A dev or test set as a DataFrame with label names, as score() expects."""
    df = dataset.to_pandas()
    df["label"] = df["label"].map(dict(enumerate(LABELS)))
    return df


def summarize(results: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Mean ± std over seeds per experiment, best dev F1 first, plus the ensemble."""
    metrics = [c for c in SUMMARY_COLUMNS if c in results]
    grouped = results.groupby("name", sort=False)
    mean, std, runs = grouped[metrics].mean(), grouped[metrics].std(), grouped.size()

    rows = []
    # Runs from before the dev set existed have no dev score and sort last
    order = "dev/macro_f1" if "dev/macro_f1" in mean else "val/f1"
    for name in mean.sort_values(order, ascending=False).index:
        row = {"experiment": name, "runs": runs[name]}
        for metric in metrics:
            if pd.isna(mean.loc[name, metric]):
                row[metric] = ""
                continue
            value = f"{mean.loc[name, metric]:.3f}"
            if runs[name] > 1:
                value += f" ± {std.loc[name, metric]:.3f}"
            row[metric] = value
        files = sorted((OUTPUT_DIR / "predictions").glob(f"{name}-seed*.npy"))
        if len(files) > 1:
            ensemble = np.mean([np.load(f) for f in files], axis=0)
            row["ensemble macro_f1"] = f"{test_metrics(ensemble, test_df)['test/macro_f1']:.3f}"
        else:
            row["ensemble macro_f1"] = ""
        rows.append(row)
    lines = [" | ".join(rows[0]), " | ".join("---" for _ in rows[0])]
    lines += [" | ".join(str(value) for value in row.values()) for row in rows]
    return "\n".join(f"| {line} |" for line in lines)


def upload(folder: Path, repo: str, private: bool):
    """Upload a saved model; the repo is created if needed (an existing one keeps its visibility)."""
    readme = folder / "README.md"
    readme.write_text(readme.read_text().replace("<this repo>", repo))
    api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, commit_message="Upload model")
    print(f"Uploaded {folder} to https://huggingface.co/{repo}")


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning experiments")
    parser.add_argument("--config", default="experiments.json")
    parser.add_argument("--only", nargs="+", help="Only run these experiments, by name")
    parser.add_argument(
        "--rerun", action="store_true", help="Drop earlier results of the selected experiments"
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the seeds in the config")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Keep the trained models in output/experiments/models (retrains runs already done)",
    )
    parser.add_argument("--hub-repo", help="Upload the saved model to this Hugging Face repo")
    parser.add_argument(
        "--public", action="store_true", help="Make a new --hub-repo public (default private)"
    )
    args = parser.parse_args()
    if args.hub_repo:
        args.save = True

    with open(args.config) as file:
        plan = json.load(file)
    with open("config.json") as file:
        config = json.load(file)

    if not wandb_enabled():
        print("Not logged in to wandb (run `wandb login`), so results are only saved locally")

    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    experiments = expand(plan["experiments"])
    if args.only:
        experiments = [e for e in experiments if any(e["name"].startswith(o) for o in args.only)]
    seeds = args.seeds or plan.get("seeds", [42])
    if args.hub_repo and len(experiments) * len(seeds) != 1:
        parser.error(
            "--hub-repo uploads a single model: select one experiment with --only "
            f"and one seed with --seeds (selected: {[e['name'] for e in experiments]}, {seeds})"
        )
    if args.rerun and len(results):
        results = results[~results["name"].isin([e["name"] for e in experiments])]
    if args.save and len(results):
        # Retrain the selected runs to get their models; their new scores replace the old
        selected = {(e["name"], seed) for e in experiments for seed in seeds}
        results = results[[run not in selected for run in zip(results["name"], results["seed"])]]
    done = set(zip(results["name"], results["seed"])) if len(results) else set()

    base = plan.get("base", config["finetune"]["base_model"])
    # Experiments with the same extra training files share one copy of the data
    datasets = {}

    def finetuning_data(experiment: dict) -> tuple:
        extra = experiment.get("extra_train", config["finetune"].get("extra_train"))
        key = json.dumps(extra)
        if key not in datasets:
            datasets[key] = load_finetuning_data(
                drop_sources=config["finetune"].get("drop_sources"),
                relabel=config["finetune"].get("relabel", True),
                extra_train=extra,
            )
        return datasets[key]

    dev = load_dev_data()
    test = load_test_data()

    for experiment in experiments:
        for seed in seeds:
            if (experiment["name"], seed) in done:
                print(f"Skipping {experiment['name']} seed {seed}: already in {results_path}")
                continue
            print(f"=== {experiment['name']} seed {seed}")
            out = OUTPUT_DIR / "checkpoints" / f"{experiment['name']}-seed{seed}"
            row = {"name": experiment["name"], "seed": seed}
            save_dir = (
                OUTPUT_DIR / "models" / f"{experiment['name']}-seed{seed}" if args.save else None
            )
            row.update(
                run(experiment, seed, base, finetuning_data(experiment), dev, test, config, out, save_dir)
            )
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            results.to_csv(results_path, index=False)
            print(
                f"dev macro F1 {row['dev/macro_f1']:.3f}  test macro F1 {row['test/macro_f1']:.3f}"
            )
            if args.hub_repo:
                upload(save_dir, args.hub_repo, private=not args.public)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize(results, labeled(test))
    (OUTPUT_DIR / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
