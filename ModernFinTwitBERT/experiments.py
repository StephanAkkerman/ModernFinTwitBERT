"""
Fine-tuning experiments: hyperparameter grids and training methods, tracked in wandb.

Each experiment in experiments.json fine-tunes the base model once per seed and records
validation and test scores. Pick the best experiment by validation macro F1 (`val_f1`)
and use the test set only to confirm it: choosing by test score would tune on the test
set, and with 1,237 tweets the noise is about as large as the differences we look for.

Experiment options, on top of TrainingArguments (finetune_args):
- "grid": {"learning_rate": [1e-5, 2e-5]} runs every combination as its own experiment
- "llrd": 0.9 gives each lower layer 0.9x the learning rate of the layer above it
- "class_weights": "balanced" weighs the loss by inverse class frequency, which trades
  some precision for bullish/bearish recall
- label_smoothing_factor, warmup_steps, weight_decay, ... are TrainingArguments

After training, a temperature is fitted on the validation set, so `test/ece_calibrated`
shows how reliable the confidence scores can be made. Test probabilities are saved per
run, and the summary also scores the average of each experiment's seeds (an ensemble).

Runs already in the results file are skipped. Run from the repository root:

    wandb login   # once
    python ModernFinTwitBERT/experiments.py
    python ModernFinTwitBERT/experiments.py --only baseline llrd-0.9
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
from benchmark import load_base_model
from eval import score
from model import compute_metrics
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data import LABELS, load_finetuning_data, load_test_data

OUTPUT_DIR = Path("output/experiments")
SUMMARY_COLUMNS = [
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
    netrc = Path.home() / ".netrc"
    return "WANDB_API_KEY" in os.environ or (
        netrc.exists() and "api.wandb.ai" in netrc.read_text()
    )


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


def run(experiment: dict, seed: int, base: str, data: tuple, test, config: dict, out: Path):
    tokenizer = AutoTokenizer.from_pretrained(base, cache_dir="models")
    max_length = config["finetune"].get("max_length", 128)

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train, val = (split.map(encode, batched=True, remove_columns=["text"]) for split in data)
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
    temperature = fit_temperature(val_output.predictions, val_output.label_ids)
    logits = trainer.predict(test_encoded).predictions
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1).numpy()
    calibrated = torch.softmax(torch.tensor(logits / temperature, dtype=torch.float32), -1).numpy()

    test_df = test.to_pandas()
    test_df["label"] = test_df["label"].map(dict(enumerate(LABELS)))
    row = {
        "val/f1": val_output.metrics["test_f1"],
        "temperature": temperature,
        **test_metrics(probs, test_df),
        "test/ece_calibrated": test_metrics(calibrated, test_df)["test/ece"],
    }
    if use_wandb:
        wandb.log({k: v for k, v in row.items()})
        wandb.finish()

    np.save(OUTPUT_DIR / "predictions" / f"{experiment['name']}-seed{seed}.npy", calibrated)
    shutil.rmtree(out, ignore_errors=True)
    return row


def summarize(results: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Mean ± std over seeds per experiment, best validation F1 first, plus the ensemble."""
    metrics = [c for c in SUMMARY_COLUMNS if c in results]
    grouped = results.groupby("name", sort=False)
    mean, std, runs = grouped[metrics].mean(), grouped[metrics].std(), grouped.size()

    rows = []
    for name in mean.sort_values("val/f1", ascending=False).index:
        row = {"experiment": name, "runs": runs[name]}
        for metric in metrics:
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


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning experiments")
    parser.add_argument("--config", default="experiments.json")
    parser.add_argument("--only", nargs="+", help="Only run these experiments, by name")
    parser.add_argument(
        "--rerun", action="store_true", help="Drop earlier results of the selected experiments"
    )
    args = parser.parse_args()

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
    if args.rerun and len(results):
        results = results[~results["name"].isin([e["name"] for e in experiments])]
    done = set(zip(results["name"], results["seed"])) if len(results) else set()

    base = plan.get("base", config["finetune"]["base_model"])
    data = load_finetuning_data(
        drop_sources=config["finetune"].get("drop_sources"),
        relabel=config["finetune"].get("relabel", True),
    )
    test = load_test_data()
    test_df = test.to_pandas()
    test_df["label"] = test_df["label"].map(dict(enumerate(LABELS)))

    for experiment in experiments:
        for seed in plan.get("seeds", [42]):
            if (experiment["name"], seed) in done:
                print(f"Skipping {experiment['name']} seed {seed}: already in {results_path}")
                continue
            print(f"=== {experiment['name']} seed {seed}")
            out = OUTPUT_DIR / "checkpoints" / f"{experiment['name']}-seed{seed}"
            row = {"name": experiment["name"], "seed": seed}
            row.update(run(experiment, seed, base, data, test, config, out))
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            results.to_csv(results_path, index=False)
            print(f"val F1 {row['val/f1']:.3f}  test macro F1 {row['test/macro_f1']:.3f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize(results, test_df)
    (OUTPUT_DIR / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
