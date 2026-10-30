"""
Hugging Face model card for the domain-pretrained (masked language) model, with the
datasets, training settings and perplexity of the run.

model.py writes it to output/<pretrain model_name>/README.md at the end of pretraining,
together with pretrain_stats.json, so scripts/pretrain_remote.sh uploads it with the model.
To rewrite the card of a model that is already trained, e.g. to add the benchmark results
of the model and its base model from output/benchmark/results.csv once they exist, run
from the repository root:

    python ModernFinTwitBERT/pretrain_card.py
    python ModernFinTwitBERT/pretrain_card.py --model StephanAkkerman/FinTwitRoBERTa-large \\
        --hub-repo StephanAkkerman/FinTwitRoBERTa-large

The perplexity is measured once and kept in pretrain_stats.json; --recompute measures it again.
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from data import DEFAULT_PRETRAIN_DATASETS, load_dev_data, load_test_data

BENCHMARK_RESULTS = Path("output") / "benchmark" / "results.csv"

# What the card says about a base model, beyond its name
BASE_DESCRIPTIONS = {
    "cardiffnlp/twitter-roberta-large-2022-154m": (
        "a RoBERTa-large model (355M parameters) trained on 154M tweets up to the end of "
        "2022 ([TimeLMs](https://arxiv.org/abs/2202.03829))"
    ),
}

WIDGETS = [
    ("Paris is the {mask} of France.", "Generic"),
    ("AAPL is a {mask} sector stock.", "AAPL"),
    ("I predict that this stock will go {mask}.", "Stock Direction"),
    ("$AAPL is the ticker for the company named {mask}.", "Ticker"),
    ("Fed cuts rates, $SPY to all time {mask}.", "Market"),
]


def heldout_tweets() -> list[str]:
    """The test and dev tweets: from the same feed, but kept out of the pretraining data."""
    return list(load_test_data()["text"]) + list(load_dev_data()["text"])


def perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 128,
    mlm_probability: float = 0.15,
    batch_size: int = 64,
) -> float:
    """Masked-token perplexity, with the same random masks for every model (fixed seed)."""
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm_probability=mlm_probability
    )
    encoded = [
        tokenizer(text, truncation=True, max_length=max_length) for text in texts
    ]
    device = next(model.parameters()).device
    model.eval()
    torch.manual_seed(42)
    loss, tokens = 0.0, 0
    for start in range(0, len(encoded), batch_size):
        batch = collator(encoded[start : start + batch_size])
        masked = batch["labels"] != -100
        with torch.no_grad():
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
        loss += torch.nn.functional.cross_entropy(
            logits[masked.to(device)].float(),
            batch["labels"][masked].to(device),
            reduction="sum",
        ).item()
        tokens += masked.sum().item()
    return math.exp(loss / tokens)


def heldout_perplexities(
    model, tokenizer, base_model: str, max_length: int, mlm: float
) -> dict:
    """Perplexity of the model and of its base model on the held-out tweets."""
    texts = heldout_tweets()
    scores = {"tweets": len(texts)}
    scores["perplexity"] = perplexity(model, tokenizer, texts, max_length, mlm)
    base = AutoModelForMaskedLM.from_pretrained(base_model, cache_dir="models")
    base.to(next(model.parameters()).device)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="models")
    scores["base_perplexity"] = perplexity(base, base_tokenizer, texts, max_length, mlm)
    del base
    return scores


def benchmark_f1(names: list[str]) -> float | None:
    """Mean test macro F1 over seeds from benchmark.py, for the first name with results."""
    if not BENCHMARK_RESULTS.exists():
        return None
    results = pd.read_csv(BENCHMARK_RESULTS)
    for name in names:
        rows = results[(results["kind"] == "base") & (results["model"] == name)]
        if len(rows):
            return rows["test/macro_f1"].mean()
    return None


def round_ppl(value: float) -> str:
    return f"{value:.2f}" if value < 10 else f"{value:,.0f}"


def dataset_line(source: dict, tweets: int | None) -> str:
    name = source["name"]
    is_local = Path(name).suffix in (".csv", ".json", ".jsonl", ".parquet")
    line = (
        f"`{Path(name).name}`"
        if is_local
        else (f"[{name}](https://huggingface.co/datasets/{name})")
    )
    if tweets is not None:
        line += f": {tweets:,} tweets after cleaning"
    if source.get("repeat", 1) > 1:
        line += f", shown {source['repeat']} times per epoch to upweight it"
    return f"- {line}"


def model_card(config: dict, stats: dict, repo: str, model_dir: str, mask: str) -> str:
    """The README.md for the pretrained model."""
    pretrain = config["pretrain"]
    args = {**config["base_args"], **pretrain["pretrain_args"]}
    base = pretrain["base_model"]
    name = pretrain["model_name"]
    sources = pretrain.get("datasets") or DEFAULT_PRETRAIN_DATASETS
    counts = stats.get("tweets_per_source") or [None] * len(sources)
    hub_datasets = [
        s["name"]
        for s in sources
        if Path(s["name"]).suffix not in (".csv", ".json", ".jsonl", ".parquet")
    ]
    heldout = stats.get("heldout", {})

    front = [
        "license: mit",
        "language:\n- en",
        "library_name: transformers",
        "pipeline_tag: fill-mask",
        "tags:\n"
        + "\n".join(
            f"- {tag}"
            for tag in [
                "NLP",
                "RoBERTa",
                "FinTwitBERT",
                "FinTwitRoBERTa",
                "finance",
                "financial-analysis",
                "sentiment",
                "financial-sentiment-analysis",
                "twitter",
                "tweets",
                "tweet-analysis",
                "stocks",
                "stock-market",
                "crypto",
                "cryptocurrency",
            ]
        ),
    ]
    if hub_datasets:
        front.append("datasets:\n" + "\n".join(f"- {d}" for d in hub_datasets))
    front.append("metrics:\n- perplexity")
    front.append(
        "widget:\n"
        + "\n".join(
            f"- text: {text.format(mask=mask)}\n  example_title: {title}"
            for text, title in WIDGETS
        )
    )
    front.append(f"base_model: {base}")
    if "perplexity" in heldout:
        front.append(
            f"model-index:\n- name: {name}\n  results:\n  - task:\n"
            "      type: fill-mask\n      name: Masked Language Modeling\n"
            "    dataset:\n      name: Held-out financial tweets\n      type: financial-tweets\n"
            f"    metrics:\n    - type: perplexity\n      value: {heldout['perplexity']:.2f}"
        )

    base_link = f"[{base}](https://huggingface.co/{base})"
    description = BASE_DESCRIPTIONS.get(base)
    base_line = f"{base_link}, {description}" if description else base_link

    sentiment = (
        "## Sentiment Analysis\n"
        "The [FinTwitRoBERTa-sentiment](https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-sentiment) "
        "model fine-tunes this model to classify financial tweets as bullish, bearish or neutral."
    )
    f1, base_f1 = benchmark_f1([repo, model_dir, name]), benchmark_f1([base])
    if f1 is not None and base_f1 is not None:
        sentiment += (
            f"\n\nFine-tuned with the same sentiment recipe, starting from {name} instead of "
            f"{base} changes the macro F1 on our held-out test set of 1,237 financial tweets "
            f"from {base_f1:.3f} to {f1:.3f} (mean over seeds)."
        )

    tweets = stats.get("train_examples")
    cleaned = (
        "Before training, the tweets were cleaned: near-duplicates (such as bot price alerts "
        "that only differ in numbers), one-word tweets and tweets that are mostly hashtags, "
        "cashtags, mentions or links were removed, and quoted tweets became examples of "
        "their own. The tweets of our sentiment test and dev sets were removed from the "
        "pre-training data."
    )
    if tweets:
        cleaned += f" This leaves {tweets:,} training examples (with repeats)."

    batch = args["per_device_train_batch_size"] * args.get(
        "gradient_accumulation_steps", 1
    )
    warmup = args.get("warmup_steps", 0)
    warmup = (
        f"{warmup:.0%} warmup"
        if isinstance(warmup, float) and warmup < 1
        else f"{warmup} warmup steps"
    )
    epochs = args["num_train_epochs"]
    training = (
        f"{epochs:g} epoch{'' if epochs == 1 else 's'} of masked language modeling "
        f"({pretrain.get('mlm_probability', 0.15):.0%} masking), learning rate "
        f"{args['learning_rate']:g} with {warmup}, weight decay {args.get('weight_decay', 0):g}, "
        f"batch size {batch}, maximum length {pretrain.get('max_length', 128)} tokens"
    )
    if stats.get("gpu"):
        training += f", on {stats['gpu']}"
    if stats.get("train_hours"):
        training += f" in {stats['train_hours']:.1f} hours"

    added = pretrain.get("add_tokens") or []
    preprocessing = (
        "links are replaced with `[URL]` and user mentions with `@USER`. "
        + (
            f"{', '.join(f'`{t}`' for t in added)} were added to the tokenizer."
            if added
            else "These are not added as special tokens, so do the same replacement on "
            "your own tweets for the best results."
        )
    )

    details = [
        f"- **Base model:** {base_line}.",
        f"- **Tokenizer:** the tokenizer of the base model"
        + (" with the added tokens below" if added else ", unchanged")
        + f". The mask token is `{mask}`.",
        f"- **Preprocessing:** {preprocessing}",
        f"- **Training:** {training}.",
    ]
    if "perplexity" in heldout:
        details.append(
            f"- **Perplexity:** {round_ppl(heldout['perplexity'])} on {heldout['tweets']:,} held-out "
            "financial tweets (our sentiment test and dev sets, removed from the pre-training "
            f"data), against {round_ppl(heldout['base_perplexity'])} for the base model. These "
            "tweets come from the same feed as part of the pre-training data, so expect a "
            "smaller gap on unrelated tweets; the sentiment results are the better measure."
        )
    if stats.get("val_perplexity"):
        details.append(
            f"- **Validation perplexity:** {round_ppl(stats['val_perplexity'])} on "
            f"{stats['val_tweets']:,} random pre-training tweets held out from training."
        )

    year = stats.get("date", str(date.today()))[:4]
    return f"""---
{chr(10).join(front)}
---

# {name}

{name} is a language model pre-trained on financial tweets. It continues the pre-training of {base_link} to capture the tickers, jargon and communication style of financial Twitter, making it a strong starting point for sentiment analysis and other financial NLP tasks on tweets. It is the successor of [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT).

{sentiment}

## Dataset
{name} is pre-trained on financial tweet datasets, consisting of tweets mentioning stocks and cryptocurrencies:
{chr(10).join(dataset_line(s, c) for s, c in zip(sources, counts))}

{cleaned}

## Model Details
{chr(10).join(details)}

## More Information
For a comprehensive overview, including the complete training setup details and more, visit the [ModernFinTwitBERT GitHub repository](https://github.com/StephanAkkerman/ModernFinTwitBERT).

## Usage
Using [HuggingFace's transformers library](https://huggingface.co/docs/transformers/index) the model and tokenizer can be converted into a pipeline for masked language modeling.

```python
from transformers import pipeline

pipe = pipeline(
    "fill-mask",
    model="{repo}",
)
print(pipe("Fed cuts rates, $SPY to all time {mask}."))
```

## Citing & Authors

If you use {name} in your research, please cite me as follows:

```
@misc{{{name.replace("-", "")},
  author = {{Stephan Akkerman}},
  title = {{{name}}},
  year = {{{year}}},
  publisher = {{GitHub}},
  journal = {{GitHub repository}},
  howpublished = {{\\url{{https://github.com/StephanAkkerman/ModernFinTwitBERT}}}}
}}
```

## License
This project is licensed under the MIT License. See the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.
"""


def hub_repo(config: dict) -> str:
    pretrain = config["pretrain"]
    return pretrain.get("hub_repo") or f"StephanAkkerman/{pretrain['model_name']}"


def write_card(config: dict, model_dir: str, stats: dict, mask: str) -> Path:
    """Save pretrain_stats.json and README.md in the model folder."""
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pretrain_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    readme = folder / "README.md"
    readme.write_text(model_card(config, stats, hub_repo(config), model_dir, mask))
    print(f"Wrote the model card to {readme}")
    return readme


def main():
    parser = argparse.ArgumentParser(description="Write the pretrained model's card")
    parser.add_argument(
        "--model",
        help="Model folder or Hugging Face id, by default output/<pretrain model_name>",
    )
    parser.add_argument(
        "--recompute", action="store_true", help="Measure the held-out perplexity again"
    )
    parser.add_argument(
        "--hub-repo", help="Upload README.md and pretrain_stats.json here"
    )
    args = parser.parse_args()

    config = json.loads(Path("config.json").read_text())
    pretrain = config["pretrain"]
    model = args.model or f"output/{pretrain['model_name']}"
    # The card of a model on the hub goes to output/<its name>
    model_dir = model if Path(model).is_dir() else f"output/{Path(model).name}"
    stats_path = Path(model_dir) / "pretrain_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    tokenizer = AutoTokenizer.from_pretrained(model)
    if args.recompute or "heldout" not in stats:
        model = AutoModelForMaskedLM.from_pretrained(model)
        if torch.cuda.is_available():
            model.to("cuda")
        stats["heldout"] = heldout_perplexities(
            model,
            tokenizer,
            pretrain["base_model"],
            pretrain.get("max_length", 128),
            pretrain.get("mlm_probability", 0.15),
        )
    write_card(config, model_dir, stats, tokenizer.mask_token)

    if args.hub_repo:
        api = HfApi()
        for file in ("README.md", "pretrain_stats.json"):
            api.upload_file(
                path_or_fileobj=str(Path(model_dir) / file),
                path_in_repo=file,
                repo_id=args.hub_repo,
                commit_message="Update model card",
            )
        print(f"Uploaded the card to https://huggingface.co/{args.hub_repo}")


if __name__ == "__main__":
    main()
