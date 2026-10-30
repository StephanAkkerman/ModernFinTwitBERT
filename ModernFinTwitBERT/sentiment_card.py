"""
Hugging Face model card for the fine-tuned sentiment model, with the training data,
recipe and dev/test scores of the run.

experiments.py --save writes it to the saved model's README.md. thresholds.py --save and
export_onnx.py add their own sections ("## Decision thresholds", "## ONNX int8") before
"## Citing & Authors". To rewrite the card of a model that is already trained, e.g. after
changing this template, run from the repository root:

    python ModernFinTwitBERT/sentiment_card.py --model StephanAkkerman/FinTwitRoBERTa-sentiment \\
        --experiment llrd-0.9-feed --seed 42
    python ModernFinTwitBERT/sentiment_card.py --model StephanAkkerman/FinTwitRoBERTa-sentiment \\
        --experiment llrd-0.9-feed --seed 42 --hub-repo StephanAkkerman/FinTwitRoBERTa-sentiment

This scores the model on the dev and test sets again (a few minutes on CPU), counts the
training data of the experiment and keeps the threshold and ONNX sections of the current
card. The card goes to output/<model name>/README.md; --hub-repo uploads only README.md.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from data import REPO_ROOT

# The previous model, on the same test set (benchmark.py)
FINTWITBERT_SENTIMENT_F1 = 0.565

# The card names the model after this repo unless experiments.py uploads to another one
DEFAULT_REPO = "StephanAkkerman/FinTwitRoBERTa-sentiment"

# What the card says about a base model, beyond its name
BASE_DESCRIPTIONS = {
    "StephanAkkerman/FinTwitRoBERTa-large": (
        "a RoBERTa-large model (355M parameters) pre-trained on millions of financial tweets"
    ),
}

# Sections that other scripts write into the card; kept when the card is rewritten
ADDED_SECTIONS = ["Decision thresholds", "ONNX int8"]

WIDGETS = [
    ("Nice 9% pre market move for $para, pump my calls Uncle Buffett 🤑", "Bullish Crypto Tweet"),
    ("It is about damn time that my $ARB and $ETH bags pump FFS. 🚀", "Bullish Crypto Tweet 2"),
    (
        "$TCBP Lowest float stock in the market. Float just 325k. Don’t sell for pennies, "
        "this one will be a monster. Still early",
        "Bullish Stock Tweet",
    ),
    ("$TSLA breaking down below the 200 day, loaded puts for next week", "Bearish Stock Tweet"),
    ("Bitcoin dumping again, this market is cooked. Sold all my alts.", "Bearish Crypto Tweet"),
    ("$AAPL reports earnings on Thursday after the close", "Neutral Stock Tweet"),
    ("Italian companies braced for more political uncertainty", "Neutral News"),
]

TAGS = [
    "NLP",
    "RoBERTa",
    "FinTwitBERT",
    "FinTwitRoBERTa",
    "sentiment",
    "finance",
    "financial-analysis",
    "sentiment-analysis",
    "financial-sentiment-analysis",
    "twitter",
    "tweets",
    "tweet-analysis",
    "stocks",
    "stock-market",
    "crypto",
    "cryptocurrency",
]


def training_args(config: dict, experiment: dict) -> dict:
    """The fine-tuning arguments of an experiment, as experiments.py merges them."""
    return {
        **config["base_args"],
        **config["finetune"]["finetune_args"],
        **experiment.get("finetune_args", {}),
        "max_length": config["finetune"].get("max_length", 128),
    }


def data_stats(train, val, extra_train: list[dict] | None) -> dict:
    """Tweet counts of a fine-tuning run, split into the labeled sources and the extra files."""
    extra = []
    for source in extra_train or []:
        tweets = len(pd.read_csv(REPO_ROOT / source["path"]))
        extra.append({"path": source["path"], "tweets": tweets, "repeat": source.get("repeat", 1)})
    in_extra = sum(e["tweets"] * e["repeat"] for e in extra)
    return {
        "sentiment_tweets": len(train) + len(val) - in_extra,
        "train": len(train),
        "val": len(val),
        "extra": extra,
    }


def scores_table(row: dict) -> str:
    """Test and dev scores side by side, with argmax predictions."""
    lines = [
        ("Macro F1", "macro_f1"),
        ("Accuracy", "accuracy"),
        ("Bullish precision", "precision_bullish"),
        ("Bullish recall", "recall_bullish"),
        ("Bearish precision", "precision_bearish"),
        ("Bearish recall", "recall_bearish"),
        ("Neutral recall", "recall_neutral"),
        ("Calibration error (ECE)", "ece_calibrated"),
    ]
    table = ["| Metric | Test (1,237 tweets) | Dev (749 tweets) |", "| --- | --- | --- |"]
    for title, key in lines:
        cells = [row.get(f"{split}/{key}") for split in ("test", "dev")]
        if all(c is None for c in cells):
            continue
        table.append(
            f"| {title} | " + " | ".join("–" if c is None else f"{c:.3f}" for c in cells) + " |"
        )
    return "\n".join(table)


def model_card(
    repo: str, base: str, experiment: dict, seed: int, args: dict, data: dict, row: dict
) -> str:
    """The README.md for a fine-tuned sentiment model."""
    name = repo.split("/")[-1]
    base_name = base.rstrip("/").split("/")[-1]
    base_link = f"[{base_name}](https://huggingface.co/{base})" if "/" in base else f"`{base}`"
    description = BASE_DESCRIPTIONS.get(base)

    front = [
        "license: mit",
        "datasets:\n- TimKoornstra/financial-tweets-sentiment\n- StephanAkkerman/financial-tweets",
        "language:\n- en",
        "metrics:\n- accuracy\n- f1\n- precision\n- recall",
        "library_name: transformers",
        "pipeline_tag: text-classification",
        "tags:\n" + "\n".join(f"- {tag}" for tag in TAGS),
        f"base_model: {base}",
        "widget:\n"
        + "\n".join(
            f"- text: {json.dumps(text, ensure_ascii=False)}\n  example_title: {title}"
            for text, title in WIDGETS
        ),
    ]
    if "test/macro_f1" in row:
        metrics = [("f1", "macro_f1", "Macro F1"), ("accuracy", "accuracy", "Accuracy")]
        front.append(
            f"model-index:\n- name: {name}\n  results:\n  - task:\n"
            "      type: text-classification\n      name: Sentiment Analysis\n"
            "    dataset:\n      name: Held-out financial tweets (test_v0)\n"
            "      type: financial-tweets\n    metrics:\n"
            + "\n".join(
                f"    - type: {kind}\n      name: {title}\n      value: {row[f'test/{key}']:.3f}"
                for kind, key, title in metrics
                if f"test/{key}" in row
            )
        )

    datasets = [
        "- [TimKoornstra/financial-tweets-sentiment](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment): "
        f"{data['sentiment_tweets']:,} human-labeled tweets after cleaning. The sources without "
        "a neutral class or with inconsistent labels were left out, and the bullish and bearish "
        "tweets of the other sources were relabeled by hand with one labeling guideline (most "
        "became neutral, since they state facts or news rather than an opinion)."
    ]
    for extra in data["extra"]:
        line = (
            f"- {extra['tweets']:,} tweets from [StephanAkkerman/financial-tweets]"
            "(https://huggingface.co/datasets/StephanAkkerman/financial-tweets), posted from 2024 "
            "onwards, picked where an earlier model was least sure and labeled with the same "
            f"guideline ([`{extra['path']}`](https://github.com/StephanAkkerman/ModernFinTwitBERT/blob/main/{extra['path']}))"
        )
        if extra["repeat"] > 1:
            line += f", shown {extra['repeat']} times per epoch"
        datasets.append(line)

    batch = args["per_device_train_batch_size"] * args.get("gradient_accumulation_steps", 1)
    warmup = args.get("warmup_steps", 0)
    warmup = (
        f"{warmup:.0%} warmup" if isinstance(warmup, float) and warmup < 1 else f"{warmup} warmup steps"
    )
    epochs = args["num_train_epochs"]
    recipe = (
        f"{epochs:g} epoch{'' if epochs == 1 else 's'}, learning rate {args['learning_rate']:g} "
        f"with {warmup}"
    )
    if "llrd" in experiment:
        recipe += f" and layer-wise learning-rate decay of {experiment['llrd']:g}"
    recipe += (
        f", weight decay {args.get('weight_decay', 0):g}, batch size {batch}, maximum length "
        f"{args.get('max_length', 128)} tokens, seed {seed}"
    )
    if experiment.get("class_weights"):
        recipe += f", {experiment['class_weights']} class weights"
    if args.get("label_smoothing_factor"):
        recipe += f", label smoothing {args['label_smoothing_factor']:g}"
    recipe += (
        f". The checkpoint with the best validation F1 ({data['val']:,} tweets held out from the "
        "training sources) is kept"
    )

    details = [
        f"- **Base model:** {base_link}" + (f", {description}." if description else "."),
        "- **Labels:** `NEUTRAL`, `BULLISH`, `BEARISH`.",
        f"- **Training:** {recipe}. The recipe (`{experiment['name']}` in `experiments.json`) was chosen "
        "out of a grid of learning rates, epochs, class weights, label smoothing and layer-wise "
        "decay by its macro F1 on the dev set.",
    ]
    if "temperature" in row:
        details.append(
            f"- **Calibration:** `config.calibration_temperature` = {row['temperature']:.3f}, "
            "fitted on the dev set. Divide the logits by it before the softmax to get confidence "
            "scores that match how often the model is right."
        )

    comparison = (
        f" For comparison, [FinTwitBERT-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment) "
        f"scores a macro F1 of {FINTWITBERT_SENTIMENT_F1:.3f} on the same test set."
    )
    interval = ""
    if "test/macro_f1_low" in row:
        interval = (
            f" The 95% bootstrap interval of the test macro F1 is "
            f"{row['test/macro_f1_low']:.3f}–{row['test/macro_f1_high']:.3f}."
        )

    return f"""---
{chr(10).join(front)}
---

# {name}

{name} is a fine-tuned model for classifying the sentiment of financial tweets as bullish, bearish or neutral. It uses {base_link} as a base model{", " + description if description else ""}.
This approach ensures that {name} has seen enough financial tweets, which have an informal nature, compared to other financial texts, such as news headlines.
Therefore this model performs great on informal financial texts, seen on social media.
It is the successor of [FinTwitBERT-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment).

## Intended Uses

{name} is intended for classifying financial tweets or other financial social media texts, such as a feed of tweets about stocks and crypto. A tweet is bullish or bearish only when its author expresses an expectation or position; facts, news, questions and price alerts are neutral, which is most of a typical feed. The sentiment is that of the tweet's own text: for a quote tweet, classify the new text without the quoted tweet.

## Dataset

{name} has been trained on human-labeled financial tweets from two datasets:
{chr(10).join(datasets)}

In total {data['train']:,} tweets for training and {data['val']:,} for validation. The tweets of our dev and test sets were removed from the training data.

## Model Details

{chr(10).join(details)}

## Evaluation

The test set has 1,237 tweets from a real financial Twitter feed (first seen from August 2023 onwards) and the dev set 749 tweets from the same feed (2024 onwards), labeled with the same guideline. Neither was used for training; the dev set was used to choose the recipe and fit the temperature.{interval}{comparison}

{scores_table(row)}

## More Information

For a comprehensive overview, including the training setup and analysis of the model, visit the [ModernFinTwitBERT GitHub repository](https://github.com/StephanAkkerman/ModernFinTwitBERT).

## Usage

Using [HuggingFace's transformers library](https://huggingface.co/docs/transformers/index) the model and tokenizers can be converted into a pipeline for text classification.

```python
from transformers import pipeline

# Create a sentiment analysis pipeline
pipe = pipeline(
    "sentiment-analysis",
    model="{repo}",
)

# Get the predicted sentiment
print(pipe("Nice 9% pre market move for $para, pump my calls Uncle Buffett 🤑"))
```

The model was trained on tweets with links replaced by `[URL]` and user mentions by `@USER`, so do the same for the best results. The pipeline's scores are not calibrated; for confidence scores, divide the logits by `config.calibration_temperature`:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForSequenceClassification.from_pretrained("{repo}").eval()

inputs = tokenizer(["$NVDA breaking out, loading calls"], return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
probs = (logits / model.config.calibration_temperature).softmax(-1)
print({{model.config.id2label[i]: p for i, p in enumerate(probs[0].tolist())}})
```

## Long texts and several tickers

A tweet is best classified as a whole: on our test and dev sets, splitting every tweet into sentences and averaging their scores was about 0.04 macro F1 worse than reading the whole tweet, because a sentence on its own loses the context of the others ("I hate this coin. But above trend, good"). Two cases still need more than one call: a text longer than the model's 512 tokens, which would otherwise be cut off, and a tweet with a different view on each of its tickers. The snippet below reads a text as a whole when it fits, and otherwise averages the signed scores of its sentences. For each ticker, it averages the sentences that name only that ticker. These per-ticker scores are not measured, since our test sets have one label per tweet.

```python
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "{repo}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
SIGN = {{"BULLISH": 1.0, "BEARISH": -1.0}}
SEGMENT = re.compile(r"(?<=[.!?])\\s+|[\\n;]+")


def preprocess(text):
    return re.sub(r"@\\S+", "@USER", re.sub(r"http\\S+", "[URL]", text))


def signed_scores(texts):
    # +p for bullish, -p for bearish and 0 for neutral, with calibrated confidence p
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        probs = (model(**inputs).logits / model.config.calibration_temperature).softmax(-1)
    conf, label = probs.max(-1)
    return [SIGN.get(model.config.id2label[i], 0.0) * c for i, c in zip(label.tolist(), conf.tolist())]


def sentiment(text, tickers=(), threshold=0.1):
    # The text's sentiment as a signed score, and one per ticker with sentences of its own
    text = preprocess(text)
    segments = [s.strip() for s in SEGMENT.split(text) if len(s.strip()) >= 12] or [text]
    # A tweet that fits the model is read as a whole: splitting it loses context
    fits = len(tokenizer(text)["input_ids"]) <= 512
    scores = signed_scores(segments)
    overall = signed_scores([text])[0] if fits else sum(scores) / len(scores)

    # Per ticker: the sentences that name it and no other ticker, if the text has several sentences
    patterns = {{t: re.compile(rf"[$#]?\\b{{re.escape(t)}}\\b", 0 if len(t) < 4 else re.I) for t in tickers}}
    named = [{{t for t, p in patterns.items() if p.search(s)}} for s in segments]
    per_ticker = {{}}
    for ticker in tickers:
        own = [score for score, names in zip(scores, named) if names == {{ticker}}]
        if own and len(segments) > 1:
            per_ticker[ticker] = sum(own) / len(own)

    label = lambda s: "BULLISH" if s > threshold else "BEARISH" if s < -threshold else "NEUTRAL"
    return label(overall), overall, {{t: (label(s), s) for t, s in per_ticker.items()}}


print(sentiment("$NVDA breaking out, loading calls. $INTC looks dead, sold my shares.", ["NVDA", "INTC"]))
# ('BULLISH', 0.55, {{'NVDA': ('BULLISH', 0.99), 'INTC': ('BEARISH', -0.96)}})
```

## Citing & Authors

If you use {name} or FinTwitRoBERTa-large in your research, please cite me as follows:

```
@misc{{FinTwitRoBERTalarge,
  author = {{Stephan Akkerman}},
  title = {{FinTwitRoBERTa-large}},
  year = {{2026}},
  publisher = {{GitHub}},
  journal = {{GitHub repository}},
  howpublished = {{\\url{{https://github.com/StephanAkkerman/ModernFinTwitBERT}}}}
}}
```

Additionally, if you utilize the sentiment classifier, please cite:

```
@misc{{{name.replace("-", "")},
  author = {{Stephan Akkerman}},
  title = {{{name}: A Sentiment Classifier for Financial Tweets}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo}}}}}
}}
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.
"""


def set_section(card: str, section: str) -> str:
    """
    Put a "## Title" section in the card: in place of the one with the same title, else
    before "## Citing & Authors", else at the end.
    """
    title = section.splitlines()[0]
    section = section.strip() + "\n"
    pattern = re.compile(rf"^{re.escape(title)}\n.*?(?=^## |\Z)", re.DOTALL | re.MULTILINE)
    if pattern.search(card):
        return pattern.sub(lambda _: section + "\n", card, count=1).rstrip() + "\n"
    if "\n## Citing & Authors" in card:
        return card.replace("\n## Citing & Authors", f"\n{section}\n## Citing & Authors", 1)
    return card.rstrip() + "\n\n" + section


def added_sections(card: str) -> list[str]:
    """The sections that other scripts added to a card, see ADDED_SECTIONS."""
    sections = []
    for title in ADDED_SECTIONS:
        match = re.search(rf"^## {re.escape(title)}\n.*?(?=^## |\Z)", card, re.DOTALL | re.MULTILINE)
        if match:
            sections.append(match.group(0))
    return sections


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from data import load_dev_data, load_finetuning_data, load_test_data
    from experiments import expand, labeled, test_metrics

    parser = argparse.ArgumentParser(description="Write the sentiment model's card")
    parser.add_argument("--model", default="StephanAkkerman/FinTwitRoBERTa-sentiment")
    parser.add_argument("--experiment", required=True, help="Its name in experiments.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", help="Repo id in the card, by default --hub-repo or --model")
    parser.add_argument("--hub-repo", help="Upload README.md here")
    args = parser.parse_args()

    config = json.loads(Path("config.json").read_text())
    plan = json.loads(Path("experiments.json").read_text())
    experiment = next((e for e in expand(plan["experiments"]) if e["name"] == args.experiment), None)
    if experiment is None:
        parser.error(f"No experiment {args.experiment} in experiments.json")
    base = plan.get("base", config["finetune"]["base_model"])
    repo = args.repo or args.hub_repo or (args.model if not Path(args.model).is_dir() else DEFAULT_REPO)

    extra = experiment.get("extra_train", config["finetune"].get("extra_train"))
    train, val = load_finetuning_data(
        drop_sources=config["finetune"].get("drop_sources"),
        relabel=config["finetune"].get("relabel", True),
        extra_train=extra,
    )
    data = data_stats(train, val, extra)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    temperature = getattr(model.config, "calibration_temperature", 1.0) or 1.0
    max_length = config["finetune"].get("max_length", 128)
    row = {"temperature": temperature}
    for split, dataset in (("test", load_test_data()), ("dev", load_dev_data())):
        texts, logits = dataset["text"], []
        for start in range(0, len(texts), 32):
            inputs = tokenizer(
                texts[start : start + 32],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            with torch.no_grad():
                logits.append(model(**inputs).logits.float())
        logits = torch.cat(logits)
        frame = labeled(dataset)
        row.update(test_metrics(logits.softmax(-1).numpy(), frame, prefix=split))
        calibrated = (logits / temperature).softmax(-1).numpy()
        row[f"{split}/ece_calibrated"] = test_metrics(calibrated, frame, prefix=split)[f"{split}/ece"]
        print(f"{split}: macro F1 {row[f'{split}/macro_f1']:.3f}")

    card = model_card(
        repo, base, experiment, args.seed, training_args(config, experiment), data, row
    )
    if Path(args.model).is_dir():
        folder, current = Path(args.model), Path(args.model) / "README.md"
    else:
        folder = Path("output") / Path(args.model).name
        try:
            current = Path(hf_hub_download(args.model, "README.md"))
        except Exception:
            current = None
    if current and current.exists():
        for section in added_sections(current.read_text()):
            card = set_section(card, section)
    folder.mkdir(parents=True, exist_ok=True)
    readme = folder / "README.md"
    readme.write_text(card)
    print(f"Wrote the model card to {readme}")

    if args.hub_repo:
        HfApi().upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=args.hub_repo,
            commit_message="Update model card",
        )
        print(f"Uploaded the card to https://huggingface.co/{args.hub_repo}")


if __name__ == "__main__":
    main()
