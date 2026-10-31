"""
r/wallstreetbets data for the WSB sentiment model (experiments_wsb.json).

The labels come from the WSBS dataset (https://github.com/moonscape95/WSBS): Reddit
posts and comments labeled 0: neutral, 1: bullish, 2: bearish, as ours. WSB_all_agree.csv
holds the 1,509 posts all annotators agreed on; WSB_full.csv adds 1,411 more that
they did not all agree on.

The split is the one FinTwitBERT-wsb-sentiment was trained with (its train.py on
Hugging Face): 70% train, 15% validation, 15% test of WSB_all_agree.csv, stratified,
random_state 42. Its test posts are therefore new to both models, and scores compare
directly. The test set is small (227 posts), so look at the bootstrap interval.

Score a model on the WSB test split (e.g. the old one):

    python ModernFinTwitBERT/wsb.py --score StephanAkkerman/FinTwitBERT-wsb-sentiment
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

from data import LABELS, dedup_key, preprocess_dataset
from preprocess import preprocess_tweet

WSB_REPO = "StephanAkkerman/FinTwitRoBERTa-wsb-sentiment"
# FinTwitBERT-wsb-sentiment on the test split, measured with `wsb.py --score`
OLD_WSB_F1 = 0.821
WSBS_URL = "https://raw.githubusercontent.com/moonscape95/WSBS/main/data/{}"
CACHE = Path("datasets") / "wsbs"
# Row counts of the published files, to notice if the upstream data changes
EXPECTED_ROWS = {"WSB_all_agree.csv": 1509, "WSB_full.csv": 2920}

# Reddit emotes and images: ![img](emote|t5_2th52|4271), ![gif](giphy|...)
REDDIT_MEDIA = re.compile(r"!\[(?:img|gif)\]\([^)]*\)")
# u/name and /u/name mentions
REDDIT_USER = re.compile(r"(?<![\w/])/?u/[\w-]+")


def download(name: str) -> pd.DataFrame:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WSBS_URL.format(name), path)
    dataframe = pd.read_csv(path)
    if len(dataframe) != EXPECTED_ROWS[name]:
        print(
            f"Warning: {name} has {len(dataframe)} rows, expected {EXPECTED_ROWS[name]}; "
            "the test split no longer matches FinTwitBERT-wsb-sentiment's"
        )
    return dataframe


def preprocess_post(text: str) -> str:
    """Tweet preprocessing plus Reddit markup: emotes and images dropped, u/ mentions to @USER."""
    text = REDDIT_MEDIA.sub(" ", text)
    text = REDDIT_USER.sub("@USER", text)
    text = preprocess_tweet(text)
    return re.sub(r"[ \t]+", " ", text).strip()


def raw_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, validation and test rows of WSB_all_agree.csv, exactly as the old model's."""
    dataframe = download("WSB_all_agree.csv")
    dataframe["label"] = dataframe["label"].map({0: 0, 1: 1, 2: 2, -1: 2})
    dataframe = dataframe.dropna(subset=["label", "text"])
    dataframe["label"] = dataframe["label"].astype(int)
    train_val, test = train_test_split(
        dataframe, test_size=0.15, stratify=dataframe["label"], random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=0.176, stratify=train_val["label"], random_state=42
    )
    return train, val, test


def to_dataset(dataframe: pd.DataFrame, file: str | None = None) -> Dataset:
    dataframe = dataframe.assign(text=dataframe["text"].map(preprocess_post))
    columns = ["text", "label"]
    if file:
        # score() reports accuracy per `file`
        dataframe = dataframe.assign(file=file)
        columns.append("file")
    dataset = Dataset.from_pandas(dataframe[columns], preserve_index=False)
    return dataset.cast_column("label", ClassLabel(names=LABELS))


def load_wsb_data(full: bool = False) -> tuple[Dataset, Dataset]:
    """
    WSB training and validation posts.

    Parameters
    ----------
    full : bool, optional
        Also train on the WSB_full.csv posts the annotators did not all agree on
        (their majority label), by default False. Validation and test posts are never added.
    """
    train, val, test = raw_splits()
    if full:
        held_out = set(pd.concat([val, test])["text"].map(dedup_key))
        extra = download("WSB_full.csv").dropna(subset=["label", "text"])
        extra = extra[~extra["text"].map(dedup_key).isin(held_out)]
        train = pd.concat([train, extra[["text", "label"]]])
    train_set = to_dataset(train)
    # The same cleanup as the tweets (drops near-duplicates and one-word posts), train only
    train_set = preprocess_dataset(train_set).cast_column("label", ClassLabel(names=LABELS))
    return train_set.shuffle(seed=42), to_dataset(val)


def load_wsb_dev() -> Dataset:
    """The validation posts, also used to fit the temperature and to pick experiments."""
    return to_dataset(raw_splits()[1], file="wsb")


def load_wsb_test() -> Dataset:
    """The 227 test posts of the old model's split."""
    return to_dataset(raw_splits()[2], file="wsb")


def mix(wsb: Dataset, tweets: Dataset, wsb_repeat: int) -> Dataset:
    """Tweet training data plus the WSB posts repeated `wsb_repeat` times."""
    tweets = tweets.select_columns(["text", "label"]).cast(wsb.features)
    return concatenate_datasets([tweets] + [wsb] * wsb_repeat).shuffle(seed=42)


WSB_TAGS = [
    "finance",
    "reddit",
    "wallstreetbets",
    "sentiment-analysis",
    "NLP",
    "RoBERTa",
    "FinTwitBERT",
    "FinTwitRoBERTa",
    "sentiment",
    "financial-analysis",
    "financial-sentiment-analysis",
    "stocks",
    "stock-market",
    "crypto",
    "cryptocurrency",
]

WSB_WIDGETS = [
    ("Come on TSLA. Fly you regarded bird!!", "Bullish WSB post"),
    ("Lmfao so many puts got bent", "Bullish WSB post 2"),
    ("Bulls asses are about to be bleeding after this upcoming dump", "Bearish WSB post"),
    ("$BBBY has 40% short interest. It will probably go bankrupt and everyone knows it.", "Bearish WSB post 2"),
    ("What are SPY calls? Asking for a friend", "Neutral WSB post"),
]
# FinTwitRoBERTa-sentiment without WSB training, on the same test posts
FINTWITROBERTA_SENTIMENT_WSB_F1 = 0.657


def wsb_card(
    repo: str, base: str, experiment: dict, seed: int, args: dict, row: dict, data: dict
) -> str:
    """
    The README.md for a WSB sentiment model saved by experiments.py --save, in the layout
    of the FinTwitRoBERTa-sentiment card. `data` has the "train" and "val" post counts.
    """
    name = repo.split("/")[-1]
    base_name = base.rstrip("/").split("/")[-1]
    base_link = f"[{base_name}](https://huggingface.co/{base})" if "/" in base else f"`{base}`"

    front = [
        "license: mit",
        "datasets:\n- moonscape95/WSBS",
        "language:\n- en",
        "metrics:\n- accuracy\n- f1\n- precision\n- recall",
        "library_name: transformers",
        "pipeline_tag: text-classification",
        "tags:\n" + "\n".join(f"- {tag}" for tag in WSB_TAGS),
        f"base_model: {base}",
        "widget:\n"
        + "\n".join(
            f"- text: {json.dumps(text, ensure_ascii=False)}\n  example_title: {title}"
            for text, title in WSB_WIDGETS
        ),
    ]
    if "test/macro_f1" in row:
        metrics = [("f1", "macro_f1", "Macro F1"), ("accuracy", "accuracy", "Accuracy")]
        front.append(
            f"model-index:\n- name: {name}\n  results:\n  - task:\n"
            "      type: text-classification\n      name: Sentiment Analysis\n"
            "    dataset:\n      name: WSBS all-agree test split (227 posts)\n"
            "      type: moonscape95/WSBS\n    metrics:\n"
            + "\n".join(
                f"    - type: {kind}\n      name: {title}\n      value: {row[f'test/{key}']:.3f}"
                for kind, key, title in metrics
                if f"test/{key}" in row
            )
        )

    sources = [
        "- [WSBS](https://github.com/moonscape95/WSBS) `WSB_all_agree.csv`: 1,509 r/wallstreetbets "
        "posts and comments that all annotators labeled the same, split 70/15/15 into training, "
        "validation and test posts exactly as for FinTwitBERT-wsb-sentiment."
    ]
    if experiment.get("wsb_full"):
        sources.append(
            "- WSBS `WSB_full.csv`: the posts the annotators did not all agree on, with their "
            "majority label, added to the training posts only."
        )
    if "with_tweets" in experiment:
        repeat = experiment["with_tweets"].get("wsb_repeat", 1)
        sources.append(
            "- The labeled financial tweets that "
            "[FinTwitRoBERTa-sentiment](https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-sentiment) "
            f"was trained on, with the WSB posts repeated {repeat} times so they are not drowned out."
        )

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
        f"{args.get('max_length', 256)} tokens, seed {seed}. The checkpoint with the best "
        f"validation F1 ({data['val']:,} WSB posts) is kept"
    )
    details = [
        f"- **Base model:** {base_link}.",
        "- **Labels:** `NEUTRAL`, `BULLISH`, `BEARISH`.",
        f"- **Training:** {recipe}. The recipe (`{experiment['name']}` in `experiments_wsb.json`) "
        "was chosen by its macro F1 on the WSB validation posts.",
    ]
    if "temperature" in row:
        details.append(
            f"- **Calibration:** `config.calibration_temperature` = {row['temperature']:.3f}, "
            "fitted on the WSB validation posts. Divide the logits by it before the softmax to get "
            "confidence scores that match how often the model is right."
        )

    interval = ""
    if "test/macro_f1_low" in row:
        interval = (
            f" The 95% bootstrap interval of the test macro F1 is "
            f"{row['test/macro_f1_low']:.3f}–{row['test/macro_f1_high']:.3f}: with 227 posts, "
            "differences of a few points are noise."
        )
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
    table = [
        "| Metric | WSB test (227 posts) | WSB validation (226 posts) | Tweet test (1,237 tweets) |",
        "| --- | --- | --- | --- |",
    ]
    for title, key in lines:
        cells = [row.get(f"{split}/{key}") for split in ("test", "dev", "tweets")]
        if all(c is None for c in cells):
            continue
        table.append(
            f"| {title} | " + " | ".join("–" if c is None else f"{c:.3f}" for c in cells) + " |"
        )

    return f"""---
{chr(10).join(front)}
---

# {name}

{name} is a fine-tuned model for classifying the sentiment of r/wallstreetbets posts and comments as bullish, bearish or neutral. It uses {base_link} as a base model and was fine-tuned on Reddit posts, because the slang, sarcasm and longer texts of retail trading subreddits differ from tweets.
It is the successor of [FinTwitBERT-wsb-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-wsb-sentiment). For tweets, use [FinTwitRoBERTa-sentiment](https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-sentiment).

## Intended Uses

{name} is intended for classifying posts and comments on r/wallstreetbets and similar retail trading subreddits. Posts up to {args.get('max_length', 256)} tokens are read as a whole (about 97% of WSB posts); longer ones are cut off unless you split them.

## Dataset

{name} has been trained on:
{chr(10).join(sources)}

In total {data['train']:,} texts for training and {data['val']:,} WSB posts for validation. The test posts were never used for training or for choosing the recipe.

## Model Details

{chr(10).join(details)}

## Evaluation

The test set is the 227 held-out posts of the WSBS split that FinTwitBERT-wsb-sentiment was also tested on, so the scores compare directly: FinTwitBERT-wsb-sentiment scores a macro F1 of {OLD_WSB_F1:.3f} and FinTwitRoBERTa-sentiment (no WSB training) {FINTWITROBERTA_SENTIMENT_WSB_F1:.3f}.{interval} The tweet test set (1,237 tweets from a financial Twitter feed) shows how well the model still reads tweets.

{chr(10).join(table)}

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
print(pipe("Come on TSLA. Fly you regarded bird!!"))
```

The model was trained with links replaced by `[URL]`, user mentions (`u/name`) by `@USER` and Reddit emote markup removed, so do the same for the best results. The pipeline's scores are not calibrated; for confidence scores, divide the logits by `config.calibration_temperature`:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForSequenceClassification.from_pretrained("{repo}").eval()

inputs = tokenizer(["GME to the moon, holding my calls"], return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
probs = (logits / model.config.calibration_temperature).softmax(-1)
print({{model.config.id2label[i]: p for i, p in enumerate(probs[0].tolist())}})
```

## Citing & Authors

If you use {name} in your research, please cite me as follows:

```
@misc{{{name.replace("-", "")},
  author = {{Stephan Akkerman}},
  title = {{{name}: A Sentiment Classifier for r/wallstreetbets}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo}}}}}
}}
```

The training data is the WSBS dataset by [moonscape95](https://github.com/moonscape95/WSBS).

## License

This project is licensed under the MIT License. See the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.
"""


def score_model(repo: str, max_length: int = 512):
    """Macro F1 and per-class scores of a Hugging Face model on the WSB test split."""
    from eval import ALIASES, score
    from transformers import pipeline

    test = load_wsb_test().to_pandas()
    pipe = pipeline("text-classification", model=repo, truncation=True, max_length=max_length)
    outputs = pipe(test["text"].tolist(), batch_size=16)
    test["label"] = test["label"].map(dict(enumerate(LABELS)))
    test["pred"] = [ALIASES.get(o["label"].lower(), o["label"].lower()) for o in outputs]
    test["confidence"] = [o["score"] for o in outputs]
    for metric, value in score(test).items():
        print(f"{metric}: {value:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score", required=True, help="Hugging Face model to score")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    score_model(args.score, args.max_length)
