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


WSB_TAGS = ["finance", "reddit", "wallstreetbets", "sentiment-analysis", "RoBERTa", "FinTwitBERT", "stocks"]


def wsb_card(repo: str, base: str, experiment: dict, seed: int, args: dict, row: dict) -> str:
    """The README.md for a WSB sentiment model saved by experiments.py --save."""
    name = repo.split("/")[-1]
    base_link = f"[{base.split('/')[-1]}](https://huggingface.co/{base})" if "/" in base else base
    rows = [
        ("Macro F1", "macro_f1"),
        ("Accuracy", "accuracy"),
        ("Bullish precision", "precision_bullish"),
        ("Bullish recall", "recall_bullish"),
        ("Bearish precision", "precision_bearish"),
        ("Bearish recall", "recall_bearish"),
        ("Neutral recall", "recall_neutral"),
    ]
    table = ["| Metric | WSB test (227 posts) | Tweet test (1,237 tweets) |", "| --- | --- | --- |"]
    for title, key in rows:
        cells = [row.get(f"{split}/{key}") for split in ("test", "tweets")]
        table.append(f"| {title} | " + " | ".join("–" if c is None else f"{c:.3f}" for c in cells) + " |")
    data = "the training posts (70%) of the all-agree split"
    if experiment.get("wsb_full"):
        data += ", plus the posts the annotators did not all agree on (majority label)"
    if "with_tweets" in experiment:
        data += (
            f", repeated {experiment['with_tweets'].get('wsb_repeat', 1)} times, mixed with the "
            "labeled tweets FinTwitRoBERTa-sentiment was trained on"
        )
    batch = args["per_device_train_batch_size"] * args.get("gradient_accumulation_steps", 1)
    llrd = f", layer-wise learning rate decay {experiment['llrd']}" if "llrd" in experiment else ""
    temperature = row.get("temperature")
    front = "\n".join(
        [
            "---",
            "license: mit",
            "datasets:\n- moonscape95/WSBS",
            "language:\n- en",
            "library_name: transformers",
            "pipeline_tag: text-classification",
            "tags:\n" + "\n".join(f"- {tag}" for tag in WSB_TAGS),
            f"base_model: {base}",
            "widget:\n- text: " + json.dumps("Come on TSLA. Fly you regarded bird!!"),
            "---",
        ]
    )
    calibration = (
        f" The scores are not calibrated; for confidence scores, divide the logits by "
        f"`config.calibration_temperature` ({temperature:.3f}, fitted on the validation posts)."
        if temperature
        else ""
    )
    return f"""{front}
# {name}

{name} classifies r/wallstreetbets posts and comments as **bullish**, **bearish** or **neutral**. It is {base_link} fine-tuned on Reddit posts, for the slang, sarcasm and longer texts of retail trading subreddits. For tweets, use [FinTwitRoBERTa-sentiment](https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-sentiment).

## Results

The WSB test posts are the 15% of [WSBS](https://github.com/moonscape95/WSBS) `WSB_all_agree.csv` held out by [FinTwitBERT-wsb-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-wsb-sentiment), so the two compare directly. With 227 posts, differences of a few points are within noise. FinTwitBERT-wsb-sentiment scores {OLD_WSB_F1:.3f} macro F1 on them. The tweet test set shows how well it still reads tweets.

{chr(10).join(table)}

## Training

- **Data:** {data}. The checkpoint and the temperature were chosen on the 226 validation posts.
- **Arguments:** learning rate {args["learning_rate"]}, {args["num_train_epochs"]} epochs, batch size {batch}, max length {args["max_length"]} tokens{llrd}, seed {seed}.
- **Code:** [ModernFinTwitBERT](https://github.com/StephanAkkerman/ModernFinTwitBERT) (`experiments.py --config experiments_wsb.json`, experiment `{experiment["name"]}`).

## Usage

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="<this repo>")
print(pipe("Come on TSLA. Fly you regarded bird!!"))
```

Links were replaced by `[URL]` and user mentions by `@USER` during training.{calibration}

## Labels

- 0: NEUTRAL
- 1: BULLISH
- 2: BEARISH
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
