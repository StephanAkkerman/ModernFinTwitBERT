# > Imports
# Standard library
import hashlib
import html
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Third party
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset

# Frozen test set: 1,237 tweets from the financial-tweets export, first seen on or
# after 2023-08-01, labeled with one guideline. Never train or tune on it.
TEST_SET_PATH = Path(__file__).parent.parent / "data" / "test" / "test_v0.csv"
LABELS = ["neutral", "bullish", "bearish"]

# Tweet ids of all 1,500 test candidates (test_v0 plus the tweets dropped as unsure), to
# keep them out of pretraining even when their text was reformatted
HELDOUT_IDS_PATH = TEST_SET_PATH.parent / "heldout_ids_v0.txt"

# Dev set: tweets from the same feed, 2024 onwards, labeled with the same guideline, for
# choosing between experiments so the test set stays untouched. dev_ids_v0.txt lists all
# 750 dev candidates (the unsure ones too), kept out of training like the test tweets.
DEV_SET_PATH = TEST_SET_PATH.parent / "dev_v0.csv"
DEV_IDS_PATH = TEST_SET_PATH.parent / "dev_ids_v0.txt"

# Cleaned pretraining tweets, one folder per version of the sources, this file and the
# test set, so changing any of them cleans again
PRETRAIN_CACHE = Path("datasets") / "pretrain_clean"

# Used when config.json has no pretrain "datasets"
DEFAULT_PRETRAIN_DATASETS = [{"name": "StephanAkkerman/crypto-stock-tweets"}]

# The directional tweets of the moritz, pooja, fiqa, zeroshot and ieee sources, relabeled
# by hand to the same guideline as the test set (columns: tweet, label). Most become neutral.
RELABELS_PATH = Path(__file__).parent.parent / "data" / "train" / "relabels_v1.csv"

# Extra training files are relative to the repository root, e.g. data/train/feed_v1.csv:
# 1,500 tweets from the same feed as the dev and test sets (2024 onwards, none of the dev
# or test tweets), picked where the model was least sure and labeled with the guideline
# (columns: tweet_id, text, label, label_source)
REPO_ROOT = Path(__file__).parent.parent

# Short names for the sources in TimKoornstra/financial-tweets-sentiment (url column)
SOURCES = {
    "moritzwilksch": "moritz",
    "poojathakoor": "pooja",
    "crypto-sentiment": "surge_crypto",
    "stock-sentiment": "surge_stock",
    "fiqa": "fiqa",
    "zeroshot": "zeroshot",
    "ieee": "ieee",
    "mattgilgo": "mattgilgo",
    "yash612": "yash",
}


# Discord embeds add the quoted (or replied-to) tweet below the tweet itself, as
# "> [@user](https://twitter.com/user):" followed by "> " lines
QUOTE = re.compile(r"^> \[@[^\]]*\]\(https?://[^)\s]*\):\n?((?:^>.*(?:\n|$))*)", re.MULTILINE)


def split_quote(tweet: str) -> list[str]:
    """The tweet's own text and, if it quotes one, the quoted tweet's text."""
    # Some exports use \r\n, which QUOTE would not see past the "):" header
    tweet = tweet.replace("\r\n", "\n")
    match = QUOTE.search(tweet)
    if not match:
        return [tweet]
    own = tweet[: match.start()].strip()
    quoted = re.sub(r"^> ?", "", match.group(1), flags=re.MULTILINE).strip()
    return [part for part in (own, quoted) if part] or [own]


def preprocess_tweet(tweet: str) -> str:
    """
    Preprocess a tweet by replacing URLs, @mentions, and cash tags with tokens.

    Parameters
    ----------
    tweet : str
        The tweet to preprocess.

    Returns
    -------
    str
        The preprocessed tweet.
    """
    # Unescape HTML characters
    tweet = html.unescape(tweet)

    # Only the tweet's own words carry its sentiment; quoted tweets are scored on their own
    tweet = split_quote(tweet)[0]
    # Markdown links from Discord embeds
    tweet = re.sub(r"\[([^\]]*)\]\(https?://[^)\s]*\)", r"\1", tweet)

    # Replace URLs wiht URL token
    tweet = re.sub(r"http\S+", "[URL]", tweet)

    # Replace @mentions with @USER token
    tweet = re.sub(r"@\S+", "@USER", tweet)

    # Replace cash tags with [TICKER] token
    # tweet = re.sub(r"\$[A-Z]{1,5}\b", "[TICKER]", tweet)

    return tweet


def dedup_key(text: str) -> str:
    """Normalized text used to find near-duplicates (same tweet, different link or mention)."""
    text = re.sub(r"https?://\S+|\[URL\]", " ", html.unescape(text).lower())
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\W+", " ", text).strip()


def test_set_keys() -> set:
    """Dedup keys of the frozen test set and the dev set, to remove them from training data."""
    texts = pd.concat([pd.read_csv(path)["text"] for path in (TEST_SET_PATH, DEV_SET_PATH)])
    return set(texts.map(dedup_key))


def split_dataset(
    dataset: Dataset, val_size: float = 0.1, test_size: float = 0.1
) -> tuple:
    """Split the dataset into 3 parts: train, validation, and test.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    val_size : float, optional
        The size of the validation set, by default 0.1
    test_size : float, optional
        The size of the test set, by default 0.1

    Returns
    -------
    tuple
        Datsets for training, validation, and testing.
    """
    # First split: 80% train, 20% temp (val + test)
    split_ds = dataset.train_test_split(test_size=val_size + test_size, seed=42)

    # Second split on the temporary set to get validation and test (split temp into 50/50)
    temp_ds = split_ds["test"]
    split_size = val_size / (val_size + test_size)
    split_temp = temp_ds.train_test_split(test_size=split_size, seed=42)

    return split_ds["train"], split_temp["train"], split_temp["test"]


def load_pretraining_source(source: dict) -> Dataset:
    """
    One pretraining source with a "text" column and a "heldout" column that marks the
    test candidates, found by id where the source has tweet ids or tweet urls.
    """
    name = source["name"]
    suffix = Path(name).suffix.lstrip(".")
    if suffix in ("csv", "json", "jsonl", "parquet"):
        # A local file, e.g. a fresh scrape
        builder = "json" if suffix == "jsonl" else suffix
        dataset = load_dataset(builder, data_files=name, split="train")
    else:
        dataset = load_dataset(name, split="train", cache_dir="datasets")
    heldout_ids = set(HELDOUT_IDS_PATH.read_text().split() + DEV_IDS_PATH.read_text().split())
    id_column = next((c for c in ("tweet_id", "url") if c in dataset.column_names), None)
    status = re.compile(r"(\d+)$" if id_column == "tweet_id" else r"status/(\d+)")

    def mark(x):
        match = status.search(str(x[id_column] or "")) if id_column else None
        return {"heldout": bool(match) and match.group(1) in heldout_ids}

    text_column = source.get("text_column", "text")
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    return dataset.map(mark, num_proc=os.cpu_count()).select_columns(["text", "heldout"])


def load_pretraining_data(
    sources: list[dict] | None = None, val_size: int = 10_000, stats: dict | None = None
) -> tuple:
    """
    Loads the pretraining tweets, without the tweets of the frozen test set.

    Parameters
    ----------
    sources : list[dict], optional
        Datasets to pretrain on, each {"name": Hugging Face id or local .csv/.jsonl/
        .parquet file, "text_column": default "text", "repeat": default 1}.
        Near-duplicates across sources are kept once, in the earliest source listed, so
        put curated sources first. "repeat" upweights a source by showing its training
        tweets that many times (with new masks each time); validation is not repeated.
        By default only StephanAkkerman/crypto-stock-tweets.
    val_size : int, optional
        Number of tweets held out to track the MLM loss, by default 10,000.
    stats : dict, optional
        Filled with the tweets per source after cleaning ("tweets_per_source") and the
        number of training examples after repeats ("train_examples"), for the model card.
        A small fixed set is enough; everything else is used for training.

    Returns
    -------
    tuple
        The training and validation datasets (validation is None if val_size is 0).
    """
    sources = sources or DEFAULT_PRETRAIN_DATASETS

    # Cleaning 8M tweets takes a while, so reuse the result on a rerun or resume
    parts = [load_pretraining_source(source) for source in sources]
    version = hashlib.blake2b(digest_size=4)
    for part in parts:
        version.update(part._fingerprint.encode())
    for path in (Path(__file__), TEST_SET_PATH, HELDOUT_IDS_PATH, DEV_SET_PATH, DEV_IDS_PATH):
        version.update(path.read_bytes())
    cache = PRETRAIN_CACHE / version.hexdigest()

    if cache.exists():
        dataset = Dataset.load_from_disk(str(cache))
    else:
        dataset = concatenate_datasets(
            [part.add_column("source", [i] * len(part)) for i, part in enumerate(parts)]
        )

        # A quoted tweet becomes a pretraining tweet of its own (dedup drops it if the
        # data has it already), so each example is one author's words
        def split(batch):
            rows = [
                (part, source, heldout)
                for text, source, heldout in zip(
                    batch["text"], batch["source"], batch["heldout"]
                )
                for part in (split_quote(text) if text else [text])
            ]
            return dict(zip(["text", "source", "heldout"], map(list, zip(*rows))))

        dataset = dataset.map(split, batched=True, num_proc=os.cpu_count())

        # The test tweets come from the same feed export, so keep them out of MLM: by id,
        # and by text for copies in sources without ids
        heldout = dataset.filter(lambda x: x["heldout"])
        test_keys = test_set_keys() | {dedup_key(t) for t in heldout["text"] if t}
        dataset = dataset.filter(
            lambda x: not x["heldout"]
            and x["text"] is not None
            and dedup_key(x["text"]) not in test_keys,
            num_proc=os.cpu_count(),
        ).remove_columns("heldout")

        dataset = clean_pretraining_tweets(dataset)
        dataset.save_to_disk(str(cache))

    counts = np.bincount(dataset["source"], minlength=len(sources))
    for source, count in zip(sources, counts):
        print(f"{source['name']}: {count:,} tweets after cleaning")

    if val_size:
        split = dataset.train_test_split(test_size=val_size, seed=42)
        train, val = split["train"], split["test"].remove_columns("source")
    else:
        train, val = dataset, None

    # Upweight sources by repeating their training tweets
    sources_column = np.asarray(train["source"])
    extra = [
        train.select(np.flatnonzero(sources_column == i))
        for i, source in enumerate(sources)
        for _ in range(source.get("repeat", 1) - 1)
    ]
    train = concatenate_datasets([train, *extra]).remove_columns("source")
    if stats is not None:
        stats["tweets_per_source"] = [int(count) for count in counts]
        stats["train_examples"] = len(train)
    return train, val


def pretrain_dedup_key(text: str) -> str:
    """
    `dedup_key` with numbers masked, so bot templates such as price and whale alerts
    ("BTC is $27,104 (+1.2%)") collapse into one tweet.
    """
    return re.sub(r"\d+", "0", dedup_key(text))


TAG = re.compile(r"^([#$]\S+|@USER|\[URL\])$")


def tag_share(text: str) -> float:
    """Share of words that are hashtags, cashtags, mentions or links."""
    words = text.split()
    return sum(bool(TAG.match(word)) for word in words) / max(1, len(words))


def clean_pretraining_tweets(dataset: Dataset) -> Dataset:
    """
    Like `preprocess_dataset`, but in Arrow instead of pandas (pandas needs more than
    15 GB of RAM for the 8M pretraining tweets) and stricter about bot and tag spam.
    """
    dataset = dataset.filter(lambda x: x["text"] is not None, num_proc=os.cpu_count())

    def clean(batch):
        texts = [preprocess_tweet(text) for text in batch["text"]]
        # A stable hash of the near-duplicate key (Python's hash() differs per process)
        keys = [
            int.from_bytes(
                hashlib.blake2b(pretrain_dedup_key(text).encode(), digest_size=8).digest(),
                signed=True,
            )
            for text in texts
        ]
        return {"text": texts, "key": keys, "source": batch["source"]}

    dataset = dataset.map(
        clean,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )

    # Keep the first tweet of each near-duplicate group
    _, first = np.unique(np.asarray(dataset["key"], dtype=np.int64), return_index=True)
    dataset = dataset.select(np.sort(first)).remove_columns("key")

    # Drop 1 word tweets and tweets that are mostly hashtags, cashtags, mentions and links
    return dataset.filter(
        lambda x: len(x["text"].split()) > 1 and tag_share(x["text"]) <= 0.5,
        num_proc=os.cpu_count(),
    )


def load_extra_training(extra: list[dict]) -> Dataset:
    """
    Labeled tweets from local CSV files (columns text and label), each
    {"path": "data/train/feed_v1.csv", "repeat": default 1}, without dev or test tweets.
    """
    parts = []
    for source in extra:
        dataframe = pd.read_csv(REPO_ROOT / source["path"])[["text", "label"]]
        unknown = set(dataframe["label"]) - set(LABELS)
        if unknown:
            raise ValueError(f"Unknown labels in {source['path']}: {unknown}")
        dataframe["label"] = dataframe["label"].map(LABELS.index)
        dataframe = dataframe[~dataframe["text"].map(dedup_key).isin(test_set_keys())]
        # Repeat after preprocessing, which drops duplicates
        dataset = preprocess_dataset(Dataset.from_pandas(dataframe, preserve_index=False))
        print(f"{source['path']}: {len(dataset):,} tweets x{source.get('repeat', 1)}")
        parts += [dataset] * source.get("repeat", 1)
    return concatenate_datasets(parts).cast_column("label", ClassLabel(names=LABELS))


def load_finetuning_data(
    val_size: float = 0.1,
    drop_sources: list[str] | None = None,
    relabel: bool = True,
    extra_train: list[dict] | None = None,
) -> tuple:
    """
    Loads and cleans the finetuning data and splits it into a training and validation set.
    Testing is done on the frozen test set instead, see `load_test_data()`.

    Parameters
    ----------
    val_size : float, optional
        The size of the validation set, by default 0.1
    drop_sources : list[str], optional
        Sources to leave out, by their short name in `SOURCES`, e.g. ["mattgilgo"]
    relabel : bool, optional
        Replace labels with the guideline relabels in `RELABELS_PATH`, by default True
    extra_train : list[dict], optional
        Extra labeled CSV files added to the training split only, see
        `load_extra_training`, e.g. [{"path": "data/train/feed_v1.csv"}]. The validation
        split stays the same, so runs with and without them are compared on equal terms.

    Returns
    -------
    tuple
        The training and validation datasets.
    """

    # https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment
    # 0: neutral, 1: bullish, 2: bearish
    dataframe = load_dataset(
        "TimKoornstra/financial-tweets-sentiment",
        split="train",
        cache_dir="datasets",
    ).to_pandas()

    dataframe = dataframe.rename(columns={"tweet": "text", "sentiment": "label"})
    dataframe["source"] = dataframe["url"].map(
        lambda url: next(v for k, v in SOURCES.items() if k in url.lower())
    )
    if drop_sources:
        unknown = set(drop_sources) - set(SOURCES.values())
        if unknown:
            raise ValueError(f"Unknown sources: {unknown}")
        dataframe = dataframe[~dataframe["source"].isin(drop_sources)]

    # Near-duplicates with conflicting labels are noise, drop every copy
    dataframe["key"] = dataframe["text"].map(dedup_key)
    conflicting = dataframe.groupby("key")["label"].transform("nunique") > 1
    dataframe = dataframe[~conflicting].drop_duplicates(subset=["key"])

    if relabel:
        relabels = pd.read_csv(RELABELS_PATH)
        new_label = dict(
            zip(relabels["tweet"].map(dedup_key), relabels["label"].map(LABELS.index))
        )
        dataframe["label"] = dataframe["key"].map(new_label).fillna(dataframe["label"]).astype(int)

    # Never train on tweets from the test set
    dataframe = dataframe[~dataframe["key"].isin(test_set_keys())]

    dataset = preprocess_dataset(Dataset.from_pandas(dataframe, preserve_index=False))
    split = dataset.cast_column("label", ClassLabel(names=LABELS)).train_test_split(
        test_size=val_size, seed=42, stratify_by_column="label"
    )
    train = split["train"]
    if extra_train:
        extra = load_extra_training(extra_train).cast(train.features)
        train = concatenate_datasets([train, extra]).shuffle(seed=42)
    return train, split["test"]


def simple_oversample(dataset: Dataset) -> pd.DataFrame:
    from imblearn.over_sampling import RandomOverSampler

    dataframe = dataset.to_pandas()

    # Extract texts and labels
    texts = dataframe["text"].tolist()
    labels = dataframe["label"].tolist()

    # Define RandomOverSampler
    ros = RandomOverSampler(random_state=0)

    # Resample indices and labels
    resampled_indices, resampled_labels = ros.fit_resample(
        np.array(list(range(len(texts)))).reshape(-1, 1), labels
    )

    # Extract resampled texts based on the resampled indices
    resampled_texts = [texts[i[0]] for i in resampled_indices]

    # Convert back to dataframe
    dataframe = pd.DataFrame({"text": resampled_texts, "label": resampled_labels})

    # To dataset
    return Dataset.from_pandas(dataframe, preserve_index=False)


def synonym_oversample(dataset: Dataset) -> pd.DataFrame:
    import nlpaug.augmenter.word as naw

    dataframe = dataset.to_pandas()

    # Extract texts and labels
    texts = [example["text"] for example in dataset]
    labels = [example["label"] for example in dataset]

    # Determine class frequencies and find the maximum frequency
    class_counts = Counter(labels)
    max_count = max(class_counts.values())

    # Initialize nlpaug synonym augmenter
    augmenter = naw.SynonymAug(aug_src="wordnet")

    # Augment data for each class to match the count of the majority class
    augmented_texts = []
    augmented_labels = []
    for label in class_counts.keys():
        class_texts = [text for text, lbl in zip(texts, labels) if lbl == label]
        augment_count = max_count - class_counts[label]

        # Randomly choose and augment texts from the class until reaching the desired count
        while augment_count > 0:
            for text in random.choices(
                class_texts, k=min(augment_count, len(class_texts))
            ):
                augmented_text = augmenter.augment(text)
                augmented_texts.append(augmented_text[0])
                augmented_labels.append(label)
                augment_count -= 1

    # Combine original and augmented data
    resampled_texts = texts + augmented_texts
    resampled_labels = labels + augmented_labels

    # Convert back to dataframe
    dataframe = pd.DataFrame({"text": resampled_texts, "label": resampled_labels})

    # To dataset
    return Dataset.from_pandas(dataframe, preserve_index=False)


def preprocess_dataset(dataset: Dataset) -> Dataset:
    # Convert to pandas
    dataframe = dataset.to_pandas()

    # Set labels to int
    if "label" in dataframe.columns:
        dataframe["label"] = dataframe["label"].astype(int)

        # Drop all columns that are not text or label
        dataframe = dataframe[["text", "label"]]
    else:
        dataframe = dataframe[["text"]]

    # Preprocess tweets
    dataframe["text"] = dataframe["text"].apply(preprocess_tweet)

    # Drop near-duplicates (same tweet with another link or mention), common in bot tweets
    dataframe = dataframe[~dataframe["text"].map(dedup_key).duplicated()]

    # Drop empty text tweets
    dataframe = dataframe.dropna(subset=["text"])

    # Drop 1 word tweets
    dataframe = dataframe[dataframe["text"].apply(lambda x: len(x.split()) > 1)]

    # Convert dataframe back to dataset
    return Dataset.from_pandas(dataframe, preserve_index=False)


def adjust_labels(dataset):
    # Original labels: 0: negative, 1: neutral, 2: positive
    # New labels: 0: neutral, 1: bullish, 2: bearish
    label_mapping = {
        0: 2,  # negative to bearish
        1: 0,  # neutral to neutral
        2: 1,  # positive to bullish
    }
    dataset["label"] = label_mapping[dataset["label"]]
    return dataset


def load_tweet_eval():
    # 0: negative, 1: neutral, 2: positive
    # https://huggingface.co/datasets/tweet_eval/viewer/sentiment
    dataset = load_dataset(
        "tweet_eval",
        cache_dir="data/pre-finetune/",
        name="sentiment",
    )

    # Concatenate the splits into a single dataset
    dataset = concatenate_datasets(
        [dataset["train"], dataset["test"], dataset["validation"]]
    )

    # Change labels to match other datasets
    dataset = dataset.map(adjust_labels)

    dataframe = preprocess_dataset(dataset)
    return split_dataset(dataframe)


def kfold_pretraining_data(k: int = 5) -> tuple:
    """
    Split the pretraining data into k folds.

    Parameters
    ----------
    k : int, optional
        The number of folds, by default 5

    Returns
    -------
    tuple
        The training and validation datasets.
    """
    from sklearn.model_selection import KFold

    training_datasets = []
    validation_datasets = []

    df = load_pretraining_data(val_size=0)[0].to_pandas()

    # Assuming 'dataset' is a list or array of your data
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # This will yield indices for 5 splits
    for train_index, val_index in kf.split(df):
        training_datasets.append(Dataset.from_pandas(df.iloc[train_index]))
        validation_datasets.append(Dataset.from_pandas(df.iloc[val_index]))

    return training_datasets, validation_datasets


def load_test_data() -> Dataset:
    """
    Loads the frozen test set (labels: 0: neutral, 1: bullish, 2: bearish).
    The `file` column holds its source (fintwit_crypto, fintwit_stock, fintwit_other).

    Returns
    -------
    Dataset
        The test dataset, with the same preprocessing as the training data.
    """
    dataframe = pd.read_csv(TEST_SET_PATH)
    dataframe["label"] = dataframe["label"].map(LABELS.index)
    dataframe["text"] = dataframe["text"].apply(preprocess_tweet)
    return Dataset.from_pandas(dataframe[["text", "label", "file"]])


def load_dev_data() -> Dataset:
    """
    Loads the dev set, with the same labels and preprocessing as the test set. The `file`
    column says how a tweet was sampled: "random", or "bearish_keywords" for the extra
    tweets picked by bearish words because a random sample had too few bearish ones.
    """
    dataframe = pd.read_csv(DEV_SET_PATH).rename(columns={"sample": "file"})
    dataframe["label"] = dataframe["label"].map(LABELS.index)
    dataframe["text"] = dataframe["text"].apply(preprocess_tweet)
    return Dataset.from_pandas(dataframe[["text", "label", "file"]])
