import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data import LABELS, load_test_data
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

import wandb

BASELINES = {
    "fintwitbert": "StephanAkkerman/FinTwitBERT-sentiment",
    "finbert": "ProsusAI/finbert",
    "finbert-tone": "yiyanghkust/finbert-tone",
}
# Map other label names onto ours (e.g. finbert's positive/negative)
ALIASES = {"positive": "bullish", "negative": "bearish"}


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    """Gap between confidence and accuracy, averaged over equal-width confidence bins."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


class Evaluate:
    def __init__(self, model_name: str | None = None):
        # Defaults to our own finetuned model
        if model_name is None:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
            model_name = f"output/{config['finetune']['model_name']}"
        self.model_name = BASELINES.get(model_name, model_name)

        # https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
        self.pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            truncation=True,
            max_length=128,
        )

    def predict(self, batch_size: int = 64) -> pd.DataFrame:
        dataset = load_test_data()
        df = dataset.to_pandas()
        df["label"] = df["label"].map(dict(enumerate(LABELS)))

        outputs = list(self.pipeline(KeyDataset(dataset, "text"), batch_size=batch_size))
        df["pred"] = [ALIASES.get(o["label"].lower(), o["label"].lower()) for o in outputs]
        df["confidence"] = [o["score"] for o in outputs]
        return df

    def evaluate_model(self) -> dict:
        df = self.predict()
        correct = (df["label"] == df["pred"]).to_numpy()
        report = classification_report(
            df["label"], df["pred"], labels=LABELS, output_dict=True, zero_division=0
        )
        output = {
            "test/macro_f1": f1_score(df["label"], df["pred"], labels=LABELS, average="macro"),
            "test/accuracy": correct.mean(),
            "test/precision_bullish": report["bullish"]["precision"],
            "test/precision_bearish": report["bearish"]["precision"],
            "test/ece": expected_calibration_error(df["confidence"].to_numpy(), correct),
        }
        for source, acc in df.assign(ok=correct).groupby("file")["ok"].mean().items():
            output[f"test/accuracy_{source}"] = acc

        print(f"model: {self.model_name}  rows: {len(df)}")
        print(classification_report(df["label"], df["pred"], labels=LABELS, digits=3, zero_division=0))
        print(json.dumps({k: round(float(v), 4) for k, v in output.items()}, indent=2))

        if wandb.run is not None:
            wandb.log(output)
            self.plot_confusion_matrix("test", df["label"], df["pred"])
        return output

    def plot_confusion_matrix(self, category: str, true_labels, pred_labels):
        cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            ax=ax,
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")

        # Log confusion matrix to wandb
        wandb.log({f"{category}/confusion_matrix": wandb.Image(fig)})

        # Close the plot
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a model on the frozen test set")
    parser.add_argument(
        "model",
        nargs="?",
        help=f"Model path or HF id, or a baseline: {', '.join(BASELINES)}. "
        "Defaults to output/<finetune model_name>.",
    )
    Evaluate(parser.parse_args().model).evaluate_model()
