import json

import matplotlib.pyplot as plt
import seaborn as sns
from data import load_finetuning_data
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

import wandb
from datasets import load_dataset


class Evaluate:
    def __init__(self, use_baseline: bool = False, baseline_model: int = 0):
        # Load config
        with open("config.json", "r") as config_file:
            self.config = json.load(config_file)
        self.model_name = self.config["model_name"]

        if not use_baseline:
            labels = ["NEUTRAL", "BULLISH", "BEARISH"]
            self.model = AutoModelForSequenceClassification.from_pretrained(
                f"output/{self.model_name}",
                num_labels=len(labels),
                id2label={k: v for k, v in enumerate(labels)},
                label2id={v: k for k, v in enumerate(labels)},
                attn_implementation="flash_attention_2",
                device_map="cuda",
                torch_dtype="auto",
                cache_dir="models",
            )
            self.model.config.problem_type = "single_label_classification"
            self.tokenizer = AutoTokenizer.from_pretrained(f"output/{self.model_name}")
        else:
            if baseline_model == 0:
                model_name = "StephanAkkerman/FinTwitBERT-sentiment"
            elif baseline_model == 1:
                model_name = "ProsusAI/finbert"
            elif baseline_model == 2:
                model_name = "yiyanghkust/finbert-tone"

            self.model = BertForSequenceClassification.from_pretrained(
                model_name,
                cache_dir="models",
                device_map="cuda",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir="models",
            )

        self.model.eval()
        # https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
        )

    def encode(self, batch):
        return self.tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=512
        )

    def load_test_data(self, tokenize: bool = True):
        dataset = load_dataset(
            "financial_phrasebank",
            cache_dir="data/finetune/",
            split="train",
            name="sentences_50agree",
        )

        # Rename sentence to text
        dataset = dataset.rename_column("sentence", "text")

        if tokenize:
            # Apply the tokenize function to the dataset
            tokenized_dataset = dataset.map(self.encode, batched=True)

            # Set the format for pytorch tensors
            tokenized_dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"],
            )

            return tokenized_dataset
        return dataset

    def evaluate_model(self):
        # Create a confusion matrix for the finetuning dataset on the evaluation set
        true_labels, pred_labels = self.get_labels(category="eval")
        self.plot_confusion_matrix("eval", true_labels, pred_labels)

        true_labels, pred_labels = self.get_labels(category="test")
        self.calculate_metrics(true_labels, pred_labels)
        self.plot_confusion_matrix("test", true_labels, pred_labels)

    def get_labels(self, category: str, batch_size: int = 32):

        if category == "test":
            _, _, dataset = load_finetuning_data()
            # 0: neutral, 1: bullish, 2: bearish
            int2str = {0: "neutral", 1: "bullish", 2: "bearish"}
            true_labels = [int2str[label] for label in dataset["label"]]

            # Convert numerical labels to textual labels
            # dataset = self.load_test_data()
            # true_labels = [
            #     dataset.features["label"].int2str(label) for label in dataset["label"]
            # ]

        elif category == "eval":
            _, dataset, _ = load_finetuning_data()

            # 0: neutral, 1: bullish, 2: bearish
            int2str = {0: "neutral", 1: "bullish", 2: "bearish"}
            true_labels = [int2str[label] for label in dataset["label"]]
        else:
            raise ValueError("Invalid category name")

        pred_labels = []
        for out in self.pipeline(KeyDataset(dataset, "text"), batch_size=batch_size):
            pred_labels.append(out["label"].lower())

        # Convert bullish to positive and bearish to negative
        # if category == "test":
        #     label_mapping = {"bullish": "positive", "bearish": "negative"}
        #     pred_labels = [label_mapping.get(label, label) for label in pred_labels]

        return true_labels, pred_labels

    def calculate_metrics(self, true_labels, pred_labels):
        # Compute accuracy and F1 score
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="weighted")

        # Log metrics to wandb
        output = {
            "test/final_accuracy": accuracy,
            "test/final_f1_score": f1,
        }

        if wandb.run is not None:
            wandb.log(output)

        print(output)

    def plot_confusion_matrix(self, category: str, true_labels, pred_labels):
        # Create confusion matrix
        label_encoder = LabelEncoder()
        true_labels_encoded = label_encoder.fit_transform(true_labels)
        pred_labels_encoded = label_encoder.transform(pred_labels)

        cm = confusion_matrix(true_labels_encoded, pred_labels_encoded)
        labels = label_encoder.classes_

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            ax=ax,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")

        # Log confusion matrix to wandb
        if wandb.run is not None:
            wandb.log({f"{category}/confusion_matrix": wandb.Image(fig)})

        # Close the plot
        plt.close(fig)


if __name__ == "__main__":
    eval = Evaluate(use_baseline=True, baseline_model=2)
    eval.evaluate_model()
