import json
import os

import numpy as np
from data import load_finetuning_data
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
        labels, predictions, labels=labels, pos_label=1, average="weighted"
    )
    return {"f1": float(score) if score == 1 else score}


class ModernFinTwitBERT:
    def __init__(self):
        # Read model args from config.json
        with open("config.json", "r") as config_file:
            self.config = json.load(config_file)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model"], cache_dir="models"
        )

        labels = ["NEUTRAL", "BULLISH", "BEARISH"]

        # Load the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["base_model"],
            num_labels=len(labels),
            id2label={k: v for k, v in enumerate(labels)},
            label2id={v: k for k, v in enumerate(labels)},
            attn_implementation="flash_attention_2",
            device_map="cuda",
            torch_dtype="auto",
            cache_dir="models",
        )
        self.model.config.problem_type = "single_label_classification"
        self.output_dir = "output/ModernFinTwitBERT"

        self.init_wandb()

    def init_wandb(self):
        # Check if a .env file exists
        if not os.path.exists("wandb.env"):
            print("No wandb.env file found")
            return

        # Load the .env file
        load_dotenv(dotenv_path="wandb.env")

        # Read the API key from the environment variable
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

        # set the wandb project where this run will be logged
        os.environ["WANDB_PROJECT"] = self.output_dir.split("/")[-1]

        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"] = "true"

        # turn off watch to log faster
        os.environ["WANDB_WATCH"] = "false"

    def encode(self, batch):
        return self.tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=512
        )

    def train(self):
        # Load the dataset
        train, val, _ = load_finetuning_data()

        train = train.map(self.encode, batched=True, remove_columns=["text"])
        val = val.map(self.encode, batched=True, remove_columns=["text"])

        # Define training args
        training_args = TrainingArguments(
            output_dir="checkpoints/",
            overwrite_output_dir=True,
            per_device_train_batch_size=256,
            per_device_eval_batch_size=256,
            learning_rate=5e-5,
            num_train_epochs=2,
            bf16=True,  # bfloat16 training
            optim="adamw_torch_fused",  # improved optimizer
            # logging & evaluation strategies
            logging_strategy="steps",
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="wandb",
            save_safetensors=True,
            # push to hub parameters
            # push_to_hub=True,
            # hub_strategy="every_save",
            # hub_token=HfFolder.get_token(),
        )

        # Create a Trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # Save the model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


if __name__ == "__main__":
    ModernFinTwitBERT().train()
