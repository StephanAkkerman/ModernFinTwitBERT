import argparse
import json
import os

import numpy as np
import torch
from data import load_finetuning_data, load_pretraining_data
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# flash-attn only works on GPU; fall back to PyTorch attention elsewhere
DEVICE_ARGS = (
    {"attn_implementation": "flash_attention_2", "device_map": "cuda"}
    if torch.cuda.is_available()
    else {"attn_implementation": "sdpa"}
)


def load_pretrained(auto_class, name: str, **kwargs):
    """Load with flash-attn where the architecture supports it, else the default attention."""
    try:
        return auto_class.from_pretrained(name, **kwargs, **DEVICE_ARGS)
    except (ValueError, ImportError):
        device_map = "cuda" if torch.cuda.is_available() else None
        return auto_class.from_pretrained(name, **kwargs, device_map=device_map)


def compute_metrics(eval_pred):
    """Macro F1 as the headline (the data is bullish-heavy), plus bullish/bearish precision."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision = precision_score(
        labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "f1": f1_score(labels, predictions, labels=[0, 1, 2], average="macro"),
        "precision_bullish": precision[1],
        "precision_bearish": precision[2],
    }


class ModernFinTwitBERT:
    def __init__(self, max_steps: int | None = None):
        # Read model args from config.json
        with open("config.json", "r") as config_file:
            self.config = json.load(config_file)

        # Set the mode (finetune or pretrain)
        self.mode = self.config["mode"]

        self.hyperopt = self.config[self.mode]["do_hyperopt"]

        # Mode-specific args override the shared base args
        self.mode_args = {
            **self.config["base_args"],
            **self.config[self.mode][f"{self.mode}_args"],
        }

        self.model_name = self.config[self.mode]["model_name"]

        # Checkpoints per model, so a crashed or interrupted run can resume
        self.mode_args.setdefault("output_dir", f"output/checkpoints/{self.model_name}")
        if max_steps:
            # Short trial run, e.g. to measure speed and memory on a new GPU
            self.mode_args["max_steps"] = max_steps
            self.mode_args["save_strategy"] = "no"
            self.mode_args["load_best_model_at_end"] = False

        self.labels = ["NEUTRAL", "BULLISH", "BEARISH"]

        # Load the model
        if self.mode == "pretrain":
            self.model = load_pretrained(
                AutoModelForMaskedLM,
                self.config[self.mode]["base_model"],
                dtype=torch.float32,  # bf16 autocast comes from the bf16 training arg
                cache_dir="models",
            )
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config[self.mode]["base_model"], cache_dir="models"
            )

            # New tokens for the @USER and [URL] placeholders start from random
            # embeddings, so only add them when the base model has no good split for them
            special_tokens = self.config[self.mode].get("add_tokens", [])
            if special_tokens:
                self.tokenizer.add_tokens(special_tokens)
                # Keep the vocab a multiple of 64 for faster matmuls
                self.model.resize_token_embeddings(
                    len(self.tokenizer), pad_to_multiple_of=64
                )

            # Load the data
            self.train, self.val = load_pretraining_data()

        elif self.mode == "finetune":
            # Need to update for our own pretrained model
            self.model = self.model_init()

            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config[self.mode]["base_model"], cache_dir="models"
            )

            # Load the data
            self.train, self.val = load_finetuning_data(
                drop_sources=self.config[self.mode].get("drop_sources"),
                relabel=self.config[self.mode].get("relabel", True),
            )

        self.max_length = self.config[self.mode].get("max_length", 128)
        self.output_dir = f"output/{self.model_name}"

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
        os.environ["WANDB_PROJECT"] = self.model_name

        # Upload only the final model; uploading every checkpoint slows training down
        os.environ["WANDB_LOG_MODEL"] = "end"

        # turn off watch to log faster
        os.environ["WANDB_WATCH"] = "false"

    def encode(self, batch):
        # Tweets are short: truncate at max_length and pad per batch in the collator
        return self.tokenizer(
            batch["text"], truncation=True, max_length=self.max_length
        )

    def trainer(self):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Encode the data
        # changing batch_size does not seem to change the speed
        train = self.train.map(
            self.encode,
            batched=True,
            remove_columns=["text"],
            batch_size=5_000,
            num_proc=os.cpu_count(),
        )
        val = self.val.map(
            self.encode,
            batched=True,
            remove_columns=["text"],
            batch_size=5_000,
            num_proc=os.cpu_count(),
        )

        # Use the MLM data collator when pretraining
        if self.mode == "pretrain":
            # Match the base model's own masking rate (RoBERTa 15%, ModernBERT 30%)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=self.config[self.mode].get("mlm_probability", 0.15),
            )

        # Compute F1 and accuracy scores when finetuning
        compute_metrics_fn = (
            compute_metrics if self.mode in ["finetune", "pre-finetune"] else None
        )

        if self.hyperopt:
            # https://huggingface.co/docs/transformers/en/hpo_train
            # https://huggingface.co/docs/setfit/how_to/hyperparameter_optimization
            # Create a Trainer instance
            os.environ["WANDB_DISABLED"] = "true"
            os.environ["WANDB_MODE"] = "disabled"

            # Remove "report_to": "wandb" from args
            self.mode_args.pop("report_to", None)

            trainer = Trainer(
                model=None,
                args=TrainingArguments(**self.mode_args),
                train_dataset=train,
                eval_dataset=val,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
                model_init=self.model_init,
            )

            best_run = trainer.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                n_trials=self.config[self.mode].get("hyperopt_trials", 20),
                hp_space=optuna_hp_space,
                # The default objective is the sum of all metrics, not macro F1
                compute_objective=lambda metrics: metrics["eval_f1"],
            )

            print(best_run)
            print(best_run.hyperparameters)

            merged_args = {**self.mode_args, **best_run.hyperparameters}

            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**merged_args),
                train_dataset=train,
                eval_dataset=val,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
            )
        else:
            # Create a Trainer instance, without model_init
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**self.mode_args),
                train_dataset=train,
                eval_dataset=val,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
            )

        checkpoint = get_last_checkpoint(self.mode_args["output_dir"])
        if checkpoint and not self.mode_args.get("max_steps"):
            print(f"Resuming from {checkpoint}")
        else:
            checkpoint = None
            print("Starting training")

        trainer.train(resume_from_checkpoint=checkpoint)

        print("Finished training")

        if self.mode_args.get("max_steps"):
            print("Trial run finished, not saving the model")
            return

        # Save the model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print("Saved model and tokenizer")
        return

    def model_init(self, trial=None):
        model = load_pretrained(
            AutoModelForSequenceClassification,
            self.config[self.mode]["base_model"],
            num_labels=len(self.labels),
            id2label={k: v for k, v in enumerate(self.labels)},
            label2id={v: k for k, v in enumerate(self.labels)},
            dtype=torch.float32,  # bf16 autocast comes from the bf16 training arg
            cache_dir="models",
        )
        model.config.problem_type = "single_label_classification"
        return model


def optuna_hp_space(trial):

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128, 256]
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"]
        ),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain or fine-tune, see config.json")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop after this many steps without saving, to test speed and memory",
    )
    ModernFinTwitBERT(parser.parse_args().max_steps).trainer()
