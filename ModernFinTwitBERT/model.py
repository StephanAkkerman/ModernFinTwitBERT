import json
import os

import numpy as np
from data import load_finetuning_data, load_pretraining_data
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    ModernBertForMaskedLM,
    Trainer,
    TrainingArguments,
)


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

        # Set the mode (finetune or pretrain)
        self.mode = self.config["mode"]

        self.hyperopt = self.config[self.mode]["do_hyperopt"]

        # Get the mode-specific args
        self.mode_args = self.config[self.mode][f"{self.mode}_args"]

        self.model_name = self.config[self.mode]["model_name"]

        self.labels = ["NEUTRAL", "BULLISH", "BEARISH"]

        # Load the model
        if self.mode == "pretrain":
            self.model = ModernBertForMaskedLM.from_pretrained(
                self.config[self.mode]["base_model"],
                num_labels=len(self.labels),
                id2label={k: v for k, v in enumerate(self.labels)},
                label2id={v: k for k, v in enumerate(self.labels)},
                attn_implementation="flash_attention_2",
                device_map="cuda",
                torch_dtype="auto",
                cache_dir="models",
            )
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config[self.mode]["base_model"], cache_dir="models"
            )

            # Add special tokens
            special_tokens = ["@USER", "[URL]"]
            self.tokenizer.add_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))

            # Load the data
            self.train, self.val, _ = load_pretraining_data()

        elif self.mode == "finetune":
            # Need to update for our own pretrained model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config[self.mode]["base_model"],
                num_labels=len(self.labels),
                id2label={k: v for k, v in enumerate(self.labels)},
                label2id={v: k for k, v in enumerate(self.labels)},
                attn_implementation="flash_attention_2",
                device_map="cuda",
                torch_dtype="auto",
                cache_dir="models",
            )
            self.model.config.problem_type = "single_label_classification"

            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config[self.mode]["base_model"], cache_dir="models"
            )

            # Load the data
            self.train, self.val, _ = load_finetuning_data()

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

        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

        # turn off watch to log faster
        os.environ["WANDB_WATCH"] = "false"

    def encode(self, batch):
        return self.tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=512
        )

    def trainer(self):
        data_collator = None

        # Encode the data
        # changing batch_size does not seem to change the speed
        train = self.train.map(
            self.encode,
            batched=True,
            remove_columns=["text"],
            batch_size=5_000,
            # num_proc=4,
        )
        val = self.val.map(
            self.encode,
            batched=True,
            remove_columns=["text"],
            batch_size=5_000,
            # num_proc=4,
        )

        # Use the MLM data collator when pretraining
        if self.mode == "pretrain":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm_probability=0.15
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
            self.config["base_args"].pop("report_to", None)

            trainer = Trainer(
                model=None,
                args=TrainingArguments(**self.mode_args, **self.config["base_args"]),
                train_dataset=train,
                eval_dataset=val,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
                model_init=self.model_init,
            )

            best_run = trainer.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                n_trials=10,
                hp_space=optuna_hp_space,
            )

            # Apply the best hyperparameters and train one final time
            trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)

        else:
            # Create a Trainer instance, without model_init
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**self.mode_args, **self.config["base_args"]),
                train_dataset=train,
                eval_dataset=val,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
            )

        print("Starting training")

        trainer.train()

        print("Finished training")

        # Save the model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print("Saved model and tokenizer")
        return

    def model_init(self, trial):

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config[self.mode]["base_model"],
            num_labels=len(self.labels),
            id2label={k: v for k, v in enumerate(self.labels)},
            label2id={v: k for k, v in enumerate(self.labels)},
            attn_implementation="flash_attention_2",
            device_map="cuda",
            torch_dtype="auto",
            cache_dir="models",
        )
        model.config.problem_type = "single_label_classification"
        return model


def optuna_hp_space(trial):

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128]
        ),
    }


if __name__ == "__main__":
    ModernFinTwitBERT().trainer()
