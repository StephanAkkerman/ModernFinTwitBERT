import numpy as np
from datasets import load_dataset
from huggingface_hub import HfFolder
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

tokenizer = AutoTokenizer.from_pretrained(
    "answerdotai/ModernBERT-base", cache_dir="models"
)

# Load the dataset
dataset = load_dataset(
    "TimKoornstra/financial-tweets-sentiment", cache_dir="datsets", split="train"
)

split_dataset = dataset.train_test_split(test_size=0.1)
print(split_dataset["train"][0])


# Tokenize helper function
def tokenize(batch):
    return tokenizer(
        batch["tweet"], padding="max_length", truncation=True, return_tensors="pt"
    )


# Tokenize dataset
if "sentiment" in split_dataset["train"].features.keys():
    split_dataset = split_dataset.rename_column(
        "sentiment", "labels"
    )  # to match Trainer
tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["tweet"])

print(tokenized_dataset["train"].features.keys())

# Prepare model labels - useful for inference
labels = tokenized_dataset["train"].features["labels"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

    # Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
        labels, predictions, labels=labels, pos_label=1, average="weighted"
    )
    return {"f1": float(score) if score == 1 else score}


# Define training args
training_args = TrainingArguments(
    output_dir="ModernFinTwitBERT",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=5,
    bf16=True,  # bfloat16 training
    optim="adamw_torch_fused",  # improved optimizer
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # use_mps_device=True,
    metric_for_best_model="f1",
    # push to hub parameters
    # push_to_hub=True,
    # hub_strategy="every_save",
    # hub_token=HfFolder.get_token(),
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
