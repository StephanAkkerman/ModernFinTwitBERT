"""
Export the sentiment model to ONNX with int8 weights, for fast CPU inference (bot, Raspberry
Pi). Dynamic quantization, per-channel, of all MatMul weights and the embedding table:
about 4x smaller and 4x faster on CPU than the fp32 model, with the same dev and test F1
(measured 2026-10-29). Per-tensor quantization lost 0.01-0.02 F1, so keep per-channel.

The script checks the export against the PyTorch model on the dev and test sets (macro F1
and how often both give the same label) through onnx_predict.OnnxSentiment, so the
preprocessing and tokenizer of the predictor are checked too.

Run from the repository root:

    python ModernFinTwitBERT/export_onnx.py
    python ModernFinTwitBERT/export_onnx.py --model output/experiments/models/llrd-0.9-feed-seed42
    python ModernFinTwitBERT/export_onnx.py --hub-repo StephanAkkerman/FinTwitRoBERTa-sentiment

Writes output/onnx/<model>/ (onnx/model_int8.onnx, config.json, tokenizer.json, report.csv
and README.md, the model card with an "ONNX int8" section on size, speed and scores, replacing
an earlier one). --hub-repo uploads onnx/model_int8.onnx and that card to the Hugging Face repo.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from onnx_predict import ONNX_FILE, OnnxSentiment
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from thresholds import apply_thresholds, metrics, predict_probs
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_dev_data, load_test_data

LATENCY_TWEETS = 100


def export(model, tokenizer, folder: Path) -> Path:
    fp32 = folder / "model_fp32.onnx"
    inputs = tokenizer(["$AAPL breaking out"], return_tensors="pt")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    prepared = folder / "model_prepared.onnx"
    quant_pre_process(str(fp32), str(prepared), skip_symbolic_shape=True)
    int8 = folder / ONNX_FILE
    int8.parent.mkdir(exist_ok=True)
    quantize_dynamic(
        str(prepared),
        str(int8),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gather"],
    )
    for path in (fp32, prepared):
        path.unlink()
    return int8


def latency(predict, tweets: list[str]) -> float:
    """Milliseconds per tweet, one tweet at a time (as a bot sees them), on this CPU."""
    predict(tweets[:2])
    start = time.perf_counter()
    for tweet in tweets:
        predict([tweet])
    return (time.perf_counter() - start) / len(tweets) * 1000


def update_card(
    card: str, report: pd.DataFrame, speed: dict, sizes: dict, repo: str
) -> str:
    """The model card with a section on the ONNX int8 model, replacing an earlier one."""
    rows = [list(report.columns)] + report.astype(str).values.tolist()
    table = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    table = table.replace("\n", "\n| " + " | ".join("---" for _ in rows[0]) + " |\n", 1)
    section = f"""## ONNX int8

`{ONNX_FILE}` is this model with int8 weights (dynamic per-channel quantization of all
linear layers and the embeddings) for fast CPU inference, for example on a Raspberry Pi.
It needs only `onnxruntime` and `tokenizers`, no torch:

| | fp32 (PyTorch) | int8 (ONNX) |
| --- | --- | --- |
| size | {sizes["fp32"]:,.0f} MB | {sizes["int8"]:,.0f} MB |
| ms per tweet (batch 1, {torch.get_num_threads()} CPU threads) | {speed["fp32"]:.0f} | {speed["int8"]:.0f} |

```python
import json

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

repo = "{repo}"
config = json.loads(open(hf_hub_download(repo, "config.json")).read())
session = ort.InferenceSession(hf_hub_download(repo, "{ONNX_FILE}"))
tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
tokenizer.enable_truncation(128)
tokenizer.enable_padding(pad_id=1)

encodings = tokenizer.encode_batch(["$NVDA breaking out, loading calls"])
logits = session.run(None, {{
    "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
    "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
}})[0] / config["calibration_temperature"]
probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)  # NEUTRAL, BULLISH, BEARISH
```

Apply `config.decision_thresholds` to `probs` the same way as for the PyTorch model. ModernFinTwitBERT's
`onnx_predict.OnnxSentiment` does all of this, including the training preprocessing
(`@USER`, `[URL]`). Scores on the same dev and test sets; `same_label_as_fp32` is how often
both models give the same argmax label:

{table}
"""
    # Keep the section's place on a rerun; a new one goes before the citation section
    pattern = re.compile(r"## ONNX int8\n.*?(?=\n## |\Z)", re.DOTALL)
    if pattern.search(card):
        return pattern.sub(lambda _: section.rstrip() + "\n", card, count=1)
    if "\n## Citing & Authors" in card:
        return card.replace(
            "\n## Citing & Authors", "\n" + section + "\n## Citing & Authors", 1
        )
    return card.rstrip() + "\n\n" + section


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="StephanAkkerman/FinTwitRoBERTa-sentiment")
    parser.add_argument("--hub-repo", help="upload the int8 model to this HF repo")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    folder = Path("output/onnx") / args.model.replace("/", "--")
    folder.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(folder)
    tokenizer.save_pretrained(folder)

    int8 = export(model, tokenizer, folder)
    print(f"Saved {int8} ({int8.stat().st_size / 1e6:.0f} MB)\n")

    print("Checking the ONNX model against the PyTorch model on dev and test sets...")
    onnx_model = OnnxSentiment(str(folder))
    thresholds = getattr(model.config, "decision_thresholds", None) or {}
    report = []
    for name, dataset in (("dev", load_dev_data()), ("test", load_test_data())):
        df = dataset.to_pandas()
        labels = df["label"].to_numpy()
        # The datasets are already preprocessed; preprocess_tweet leaves such text unchanged
        probs = {
            "fp32": predict_probs(model, tokenizer, df["text"].tolist()),
            "int8": onnx_model.probs(df["text"].tolist()),
        }
        same = np.mean(probs["fp32"].argmax(1) == probs["int8"].argmax(1))
        for setting, t in (("argmax", {}), ("thresholds", thresholds)):
            for variant, p in probs.items():
                report.append(
                    {
                        "split": name,
                        "model": variant,
                        "setting": setting,
                        **metrics(labels, apply_thresholds(p, t)),
                        "same_label_as_fp32": same,
                    }
                )
    report = pd.DataFrame(report).round(3)
    report.to_csv(folder / "report.csv", index=False)
    print(report.to_string(index=False))

    tweets = load_test_data()[:LATENCY_TWEETS]["text"]
    speed = {
        "fp32": latency(lambda x: predict_probs(model, tokenizer, x), tweets),
        "int8": latency(onnx_model.probs, tweets),
    }
    sizes = {
        "fp32": sum(p.numel() for p in model.parameters()) * 4 / 1e6,
        "int8": int8.stat().st_size / 1e6,
    }
    print(f"\nms per tweet (batch 1): {speed}, size MB: {sizes}")

    readme = Path(args.model) / "README.md"
    if readme.exists():
        card = readme.read_text()
    elif args.hub_repo or not Path(args.model).is_dir():
        card = Path(
            hf_hub_download(args.hub_repo or args.model, "README.md")
        ).read_text()
    else:
        card = ""
    card = update_card(card, report, speed, sizes, args.hub_repo or args.model)
    (folder / "README.md").write_text(card)
    print(f"Wrote the model card to {folder / 'README.md'}")

    if args.hub_repo:
        message = "Add int8 ONNX model (per-channel dynamic quantization)"
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(int8),
            path_in_repo=ONNX_FILE,
            repo_id=args.hub_repo,
            commit_message=message,
        )
        api.upload_file(
            path_or_fileobj=str(folder / "README.md"),
            path_in_repo="README.md",
            repo_id=args.hub_repo,
            commit_message=message,
        )
        print(f"\nUploaded {ONNX_FILE} and the model card to {args.hub_repo}")


if __name__ == "__main__":
    main()
