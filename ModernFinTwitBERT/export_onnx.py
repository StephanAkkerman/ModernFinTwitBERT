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

Writes output/onnx/<model>/ (onnx/model_int8.onnx, config.json, tokenizer.json);
--hub-repo uploads onnx/model_int8.onnx to the onnx/ folder of that Hugging Face repo.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_dev_data, load_test_data
from onnx_predict import ONNX_FILE, OnnxSentiment
from thresholds import predict_probs


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

    onnx_model = OnnxSentiment(str(folder))
    for name, dataset in (("dev", load_dev_data()), ("test", load_test_data())):
        df = dataset.to_pandas()
        labels = df["label"].to_numpy()
        # The datasets are already preprocessed; preprocess_tweet leaves such text unchanged
        reference = predict_probs(model, tokenizer, df["text"].tolist()).argmax(1)
        quantized = onnx_model.probs(df["text"].tolist()).argmax(1)
        print(
            f"{name}: macro F1 fp32 {f1_score(labels, reference, average='macro'):.3f}, "
            f"int8 {f1_score(labels, quantized, average='macro'):.3f}, "
            f"same label {np.mean(reference == quantized):.1%}"
        )

    if args.hub_repo:
        HfApi().upload_file(
            path_or_fileobj=str(int8),
            path_in_repo=ONNX_FILE,
            repo_id=args.hub_repo,
            commit_message="Add int8 ONNX model (per-channel dynamic quantization)",
        )
        print(f"\nUploaded {ONNX_FILE} to {args.hub_repo}")


if __name__ == "__main__":
    main()
