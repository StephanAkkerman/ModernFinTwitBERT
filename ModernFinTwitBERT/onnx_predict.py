"""
Sentiment of tweets with the int8 ONNX export of FinTwitRoBERTa-sentiment (export_onnx.py).
Needs only numpy, onnxruntime, tokenizers and huggingface_hub (which brings tqdm), no torch or transformers, so
it runs on a small device such as a Raspberry Pi.

Applies the same preprocessing as training, the calibration temperature and the decision
thresholds from the model config:

    from onnx_predict import OnnxSentiment

    model = OnnxSentiment()  # or OnnxSentiment("path/to/export")
    model.predict(["$NVDA breaking out, adding here"])
    # [{"label": "bullish", "probs": {"neutral": 0.04, "bullish": 0.95, "bearish": 0.01}}]
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from tqdm import tqdm

from preprocess import preprocess_tweet

ONNX_FILE = "onnx/model_int8.onnx"
MAX_LENGTH = 128


class OnnxSentiment:
    def __init__(
        self,
        model: str = "StephanAkkerman/FinTwitRoBERTa-sentiment",
        threads: int | None = None,
    ):
        """`model` is a local export folder or a Hugging Face repo with an onnx/ folder."""
        if Path(model).is_dir():
            folder = Path(model)
        else:
            from huggingface_hub import snapshot_download

            folder = Path(
                snapshot_download(
                    model, allow_patterns=[ONNX_FILE, "config.json", "tokenizer.json"]
                )
            )
        config = json.loads((folder / "config.json").read_text())
        id2label = {int(i): label.lower() for i, label in config["id2label"].items()}
        self.labels = [id2label[i] for i in range(len(id2label))]
        self.temperature = config.get("calibration_temperature") or 1.0
        self.thresholds = config.get("decision_thresholds") or {}

        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_LENGTH)
        self.tokenizer.enable_padding(pad_id=config["pad_token_id"])

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(folder / ONNX_FILE), options, providers=["CPUExecutionProvider"]
        )

    def probs(
        self, tweets: list[str], batch_size: int = 32, progress: str | None = None
    ) -> np.ndarray:
        """
        Calibrated class probabilities, in the order of self.labels. `progress` is the
        label of a progress bar (none by default).
        """
        texts = [preprocess_tweet(tweet) for tweet in tweets]
        logits = []
        starts = range(0, len(texts), batch_size)
        for start in tqdm(starts, desc=progress, disable=progress is None):
            encodings = self.tokenizer.encode_batch(texts[start : start + batch_size])
            inputs = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array(
                    [e.attention_mask for e in encodings], dtype=np.int64
                ),
            }
            logits.append(self.session.run(None, inputs)[0])
        logits = np.concatenate(logits) / self.temperature
        exp = np.exp(logits - logits.max(1, keepdims=True))
        return exp / exp.sum(1, keepdims=True)

    def predict(self, tweets: list[str], batch_size: int = 32) -> list[dict]:
        """Argmax label, set to neutral when a bullish/bearish call is below its threshold."""
        output = []
        for p in self.probs(tweets, batch_size):
            label = self.labels[int(p.argmax())]
            if p.max() < self.thresholds.get(label, 0.0):
                label = "neutral"
            output.append(
                {"label": label, "probs": {l: float(x) for l, x in zip(self.labels, p)}}
            )
        return output
