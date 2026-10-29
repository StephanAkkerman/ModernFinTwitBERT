# ModernFinTwitBERT

<!-- Add a banner here like: https://github.com/StephanAkkerman/fintwit-bot/blob/main/img/logo/fintwit-banner.png -->

---
<!-- Adjust the link of the second badge to your own repo -->
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Supported versions">
  <img src="https://img.shields.io/github/license/StephanAkkerman/ModernFinTwitBERT?color=brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

## Introduction

**ModernFinTwitBERT** is a transformer-based sentiment analysis model fine-tuned specifically for financial Twitter (FinTwit) content. It builds on top of [ModernBERT](https://github.com/StephanAkkerman/ModernBERT), a lightweight and optimized language model for modern short-form text, and is trained to accurately classify tweets related to stocks, finance, and investing into sentiment categories such as bullish, bearish, or neutral.

This project aims to provide a fast, lightweight, and domain-adapted model for analyzing financial discourse on social media. It is ideal for researchers, retail traders, and data scientists interested in extracting sentiment signals from noisy, fast-moving FinTwit data.

## Table of Contents 🗂

- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Key Features 🔑

- ✅ Built on top of **ModernBERT** for optimized performance on short-form content  
- 🧠 Fine-tuned on tens of thousands of labeled FinTwit examples  
- 💬 Supports multi-class sentiment classification (e.g. bullish / bearish / neutral)  
- ⚡ Lightweight & efficient — suitable for real-time applications  
- 🛠 Easy integration with other NLP or finance pipelines  

## Installation ⚙️

The required packages to run this code can be found in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/StephanAkkerman/ModernFinTwitBERT.git
```

## Usage ⌨️
You can load the model using transformers and run predictions on FinTwit content
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("StephanAkkerman/ModernFinTwitBERT")
model = AutoModelForSequenceClassification.from_pretrained("StephanAkkerman/ModernFinTwitBERT")

tweet = "Feeling bullish on $AAPL after that earnings call!"
inputs = tokenizer(tweet, return_tensors="pt")
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

print(f"Predicted sentiment: {predicted_class}")
```
>    Tip: Label mappings and sentiment categories are defined in the config or id2label dictionary of the model.


## Training 🏋️

Training is configured in `config.json` and run from the repository root with `python ModernFinTwitBERT/model.py`. The base model is `cardiffnlp/twitter-roberta-large-2022-154m`, which scored best in the benchmark (macro F1 0.807 on the test set).

1. **Domain pretraining:** set `"mode": "pretrain"`. This continues masked-language-model training on the sources in `pretrain.datasets` (by default `StephanAkkerman/financial-tweets`, shown 3 times, and `StephanAkkerman/crypto-stock-tweets`, with the test and dev tweets removed) and saves to `output/<pretrain model_name>`. One epoch takes a few hours on an A100. The released result is [FinTwitRoBERTa-large](https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-large).

   - At the end, the run measures the perplexity of the model and its base model on the test and dev tweets (kept out of pretraining) and writes a Hugging Face model card (`README.md`) and `pretrain_stats.json` next to the model. `python ModernFinTwitBERT/pretrain_card.py` rewrites the card later, e.g. to add the benchmark results; see its docstring.
   - On a rented GPU, `scripts/pretrain_remote.sh` does all of this: it installs the requirements (without flash-attn, which RoBERTa does not use), runs a 50-step trial, then the full run, and uploads the model with its card to `pretrain.hub_repo` from `config.json` (or `HF_REPO`). A new repo is created private.
   - Start with `python ModernFinTwitBERT/model.py --max-steps 50`: a short trial run that saves nothing, to check speed and memory on a new GPU.
   - The pretraining batch (256) is sized for an 80 GB A100. On a 40 GB GPU, use 128 with `gradient_accumulation_steps` 2.
   - `pretrain.datasets` in `config.json` lists the sources: Hugging Face ids or local `.csv`/`.jsonl`/`.parquet` files, each with an optional `text_column` and `repeat` to upweight it. Put curated sources first: near-duplicates are kept in the first source listed.
   - The cleaned tweets are cached in `datasets/pretrain_clean`, and cleaned again automatically when a source dataset, the cleaning code or the test set changes.
   - Checkpoints go to `output/checkpoints/<model_name>`. Running the same command again resumes from the last one.
2. **Sentiment fine-tuning:** set `"mode": "finetune"` and point `finetune.base_model` at the pretraining output, e.g. `output/FinTwitRoBERTa-large`.

To check whether pretraining helped, add the pretraining output to `benchmark.json` as a `"base"` entry with `"finetune_args": {"learning_rate": 2e-05}` and compare it with `twitter-roberta-large-2022`.

## Benchmark 📊

`ModernFinTwitBERT/benchmark.py` scores models on the frozen tweet test set (`data/test/test_v0.csv`). The models are listed in `benchmark.json`:

- `"base"`: a pretrained model (Hugging Face id or local path) that is fine-tuned on our sentiment data with the recipe in `config.json`, once per seed in `"seeds"`, then scored. Use this to compare starting points, such as ModernBERT-base against a finance-pretrained ModernBERT or our own domain-pretrained model.
- `"classifier"`: an existing sentiment model that is scored as is, such as our released model or other people's. Add a `"label_map"` when its labels are not neutral/bullish/bearish or positive/negative, and a `"tokenizer"` when the model was uploaded without one.

`"finetune_args"` (top level, or per model) overrides the fine-tuning arguments in `config.json`. Run from the repository root:

```bash
python ModernFinTwitBERT/benchmark.py
python ModernFinTwitBERT/benchmark.py --only fin-modernbert   # run one entry
```

Every run is appended to `output/benchmark/results.csv` and runs already there are skipped, so adding a model to `benchmark.json` and rerunning only runs the new one. Add `--rerun` to run the selected entries again from scratch. `output/benchmark/summary.md` has the mean and standard deviation over seeds per model. `macro_f1_low`/`macro_f1_high` is a 95% bootstrap interval over the test tweets: differences inside it are likely noise.

## Experiments 🧪

`ModernFinTwitBERT/experiments.py` fine-tunes the model with every setting in `experiments.json`, once per seed, and logs each run to wandb (project `FinTwitRoBERTa-finetune`, one group per experiment). Run `wandb login` once first; without a login, results are only saved locally.

```bash
python ModernFinTwitBERT/experiments.py
python ModernFinTwitBERT/experiments.py --only llrd balanced
```

- `"grid"` runs every combination of the listed TrainingArguments values as its own experiment, e.g. `grid-lr2e-05-ep3`.
- `"llrd": 0.9` uses layer-wise learning-rate decay; `"class_weights": "balanced"` weighs the loss by inverse class frequency; any other TrainingArguments (such as `label_smoothing_factor`) go in `"finetune_args"`.
- Pick the winner by dev macro F1 (`dev/macro_f1`) and use the test scores only to confirm it, so we don't tune on the test set. The dev set (`data/test/dev_v0.csv`, 749 tweets from 2024 onwards) comes from the same feed and guideline as the test set. `val/f1` is on a split of the training sources and is not a good guide: it ranked the first round of experiments almost in reverse order of test.
- `test/ece_calibrated` is the calibration error after temperature scaling on the dev set, and `ensemble macro_f1` scores the average of an experiment's seeds.
- Results go to `output/experiments/results.csv` and `summary.md`; finished runs are skipped on a rerun.

### Confidence thresholds

`ModernFinTwitBERT/thresholds.py` tunes how confident the model must be before a tweet counts as bullish or bearish (below the threshold it is neutral). It picks, per class, the lowest threshold whose dev precision reaches `--target-precision` (default 0.9), confirms it on the test set and writes the full tradeoff tables to `output/thresholds/`. Rerun it after every retrain.

```bash
python ModernFinTwitBERT/thresholds.py
python ModernFinTwitBERT/thresholds.py --model output/experiments/models/llrd-0.9-seed42 --recompute --save
```

`--save` stores them in the model config as `decision_thresholds`, and `--hub-repo` uploads that config to Hugging Face.

### Small and fast inference (ONNX int8)

`ModernFinTwitBERT/export_onnx.py` exports the model to ONNX with int8 weights (per-channel dynamic quantization). It is about 4x smaller (358 MB instead of 1.4 GB) and 4x faster on CPU, with the same dev and test F1. The script checks this on both sets and `--hub-repo` uploads the result as `onnx/model_int8.onnx`.

```bash
python ModernFinTwitBERT/export_onnx.py --hub-repo StephanAkkerman/FinTwitRoBERTa-sentiment
```

`ModernFinTwitBERT/onnx_predict.py` runs that model with only `onnxruntime`, `tokenizers` and `huggingface_hub` (no torch), and applies the preprocessing, calibration temperature and decision thresholds:

```python
from onnx_predict import OnnxSentiment

OnnxSentiment().predict(["$NVDA breaking out, adding here"])
```

## Citation ✍️
<!-- Be sure to adjust everything here so it matches your name and repo -->
If you use this project in your research, please cite as follows:

```bibtex
@misc{akkerman2025modernfintwitbert,
  author  = {Stephan Akkerman},
  title   = {ModernFinTwitBERT},
  year    = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/ModernFinTwitBERT}}
}
```

## Contributing 🛠
<!-- Be sure to adjust the repo name here for both the URL and GitHub link -->
Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.\
![https://github.com/StephanAkkerman/template/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=StephanAkkerman/template)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
