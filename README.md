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


## Benchmark 📊

`ModernFinTwitBERT/benchmark.py` scores models on the frozen tweet test set (`data/test/test_v0.csv`). The models are listed in `benchmark.json`:

- `"base"`: a pretrained model (Hugging Face id or local path) that is fine-tuned on our sentiment data with the recipe in `config.json`, once per seed in `"seeds"`, then scored. Use this to compare starting points, such as ModernBERT-base against a finance-pretrained ModernBERT or our own domain-pretrained model.
- `"classifier"`: an existing sentiment model that is scored as is, such as our released model or other people's. Add a `"label_map"` when its labels are not neutral/bullish/bearish or positive/negative, and a `"tokenizer"` when the model was uploaded without one.

`"finetune_args"` (top level, or per model) overrides the fine-tuning arguments in `config.json`. Run from the repository root:

```bash
python ModernFinTwitBERT/benchmark.py
python ModernFinTwitBERT/benchmark.py --only fin-modernbert   # run one entry
```

Every run is appended to `output/benchmark/results.csv` and runs already there are skipped, so adding a model to `benchmark.json` and rerunning only runs the new one. `output/benchmark/summary.md` has the mean and standard deviation over seeds per model. `macro_f1_low`/`macro_f1_high` is a 95% bootstrap interval over the test tweets: differences inside it are likely noise.

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
