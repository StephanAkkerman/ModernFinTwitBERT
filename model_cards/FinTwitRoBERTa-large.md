---
license: mit
language:
- en
library_name: transformers
pipeline_tag: fill-mask
tags:
- NLP
- RoBERTa
- FinTwitBERT
- FinTwitRoBERTa
- finance
- financial-analysis
- sentiment
- financial-sentiment-analysis
- twitter
- tweets
- tweet-analysis
- stocks
- stock-market
- crypto
- cryptocurrency
datasets:
- StephanAkkerman/financial-tweets
- StephanAkkerman/crypto-stock-tweets
metrics:
- perplexity
widget:
- text: Paris is the <mask> of France.
  example_title: Generic
- text: AAPL is a <mask> sector stock.
  example_title: AAPL
- text: I predict that this stock will go <mask>.
  example_title: Stock Direction
- text: $AAPL is the ticker for the company named <mask>.
  example_title: Ticker
- text: Fed cuts rates, $SPY to all time <mask>.
  example_title: Market
base_model: cardiffnlp/twitter-roberta-large-2022-154m
model-index:
- name: FinTwitRoBERTa-large
  results:
  - task:
      type: fill-mask
      name: Masked Language Modeling
    dataset:
      name: Held-out financial tweets (2023-2026)
      type: StephanAkkerman/financial-tweets
    metrics:
    - type: perplexity
      value: 3.44
---

# FinTwitRoBERTa

FinTwitRoBERTa is a language model pre-trained on millions of financial tweets. It continues the pre-training of [Twitter-RoBERTa-large-2022](https://huggingface.co/cardiffnlp/twitter-roberta-large-2022-154m) to capture the tickers, jargon and communication style of financial Twitter, making it a strong starting point for sentiment analysis and other financial NLP tasks on tweets. It is the successor of [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT).

## Sentiment Analysis
The [FinTwitRoBERTa-sentiment](https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-sentiment) model fine-tunes FinTwitRoBERTa to classify financial tweets as bullish, bearish or neutral.

Starting from FinTwitRoBERTa instead of the original Twitter-RoBERTa-large raised the macro F1 of the fine-tuned sentiment model on our held-out test set of 1,237 financial tweets from 0.807 to 0.840 (mean of 3 seeds, same fine-tuning recipe).

## Dataset
FinTwitRoBERTa is pre-trained on two financial tweet datasets, consisting of tweets mentioning stocks and cryptocurrencies:
- [StephanAkkerman/financial-tweets](https://huggingface.co/datasets/StephanAkkerman/financial-tweets): 667,409 curated financial tweets from 2023 up to 2026. Quoted tweets are split into their own examples, and this source is shown 3 times per epoch to upweight recent, curated content.
- [StephanAkkerman/crypto-stock-tweets](https://huggingface.co/datasets/StephanAkkerman/crypto-stock-tweets): 8,024,269 tweets about stocks and cryptocurrencies.

Before training, the tweets were cleaned: near-duplicates (such as bot price alerts that only differ in numbers), one-word tweets and tweets that are mostly hashtags, cashtags, mentions or links were removed. This leaves about 7.5M training examples. The tweets of our sentiment test and dev sets were removed from the pre-training data.

## Model Details
- **Base model:** [cardiffnlp/twitter-roberta-large-2022-154m](https://huggingface.co/cardiffnlp/twitter-roberta-large-2022-154m), a RoBERTa-large model (355M parameters) trained on 154M tweets up to the end of 2022 ([TimeLMs](https://arxiv.org/abs/2202.03829)). It was the best of the dozen base models we benchmarked for financial tweet sentiment.
- **Tokenizer:** the unchanged RoBERTa byte-level BPE tokenizer of the base model (50,265 tokens). The mask token is `<mask>`.
- **Preprocessing:** links are replaced with `[URL]` and user mentions with `@USER`. These are not added as special tokens, so do the same replacement on your own tweets for the best results.
- **Training:** 1 epoch of masked language modeling (15% masking), learning rate 5e-5 with 6% warmup, weight decay 0.01, batch size 256, maximum length 128 tokens, on a single A100 80GB.
- **Perplexity:** 3.44 on 1,986 held-out financial tweets from 2023 to 2026 (our sentiment test and dev sets, removed from the pre-training data), with 15% of tokens masked. The base model scores 197 on the same tweets. These tweets come from the same feed as part of the pre-training data, so the gap is larger than you should expect on unrelated tweets; the sentiment results above are the better measure of the gain.

## More Information
For a comprehensive overview, including the complete training setup details and more, visit the [ModernFinTwitBERT GitHub repository](https://github.com/StephanAkkerman/ModernFinTwitBERT).

## Usage
Using [HuggingFace's transformers library](https://huggingface.co/docs/transformers/index) the model and tokenizer can be converted into a pipeline for masked language modeling.

```python
from transformers import pipeline

pipe = pipeline(
    "fill-mask",
    model="StephanAkkerman/FinTwitRoBERTa-large",
)
print(pipe("Fed cuts rates, $SPY to all time <mask>."))
```

## Citing & Authors

If you use FinTwitRoBERTa in your research, please cite me as follows:

```
@misc{FinTwitRoBERTa,
  author = {Stephan Akkerman},
  title = {FinTwitRoBERTa},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/ModernFinTwitBERT}}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.
