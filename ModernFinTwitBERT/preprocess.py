"""
Tweet preprocessing shared by training and inference. Only the standard library, so the
ONNX predictor (onnx_predict.py) can use it on a small device without datasets or torch.
"""

import html
import re

# Discord embeds add the quoted (or replied-to) tweet below the tweet itself, as
# "> [@user](https://twitter.com/user):" followed by "> " lines
QUOTE = re.compile(r"^> \[@[^\]]*\]\(https?://[^)\s]*\):\n?((?:^>.*(?:\n|$))*)", re.MULTILINE)


def split_quote(tweet: str) -> list[str]:
    """The tweet's own text and, if it quotes one, the quoted tweet's text."""
    # Some exports use \r\n, which QUOTE would not see past the "):" header
    tweet = tweet.replace("\r\n", "\n")
    match = QUOTE.search(tweet)
    if not match:
        return [tweet]
    own = tweet[: match.start()].strip()
    quoted = re.sub(r"^> ?", "", match.group(1), flags=re.MULTILINE).strip()
    return [part for part in (own, quoted) if part] or [own]


def preprocess_tweet(tweet: str) -> str:
    """
    Preprocess a tweet by replacing URLs, @mentions, and cash tags with tokens.

    Parameters
    ----------
    tweet : str
        The tweet to preprocess.

    Returns
    -------
    str
        The preprocessed tweet.
    """
    # Unescape HTML characters
    tweet = html.unescape(tweet)

    # Only the tweet's own words carry its sentiment; quoted tweets are scored on their own
    tweet = split_quote(tweet)[0]
    # Markdown links from Discord embeds
    tweet = re.sub(r"\[([^\]]*)\]\(https?://[^)\s]*\)", r"\1", tweet)

    # Replace URLs wiht URL token
    tweet = re.sub(r"http\S+", "[URL]", tweet)

    # Replace @mentions with @USER token
    tweet = re.sub(r"@\S+", "@USER", tweet)

    # Replace cash tags with [TICKER] token
    # tweet = re.sub(r"\$[A-Z]{1,5}\b", "[TICKER]", tweet)

    return tweet
