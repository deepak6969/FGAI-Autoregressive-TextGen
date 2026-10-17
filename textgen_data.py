"""Data preparation and text-quality metrics (no deep-learning dependency)."""
import os
import re
import urllib.request
from collections import Counter

import numpy as np

DATASET_NAME = "The Adventures of Sherlock Holmes (Project Gutenberg #1661)"
DATASET_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]")


def load_text(path):
    """Read the book (downloading it if absent) and strip Gutenberg boilerplate."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, path)
    raw = open(path, encoding="utf-8-sig").read()
    start = raw.find("*** START OF")
    start = raw.find("\n", start) + 1 if start != -1 else 0
    end = raw.find("*** END OF")
    text = raw[start:end if end != -1 else len(raw)]
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("_", "")
    return text


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(train_tokens, vocab_size):
    """Vocabulary from the TRAIN split only: <pad>=0, <unk>=1, then most frequent words."""
    words = [w for w, _ in Counter(train_tokens).most_common(vocab_size - 2)]
    itos = [PAD, UNK] + words
    return itos, {w: i for i, w in enumerate(itos)}


def encode(tokens, stoi):
    return np.array([stoi.get(t, 1) for t in tokens], dtype=np.int64)


def split_stream(ids, train=0.8, val=0.1):
    """Contiguous (chronological) split so no window straddles two splits."""
    a, b = int(len(ids) * train), int(len(ids) * (train + val))
    return ids[:a], ids[a:b], ids[b:]


def make_windows(ids, length, stride=1):
    """Input = `length` consecutive tokens, target = the very next token."""
    starts = np.arange(0, len(ids) - length, stride)
    x = ids[starts[:, None] + np.arange(length)]
    y = ids[starts + length]
    return x, y


def distinct_n(tokens, n):
    grams = list(zip(*[tokens[i:] for i in range(n)]))
    return len(set(grams)) / len(grams) if grams else 0.0


def repeated_trigram_fraction(tokens):
    """Share of trigrams that already appeared earlier in the sample (loop indicator)."""
    return 1.0 - distinct_n(tokens, 3)


def longest_repeat_run(tokens):
    """Longest run of the same token repeated back-to-back."""
    best = run = 1 if tokens else 0
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best
