"""Load and prepare corpora for evaluation."""

import itertools
from typing import List, Dict
import numpy as np
import torch
from datasets import load_dataset


def load_corpus(corpus_name: str, cfg: Dict) -> List[str]:
    """Return a list of raw text strings for the requested corpus."""
    if corpus_name == "wikitext103":
        ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
        texts = [row[cfg["text_field"]] for row in ds
                 if row[cfg["text_field"]].strip()]
        return texts

    elif corpus_name == "pile":
        ds = load_dataset(cfg["path"], split=cfg["split"], streaming=True)
        texts = [row[cfg["text_field"]]
                 for row in itertools.islice(ds, cfg["max_docs"])
                 if row[cfg["text_field"]].strip()]
        return texts

    else:
        raise ValueError(f"Unknown corpus: {corpus_name}")


def build_frequency_rank(token_ids_flat: np.ndarray, vocab_size: int):
    """
    Compute unigram frequency ranks from all observed token ids.

    Returns
    -------
    freq_rank    : torch.LongTensor (vocab_size,)
                   token_id → rank  (0 = most frequent)
    sorted_by_freq : torch.LongTensor (vocab_size,)
                   argsort of token ids by descending frequency
                   (i.e. sorted_by_freq[0] is the most frequent token id)
    """
    counts = np.bincount(token_ids_flat, minlength=vocab_size).astype(np.int64)
    # argsort ascending, so reverse for descending frequency
    sorted_by_freq = np.argsort(-counts)          # most frequent first
    freq_rank = np.empty(vocab_size, dtype=np.int64)
    freq_rank[sorted_by_freq] = np.arange(vocab_size)

    return (
        torch.from_numpy(freq_rank),
        torch.from_numpy(sorted_by_freq),
    )
