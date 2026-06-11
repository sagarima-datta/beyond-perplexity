import torch

# ── sample mode ───────────────────────────────────────────────────────────────
# Set SAMPLE_MODE = True for a fast sanity-check run:
#   • MAX_TOKENS = 2_000
#   • WikiText-103 uses the validation split
#   • Both corpora capped at 50 documents
SAMPLE_MODE = False

MODEL_CONFIGS = {
    "gpt2":        "gpt2",
    "gpt2-medium": "gpt2-medium",
    "opt-125m":    "facebook/opt-125m",
}

CORPUS_CONFIGS = {
    "wikitext103": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "validation" if SAMPLE_MODE else "test",
        "text_field": "text",
        "max_docs": 50 if SAMPLE_MODE else None,
    },
    "ptb": {
        "path": "nltk",          # loaded via nltk.corpus.treebank
        "max_docs": 50 if SAMPLE_MODE else None,
    },
}

# ── evaluation ────────────────────────────────────────────────────────────────
MAX_TOKENS      = 2_000 if SAMPLE_MODE else 20_000  # 20 000 = setting used in the paper
MAX_SEQ_LEN     = 128 if SAMPLE_MODE else 512
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── scoring ───────────────────────────────────────────────────────────────────
# For larger runs, increase MAX_TOKENS before increasing MC_SAMPLES.  The
# default 200 draws is a reasonable runtime/variance tradeoff; use 500 only for
# a final sensitivity run if energy/kernel rankings look unstable.
MC_SAMPLES          = 200    # Monte Carlo draws for CRPS / energy / kernel expectations
KERNEL_N_SUBSAMPLE  = 2_000  # random embedding pairs used for median-heuristic σ

RESULTS_DIR = "results_sample" if SAMPLE_MODE else "results"
