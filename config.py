import torch

MODEL_CONFIGS = {
    "gpt2":        "gpt2",
    "gpt2-medium": "gpt2-medium",
    "opt-125m":    "facebook/opt-125m",
}

CORPUS_CONFIGS = {
    "wikitext103": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "test",
        "text_field": "text",
    },
    "pile": {
        "path": "monology/pile-uncopyrighted",
        "split": "train",
        "text_field": "text",
        "max_docs": 10_000,
    },
}

# ── evaluation ──────────────────────────────────────────────────────────────
MAX_TOKENS      = 5_000    # token positions evaluated per (model, corpus)  [set 50_000 for full run]
MAX_SEQ_LEN     = 512      # context window / truncation length
CHUNK_SIZE      = 256      # token-positions processed at once for CRPS memory
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── scoring ──────────────────────────────────────────────────────────────────
MC_SAMPLES          = 200    # Monte Carlo draws for energy / kernel expectations
KERNEL_N_SUBSAMPLE  = 2_000  # random embedding pairs used for median-heuristic σ

RESULTS_DIR = "results"
