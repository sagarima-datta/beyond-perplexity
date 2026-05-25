"""
Experiment 2: Score behavior by token frequency.

Stratifies each evaluated token position by its unigram frequency decile
(decile 0 = most frequent 10% of vocab, decile 9 = rarest 10%) and reports
the mean score under each scoring rule per stratum.

Expected patterns (from proposal):
  - Log-score   : sharp increase toward rare tokens  (KL divergence → ∞ as p→0)
  - Quadratic   : flat across strata (bounded, L² penalises all vocab equally)
  - CRPS        : driven by rank structure — intermediate, nonlinear curve
  - Energy      : driven by semantic geometry of errors — may cross log-score curve
  - Kernel      : sharper locality than energy; sensitive to near-neighbour errors

Outputs
-------
  results/exp2_stratified_scores.csv   — mean score per (model, corpus, decile, rule)
  results/exp2_<corpus>.png            — per-corpus figure (5 rule panels + overlay)
"""

import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    MODEL_CONFIGS, CORPUS_CONFIGS,
    MAX_TOKENS, MAX_SEQ_LEN, RESULTS_DIR,
)
from utils.data import load_corpus
from transformers import AutoTokenizer

TOKEN_SCORE_DIR = os.path.join(RESULTS_DIR, "exp1_token_scores")
N_DECILES = 10
RULE_NAMES = ["log", "quadratic", "crps", "energy", "kernel"]

MODEL_COLORS = {
    "gpt2":        "#1f77b4",
    "gpt2-medium": "#ff7f0e",
    "opt-125m":    "#2ca02c",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def collect_target_ids(texts, tokenizer, max_seq_len, max_tokens):
    """
    Replay the same tokenisation order as exp1's evaluate() without model
    inference, returning the exact sequence of target token IDs that were
    scored in exp1.
    """
    target_ids = []
    tokens_processed = 0

    for text in texts:
        if tokens_processed >= max_tokens:
            break
        if not text.strip():
            continue

        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_seq_len, add_special_tokens=True)
        input_ids = enc["input_ids"][0]
        if len(input_ids) < 2:
            continue

        tgt = input_ids[1:].numpy().astype(np.int32)   # next-token targets
        remaining = max_tokens - tokens_processed
        target_ids.append(tgt[:remaining])
        tokens_processed += min(len(tgt), remaining)

    return np.concatenate(target_ids)


def build_vocab_counts(texts, tokenizer, max_seq_len, vocab_size):
    """Count unigram token frequencies across all texts (no model needed)."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    for text in tqdm(texts, desc="  counting tokens", leave=False):
        if not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_seq_len, add_special_tokens=True)
        ids = enc["input_ids"][0].numpy()
        np.add.at(counts, ids, 1)
    return counts


def frequency_decile_per_token(token_ids, vocab_counts, n_deciles=N_DECILES):
    """
    Map each token id to a frequency decile in [0, n_deciles-1].
    Decile 0 = most frequent fraction of the *observed* vocabulary.

    Ranking is restricted to tokens that actually appear in the corpus
    (vocab_counts > 0), so deciles are always well-populated regardless of
    corpus size. Tokens unseen in the corpus get the rarest decile.
    """
    # Restrict to tokens observed in this corpus
    seen_mask = vocab_counts > 0
    seen_ids = np.where(seen_mask)[0]                        # token ids that appear
    n_seen = len(seen_ids)

    if n_seen == 0:
        return np.zeros(len(token_ids), dtype=np.int64)

    # Rank within observed tokens: 0 = most frequent
    seen_counts = vocab_counts[seen_ids]
    order = np.argsort(-seen_counts)                         # descending frequency
    freq_rank = np.full(len(vocab_counts), n_seen, dtype=np.int64)  # unseen → max rank
    freq_rank[seen_ids[order]] = np.arange(n_seen)

    ranks = freq_rank[token_ids]
    deciles = (ranks * n_deciles // n_seen).clip(0, n_deciles - 1)
    return deciles


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_corpus(corpus_name, stratified_df, model_names, rule_names):
    """
    Single panel: relative degradation vs decile 0 (most frequent).
    Each line is score(decile d) / score(decile 0) for a given rule,
    averaged across models.  Values > 1 mean the score worsened relative
    to the most frequent tokens.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    deciles = np.arange(N_DECILES)
    sub = stratified_df[stratified_df["corpus"] == corpus_name]

    rule_styles = {
        "log":       ("-",  "o",  "#e41a1c"),
        "quadratic": ("--", "s",  "#377eb8"),
        "crps":      (":",  "^",  "#4daf4a"),
        "energy":    ("-.", "D",  "#ff7f00"),
        "kernel":    ((0,(3,1,1,1)), "v", "#984ea3"),
    }

    for rule in rule_names:
        # Average across models for each decile
        mean_per_decile = (
            sub[sub["rule"] == rule]
            .groupby("decile")["mean_score"]
            .mean()
            .reindex(deciles)
            .values.astype(float)
        )
        base = mean_per_decile[0]
        if np.isnan(base) or base == 0:
            continue
        rel = mean_per_decile / base
        ls, mk, col = rule_styles[rule]
        ax.plot(deciles, rel, linestyle=ls, marker=mk, color=col,
                label=rule, linewidth=1.8, markersize=5)

    ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xticks(deciles)
    ax.set_xlabel("Frequency decile  (0 = most frequent tokens)", fontsize=11)
    ax.set_ylabel("Relative score  (1 = same as decile 0)", fontsize=11)
    ax.set_title(
        f"Exp 2 — Relative score degradation by token frequency  [{corpus_name}]",
        fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []

    for corpus_name, corpus_cfg in CORPUS_CONFIGS.items():
        print(f"\nCorpus: {corpus_name}")
        texts = load_corpus(corpus_name, corpus_cfg)
        print(f"  {len(texts)} documents loaded.")

        for model_name, model_id in MODEL_CONFIGS.items():
            print(f"  Model: {model_name}")

            score_path = os.path.join(TOKEN_SCORE_DIR,
                                      f"{model_name}__{corpus_name}.npz")
            if not os.path.exists(score_path):
                print(f"    [SKIP] cached scores not found: {score_path}")
                continue

            # Load per-token scores from exp1
            data = np.load(score_path)
            scores = {r: data[r] for r in RULE_NAMES}
            n_tokens = len(scores["log"])
            print(f"    Loaded {n_tokens} token positions from cache.")

            # Load tokenizer (no model weights needed)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            vocab_size = tokenizer.vocab_size

            # Count unigram frequencies across whole corpus
            print("    Computing vocabulary frequency counts...")
            vocab_counts = build_vocab_counts(texts, tokenizer, MAX_SEQ_LEN, vocab_size)

            # Replay tokenisation to get the same target ids as exp1
            print("    Replaying tokenisation to recover target IDs...")
            target_ids = collect_target_ids(
                texts, tokenizer, MAX_SEQ_LEN, MAX_TOKENS)

            if len(target_ids) != n_tokens:
                print(f"    [WARN] target count mismatch: "
                      f"got {len(target_ids)}, expected {n_tokens}. "
                      f"Truncating to min.")
                n = min(len(target_ids), n_tokens)
                target_ids = target_ids[:n]
                scores = {r: v[:n] for r, v in scores.items()}

            # Assign frequency deciles
            decile_labels = frequency_decile_per_token(
                target_ids, vocab_counts, N_DECILES)

            # Aggregate: mean score per decile per rule
            for decile in range(N_DECILES):
                mask = decile_labels == decile
                n_in_decile = mask.sum()
                for rule in RULE_NAMES:
                    mean_s = float(scores[rule][mask].mean()) if n_in_decile > 0 else np.nan
                    records.append({
                        "model":      model_name,
                        "corpus":     corpus_name,
                        "decile":     decile,
                        "rule":       rule,
                        "mean_score": mean_s,
                        "n_tokens":   int(n_in_decile),
                    })

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "exp2_stratified_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved stratified scores to {csv_path}")

    # ── Print relative degradation: decile d vs decile 0 ─────────────────────
    print("\nRelative degradation vs decile 0 (score_d / score_0, averaged across models):")
    for corpus_name in CORPUS_CONFIGS:
        print(f"\n  {corpus_name}:")
        sub = df[df["corpus"] == corpus_name]
        for rule in RULE_NAMES:
            base = (sub[(sub["rule"] == rule) & (sub["decile"] == 0)]
                    .groupby("model")["mean_score"].mean().mean())
            last = (sub[(sub["rule"] == rule) & (sub["decile"] == N_DECILES - 1)]
                    .groupby("model")["mean_score"].mean().mean())
            if base > 0:
                print(f"    {rule:10s}  decile 0 → {N_DECILES-1}: "
                      f"{base:.4f} → {last:.4f}  ({last/base:.2f}x)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    model_names = list(MODEL_CONFIGS.keys())
    for corpus_name in CORPUS_CONFIGS:
        fig = plot_corpus(corpus_name, df, model_names, RULE_NAMES)
        png_path = os.path.join(RESULTS_DIR, f"exp2_{corpus_name}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
