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
    MAX_TOKENS, MAX_SEQ_LEN, CHUNK_SIZE, RESULTS_DIR,
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

def collect_target_ids(texts, tokenizer, max_seq_len, max_tokens, chunk_size):
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
        T = len(tgt)

        for start in range(0, T, chunk_size):
            if tokens_processed >= max_tokens:
                break
            end = min(start + chunk_size, T)
            target_ids.append(tgt[start:end])
            tokens_processed += end - start

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
    Decile 0 = most frequent fraction of the vocabulary.

    Uses the *vocabulary-level* decile: tokens in the top 1/n_deciles by
    frequency get decile 0, the next slice get decile 1, etc.
    Tokens with count 0 are placed in the rarest decile.
    """
    V = len(vocab_counts)
    # Rank: 0 = most frequent token id in vocabulary
    sorted_by_freq = np.argsort(-vocab_counts)          # most frequent first
    freq_rank = np.empty(V, dtype=np.int64)
    freq_rank[sorted_by_freq] = np.arange(V)

    ranks = freq_rank[token_ids]                        # rank of each evaluated token
    deciles = (ranks * n_deciles // V).clip(0, n_deciles - 1)
    return deciles


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_corpus(corpus_name, stratified_df, model_names, rule_names):
    """
    Two-row figure: one subplot per scoring rule (cols 0-4) plus a sixth panel
    showing log vs energy normalised to [0,1] to highlight crossing behaviour.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    deciles = np.arange(N_DECILES)
    sub = stratified_df[stratified_df["corpus"] == corpus_name]

    for idx, rule in enumerate(rule_names):
        ax = axes_flat[idx]
        for model in model_names:
            row = sub[(sub["model"] == model) & (sub["rule"] == rule)]
            row = row.sort_values("decile")
            ax.plot(row["decile"], row["mean_score"],
                    marker="o", label=model, color=MODEL_COLORS[model])
        ax.set_title(rule, fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency decile  (0 = most frequent)")
        ax.set_ylabel("Mean score (lower = better)")
        ax.set_xticks(deciles)
        ax.legend(fontsize=8)

    # Panel 6: normalised log vs energy overlay (key diagnostic from proposal)
    ax6 = axes_flat[5]
    for model in model_names:
        for rule, ls in [("log", "-"), ("energy", "--")]:
            row = sub[(sub["model"] == model) & (sub["rule"] == rule)]
            row = row.sort_values("decile")
            s = row["mean_score"].values.astype(float)
            s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
            ax6.plot(row["decile"].values, s_norm,
                     linestyle=ls, marker="o" if rule == "log" else "s",
                     color=MODEL_COLORS[model],
                     label=f"{model}/{rule}", linewidth=1.4, markersize=4)
    ax6.set_title("Log vs Energy  (normalised)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Frequency decile  (0 = most frequent)")
    ax6.set_ylabel("Normalised score")
    ax6.set_xticks(deciles)
    ax6.legend(fontsize=6, ncol=2)

    fig.suptitle(
        f"Experiment 2 — Score by token frequency decile  [{corpus_name}]",
        fontsize=14,
    )
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
                texts, tokenizer, MAX_SEQ_LEN, MAX_TOKENS, CHUNK_SIZE)

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

    # ── Print summary table ───────────────────────────────────────────────────
    print("\nDecile breakdown (mean score, averaged across models):")
    pivot = df.groupby(["corpus", "rule", "decile"])["mean_score"].mean().unstack("decile")
    print(pivot.to_string())

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
