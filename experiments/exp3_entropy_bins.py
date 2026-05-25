"""
Experiment 3: Score behavior by prediction entropy.

Groups each evaluated token position by the Shannon entropy H(p) of the
model's softmax distribution, split into low / medium / high tertiles.

Hypothesis (Gneiting & Raftery 2007 + proposal):
  High-entropy predictions place mass diffusely across the vocabulary.
  Log-score (KL) penalises the realized token harshly — it only cares about
  p(y), so if p is spread out, -log p(y) blows up regardless of whether the
  mass is semantically near y.  Energy and kernel scores depend on WHERE in
  embedding space the mass lands, so a model that is uncertain but still
  concentrates mass in y's semantic neighbourhood receives a relatively low
  energy/kernel score even at high entropy.

  => The log/energy ratio should INCREASE with entropy.
  => The gap between log-score and energy/kernel curves should widen in the
     high-entropy bin.

Outputs
-------
  results/exp3_entropy_bins.csv         — mean score per (model, corpus, bin, rule)
  results/exp3_ratio.csv                — log/energy and log/kernel ratios per bin
  results/exp3_<corpus>.png             — 6-panel figure per corpus
  results/exp1_token_scores/<m>__<c>__entropy.npy   — entropy cache (nats)
"""

import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    MODEL_CONFIGS, CORPUS_CONFIGS,
    MAX_TOKENS, MAX_SEQ_LEN, DEVICE, RESULTS_DIR,
)
from utils.data import load_corpus
from utils.models import load_model_and_tokenizer

TOKEN_SCORE_DIR = os.path.join(RESULTS_DIR, "exp1_token_scores")
RULE_NAMES      = ["log", "quadratic", "crps", "energy", "kernel"]
BIN_LABELS      = ["low", "medium", "high"]

MODEL_COLORS = {
    "gpt2":        "#1f77b4",
    "gpt2-medium": "#ff7f0e",
    "opt-125m":    "#2ca02c",
}


# ── entropy collection ────────────────────────────────────────────────────────

def collect_entropy(model, tokenizer, texts, device,
                    max_tokens, max_seq_len) -> np.ndarray:
    """
    One forward pass per document — compute H(p) = -sum_k p_k log p_k (nats)
    at every evaluated token position, in the same order as exp1's evaluate().
    No MC scoring; this is fast even on CPU.
    """
    entropies = []
    tokens_processed = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="  computing entropy", leave=False,
                         dynamic_ncols=True):
            if tokens_processed >= max_tokens:
                break
            if not text.strip():
                continue

            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_seq_len,
                            add_special_tokens=True).to(device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            logits = model(**enc).logits[0, :-1, :]   # (T-1, V)

            # Trim to remaining token budget
            remaining = max_tokens - tokens_processed
            logits = logits[:remaining]

            probs = torch.softmax(logits, dim=-1)      # (T', V)
            lp    = torch.log(probs.clamp(min=1e-40))
            H     = -(probs * lp).sum(dim=-1)          # (T',)
            entropies.append(H.cpu().numpy())
            tokens_processed += logits.shape[0]

    return np.concatenate(entropies)


# ── binning ───────────────────────────────────────────────────────────────────

def tertile_bins(entropy: np.ndarray):
    """Return bin label array ('low'/'medium'/'high') and the two cut-points."""
    p33, p66 = np.percentile(entropy, [33.3, 66.6])
    labels = np.where(entropy < p33, "low",
             np.where(entropy < p66, "medium", "high"))
    return labels, p33, p66


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_corpus(corpus_name, df_bins, model_names):
    """
    Single panel: relative score vs lowest-entropy bin, per rule.
    Each line is  score(bin) / score("low")  averaged across models.
    Values > 1 mean the score worsened relative to the low-entropy baseline.
    Mirrors Exp 2's style (relative to most common decile).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(BIN_LABELS))

    sub = df_bins[df_bins["corpus"] == corpus_name]

    rule_styles = {
        "log":       ("-",  "o",  "#e41a1c"),
        "quadratic": ("--", "s",  "#377eb8"),
        "crps":      (":",  "^",  "#4daf4a"),
        "energy":    ("-.", "D",  "#ff7f00"),
        "kernel":    ((0,(3,1,1,1)), "v", "#984ea3"),
    }

    for rule in RULE_NAMES:
        mean_per_bin = (
            sub[sub["rule"] == rule]
            .groupby("bin")["mean_score"]
            .mean()
            .reindex(BIN_LABELS)
            .values.astype(float)
        )
        base = mean_per_bin[0]          # "low" entropy bin
        if np.isnan(base) or base == 0:
            continue
        rel = mean_per_bin / base
        ls, mk, col = rule_styles[rule]
        ax.plot(x, rel, linestyle=ls, marker=mk, color=col,
                label=rule, linewidth=1.8, markersize=6)

    ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=11)
    ax.set_xlabel("Entropy bin  (low → high prediction uncertainty)", fontsize=11)
    ax.set_ylabel("Relative score  (1 = same as low-entropy bin)", fontsize=11)
    ax.set_title(
        f"Exp 3 — Relative score by prediction entropy  [{corpus_name}]",
        fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bin_records = []

    for model_name, model_id in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        # Check which corpora still need entropy computed
        corpora_needing_model = [
            c for c in CORPUS_CONFIGS
            if not os.path.exists(
                os.path.join(TOKEN_SCORE_DIR,
                             f"{model_name}__{c}__entropy.npy"))
        ]

        if corpora_needing_model:
            print(f"  Loading model for entropy inference...")
            model, tokenizer = load_model_and_tokenizer(model_id, DEVICE)
        else:
            print(f"  All entropy caches present — skipping model load.")
            model, tokenizer = None, None
            # Still need tokenizer for vocab_size (load weights-free)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        for corpus_name, corpus_cfg in CORPUS_CONFIGS.items():
            print(f"\n  Corpus: {corpus_name}")
            entropy_cache = os.path.join(
                TOKEN_SCORE_DIR, f"{model_name}__{corpus_name}__entropy.npy")
            score_cache = os.path.join(
                TOKEN_SCORE_DIR, f"{model_name}__{corpus_name}.npz")

            if not os.path.exists(score_cache):
                print(f"    [SKIP] exp1 scores not found: {score_cache}")
                continue

            # Entropy — compute once, then cache
            if os.path.exists(entropy_cache):
                print(f"    Loading entropy from cache.")
                entropy = np.load(entropy_cache)
            else:
                texts = load_corpus(corpus_name, corpus_cfg)
                print(f"    Loaded {len(texts)} documents.")
                entropy = collect_entropy(
                    model, tokenizer, texts, DEVICE,
                    MAX_TOKENS, MAX_SEQ_LEN)
                np.save(entropy_cache, entropy)
                print(f"    Entropy cached ({len(entropy)} positions).")

            # Load per-token scores from exp1
            data   = np.load(score_cache)
            scores = {r: data[r] for r in RULE_NAMES}
            n_tok  = len(scores["log"])

            # Align lengths (entropy and scores must match)
            n = min(len(entropy), n_tok)
            if n < len(entropy) or n < n_tok:
                print(f"    [WARN] length mismatch; truncating to {n}.")
            entropy = entropy[:n]
            scores  = {r: v[:n] for r, v in scores.items()}

            # Assign entropy bins (tertiles)
            bin_labels, p33, p66 = tertile_bins(entropy)
            print(f"    Entropy tertile cuts: "
                  f"low < {p33:.3f} | medium < {p66:.3f} | high >= {p66:.3f} nats")

            # Mean score per bin per rule
            for b in BIN_LABELS:
                mask = bin_labels == b
                n_b  = mask.sum()
                row  = {"model": model_name, "corpus": corpus_name,
                        "bin": b, "n_tokens": int(n_b)}
                for rule in RULE_NAMES:
                    row[f"mean_{rule}"] = float(scores[rule][mask].mean()) \
                                         if n_b > 0 else np.nan
                bin_records.append(row)


            print(f"    Bin counts: "
                  f"low={( bin_labels=='low').sum()}, "
                  f"medium={(bin_labels=='medium').sum()}, "
                  f"high={(bin_labels=='high').sum()}")
            for b in BIN_LABELS:
                m = bin_labels == b
                print(f"    [{b:6s}] "
                      + "  ".join(f"{r}={scores[r][m].mean():.4f}"
                                  for r in RULE_NAMES))

        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Long-format scores DataFrame ─────────────────────────────────────────
    df_wide = pd.DataFrame(bin_records)
    # Melt into long format (model, corpus, bin, rule, mean_score)
    id_cols   = ["model", "corpus", "bin", "n_tokens"]
    value_cols = [f"mean_{r}" for r in RULE_NAMES]
    df_long = df_wide.melt(id_vars=id_cols, value_vars=value_cols,
                           var_name="rule", value_name="mean_score")
    df_long["rule"] = df_long["rule"].str.replace("mean_", "")

    # Save
    df_long.to_csv(os.path.join(RESULTS_DIR, "exp3_entropy_bins.csv"),
                   index=False)
    print(f"\nSaved CSV to {RESULTS_DIR}/")

    # ── Print low → high relative comparison ─────────────────────────────────
    print("\nRelative change low → high entropy (averaged across models):")
    for corpus_name in CORPUS_CONFIGS:
        sub = df_long[df_long["corpus"] == corpus_name]
        print(f"\n  {corpus_name}:")
        for rule in RULE_NAMES:
            grp = (sub[sub["rule"] == rule]
                   .groupby("bin")["mean_score"]
                   .mean()
                   .reindex(BIN_LABELS))
            lo = grp["low"]
            hi = grp["high"]
            if lo > 0:
                print(f"    {rule:10s}  low={lo:.4f}  high={hi:.4f}  "
                      f"({hi/lo:.2f}x  +{(hi-lo)/lo*100:.1f}%)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    model_names = list(MODEL_CONFIGS.keys())
    for corpus_name in CORPUS_CONFIGS:
        fig = plot_corpus(corpus_name, df_long, model_names)
        png = os.path.join(RESULTS_DIR, f"exp3_{corpus_name}.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {png}")


if __name__ == "__main__":
    main()
