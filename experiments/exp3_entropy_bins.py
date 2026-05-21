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
    MAX_TOKENS, MAX_SEQ_LEN, CHUNK_SIZE, DEVICE, RESULTS_DIR,
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
                    max_tokens, max_seq_len, chunk_size) -> np.ndarray:
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
            T = logits.shape[0]

            for start in range(0, T, chunk_size):
                if tokens_processed >= max_tokens:
                    break
                end   = min(start + chunk_size, T)
                probs = torch.softmax(logits[start:end], dim=-1)   # (c, V)
                lp    = torch.log(probs.clamp(min=1e-40))
                H     = -(probs * lp).sum(dim=-1)                  # (c,)
                entropies.append(H.cpu().numpy())
                tokens_processed += end - start

    return np.concatenate(entropies)


# ── binning ───────────────────────────────────────────────────────────────────

def tertile_bins(entropy: np.ndarray):
    """Return bin label array ('low'/'medium'/'high') and the two cut-points."""
    p33, p66 = np.percentile(entropy, [33.3, 66.6])
    labels = np.where(entropy < p33, "low",
             np.where(entropy < p66, "medium", "high"))
    return labels, p33, p66


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_corpus(corpus_name, df_bins, df_ratio, model_names):
    """
    6 panels:
      0-4  — mean score per entropy bin for each of the 5 rules
      5    — log/energy and log/kernel score ratios (key diagnostic)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    flat = axes.flatten()
    x    = np.arange(len(BIN_LABELS))

    sub_bins  = df_bins[df_bins["corpus"]  == corpus_name]
    sub_ratio = df_ratio[df_ratio["corpus"] == corpus_name]

    for idx, rule in enumerate(RULE_NAMES):
        ax = flat[idx]
        for model in model_names:
            vals = (sub_bins[(sub_bins["model"] == model) &
                             (sub_bins["rule"]  == rule)]
                    .set_index("bin")
                    .reindex(BIN_LABELS)["mean_score"]
                    .values)
            ax.plot(x, vals, marker="o", label=model,
                    color=MODEL_COLORS[model])
        ax.set_xticks(x); ax.set_xticklabels(BIN_LABELS)
        ax.set_title(rule, fontweight="bold")
        ax.set_xlabel("Entropy bin")
        ax.set_ylabel("Mean score (lower = better)")
        ax.legend(fontsize=8)

    # Panel 5: log / energy and log / kernel ratios
    ax6 = flat[5]
    for model in model_names:
        for ratio_col, ls, marker in [
                ("log_over_energy", "-",  "o"),
                ("log_over_kernel", "--", "s")]:
            vals = (sub_ratio[(sub_ratio["model"] == model)]
                    .set_index("bin")
                    .reindex(BIN_LABELS)[ratio_col]
                    .values)
            ax6.plot(x, vals, linestyle=ls, marker=marker,
                     color=MODEL_COLORS[model],
                     label=f"{model} / {ratio_col.replace('log_over_', '')}",
                     linewidth=1.5, markersize=5)
    ax6.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax6.set_xticks(x); ax6.set_xticklabels(BIN_LABELS)
    ax6.set_title("Log / (Energy | Kernel) ratio", fontweight="bold")
    ax6.set_xlabel("Entropy bin")
    ax6.set_ylabel("Ratio (> 1 means log penalises more)")
    ax6.legend(fontsize=6, ncol=2)

    fig.suptitle(
        f"Experiment 3 — Score by prediction entropy  [{corpus_name}]",
        fontsize=14)
    plt.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bin_records   = []
    ratio_records = []

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
                    MAX_TOKENS, MAX_SEQ_LEN, CHUNK_SIZE)
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

                # Also store in long format for plotting
                for rule in RULE_NAMES:
                    ratio_records  # placeholder; built below

            # Log / energy and log / kernel ratios per bin
            for b in BIN_LABELS:
                mask   = bin_labels == b
                if mask.sum() == 0:
                    continue
                mean_log    = scores["log"][mask].mean()
                mean_energy = scores["energy"][mask].mean()
                mean_kernel = scores["kernel"][mask].mean()
                ratio_records.append({
                    "model":           model_name,
                    "corpus":          corpus_name,
                    "bin":             b,
                    "log_over_energy": mean_log / (mean_energy + 1e-12),
                    "log_over_kernel": mean_log / (mean_kernel + 1e-12),
                })

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

    df_ratio = pd.DataFrame(ratio_records)

    # Save
    df_long.to_csv(os.path.join(RESULTS_DIR, "exp3_entropy_bins.csv"),
                   index=False)
    df_ratio.to_csv(os.path.join(RESULTS_DIR, "exp3_ratio.csv"),
                    index=False)
    print(f"\nSaved CSVs to {RESULTS_DIR}/")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nLog/energy ratio by entropy bin (averaged across models):")
    summary = (df_ratio.groupby(["corpus", "bin"])[["log_over_energy",
                                                     "log_over_kernel"]]
               .mean()
               .reindex(BIN_LABELS, level="bin"))
    print(summary.to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    model_names = list(MODEL_CONFIGS.keys())
    for corpus_name in CORPUS_CONFIGS:
        fig = plot_corpus(corpus_name, df_long, df_ratio, model_names)
        png = os.path.join(RESULTS_DIR, f"exp3_{corpus_name}.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {png}")


if __name__ == "__main__":
    main()
