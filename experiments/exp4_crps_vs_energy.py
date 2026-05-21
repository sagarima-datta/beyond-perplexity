"""
Experiment 4: CRPS vs energy score on semantically structured errors.

CRPS orders the vocabulary by UNIGRAM FREQUENCY RANK (Cramer-von Mises
distance on the CDF).  The energy score uses EMBEDDING GEOMETRY (Euclidean
distance in the model's own representation space).

These two orderings will agree when frequent tokens also cluster together
semantically (e.g. common function words form their own tight cluster).
They will disagree for tokens where semantic proximity and frequency rank
diverge -- for example, two rare technical synonyms sit close in embedding
space but far apart in frequency rank, so CRPS sees them as very different
while the energy score treats the error as nearly costless.

Methodology
-----------
For the top-N most frequent vocabulary tokens:

  S(v)  = set of k nearest tokens by COSINE SIMILARITY in embedding space
  R(v)  = set of k nearest tokens by |freq_rank(u) - freq_rank(v)|

  discordance d(v) = 1 - |S(v) ∩ R(v)| / k   ∈ [0, 1]
     d = 0  =>  semantic and rank neighbours are identical  (concordant)
     d = 1  =>  no overlap at all                          (discordant)

Each evaluated token position is labelled:
  concordant   if its target token's d < median(d)
  discordant   if its target token's d >= median(d)
  other        if the target is not in the top-N subset

Within each group we compute:
  (a) mean CRPS and mean energy score
  (b) Kendall tau(CRPS, energy) at the token level

Prediction: tau should be LOWER in the discordant group because the two
scores use incompatible distance concepts on the tokens in that group.

Outputs
-------
  results/exp4_group_stats.csv          — mean scores per group
  results/exp4_kendall_tau.csv          — Kendall tau (CRPS, energy) per group
  results/exp4_<corpus>.png             — scatter + tau bar figure
  results/exp4_disc_<model>__<corpus>.npz — discordance map cache
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
from scipy.stats import kendalltau
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    MODEL_CONFIGS, CORPUS_CONFIGS,
    MAX_TOKENS, MAX_SEQ_LEN, CHUNK_SIZE, DEVICE, RESULTS_DIR,
)
from utils.data import load_corpus
from utils.models import load_model_and_tokenizer, get_embedding_matrix

TOKEN_SCORE_DIR = os.path.join(RESULTS_DIR, "exp1_token_scores")
DISC_CACHE_DIR  = RESULTS_DIR          # store discordance maps here

# k-NN parameters
TOP_N = 3_000    # consider only the top-N most frequent vocab tokens
K_NN  = 30      # neighbourhood size for semantic and rank k-NN

MODEL_COLORS = {
    "gpt2":        "#1f77b4",
    "gpt2-medium": "#ff7f0e",
    "opt-125m":    "#2ca02c",
}
GROUP_COLORS = {"concordant": "#2166ac", "discordant": "#d6604d"}


# ── tokenisation replay (identical to exp2) ───────────────────────────────────

def collect_target_ids(texts, tokenizer, max_seq_len, max_tokens, chunk_size):
    target_ids = []
    tokens_processed = 0
    for text in texts:
        if tokens_processed >= max_tokens:
            break
        if not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_seq_len, add_special_tokens=True)
        ids = enc["input_ids"][0]
        if len(ids) < 2:
            continue
        tgt = ids[1:].numpy().astype(np.int32)
        T   = len(tgt)
        for start in range(0, T, chunk_size):
            if tokens_processed >= max_tokens:
                break
            end = min(start + chunk_size, T)
            target_ids.append(tgt[start:end])
            tokens_processed += end - start
    return np.concatenate(target_ids)


# ── vocabulary frequency counting ─────────────────────────────────────────────

def build_vocab_counts(texts, tokenizer, max_seq_len, vocab_size):
    counts = np.zeros(vocab_size, dtype=np.int64)
    for text in tqdm(texts, desc="  counting token freqs", leave=False,
                     dynamic_ncols=True):
        if not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_seq_len, add_special_tokens=True)
        np.add.at(counts, enc["input_ids"][0].numpy(), 1)
    return counts


# ── discordance map ───────────────────────────────────────────────────────────

def compute_discordance(embeddings: torch.Tensor,
                        vocab_counts: np.ndarray,
                        top_n: int = TOP_N,
                        k: int = K_NN):
    """
    Returns
    -------
    top_tokens   : np.ndarray (top_n,)  — token IDs in descending frequency order
    disc_scores  : np.ndarray (top_n,)  — discordance d(v) in [0, 1]
    """
    V         = len(vocab_counts)
    top_n     = min(top_n, V)

    # Select top-N tokens by frequency
    sorted_by_freq = np.argsort(-vocab_counts)     # most frequent first
    top_tokens     = sorted_by_freq[:top_n]        # (top_n,) token IDs

    # ── Semantic k-NN via cosine similarity ───────────────────────────────────
    top_embs = embeddings[top_tokens].float()      # (top_n, d)

    # L2-normalise for cosine similarity
    norms    = top_embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    top_norm = top_embs / norms                    # (top_n, d)

    print(f"    Computing {top_n}x{top_n} cosine similarity matrix...")
    # Batch to avoid OOM on large top_n
    batch = 512
    sem_nn = np.empty((top_n, k), dtype=np.int32)
    for start in range(0, top_n, batch):
        end     = min(start + batch, top_n)
        chunk   = top_norm[start:end]              # (b, d)
        sim_row = torch.mm(chunk, top_norm.t())    # (b, top_n)
        # Exclude self-similarity
        for i in range(end - start):
            sim_row[i, start + i] = -2.0
        _, idx  = sim_row.topk(k, dim=1)          # (b, k)
        sem_nn[start:end] = idx.cpu().numpy()

    # ── Rank k-NN: k closest indices in the sorted_by_freq ordering ──────────
    # Since top_tokens is already sorted by frequency, token at position i
    # has rank i among top_n tokens.  Its rank-k-NN are the k positions
    # closest to i in {0, ..., top_n-1} \ {i}.
    rank_nn = np.empty((top_n, k), dtype=np.int32)
    for i in range(top_n):
        # Expand outward from i alternating left/right
        neighbors = []
        lo, hi    = i - 1, i + 1
        while len(neighbors) < k:
            if lo >= 0:
                neighbors.append(lo); lo -= 1
            if len(neighbors) < k and hi < top_n:
                neighbors.append(hi); hi += 1
            if lo < 0 and hi >= top_n:
                break                              # exhausted (top_n < k+1)
        rank_nn[i] = neighbors[:k]

    # ── Discordance: fraction of k-NN not shared ──────────────────────────────
    disc_scores = np.empty(top_n, dtype=np.float32)
    for i in range(top_n):
        overlap      = len(set(sem_nn[i]) & set(rank_nn[i]))
        disc_scores[i] = 1.0 - overlap / k

    return top_tokens, disc_scores


# ── group labelling ───────────────────────────────────────────────────────────

def label_positions(target_ids, top_tokens, disc_scores):
    """
    Map each evaluated token position to 'concordant', 'discordant', or 'other'.
    Median discordance is used as the split threshold.
    """
    # token_id -> discordance (NaN for tokens outside top_n)
    V          = int(target_ids.max()) + 1 if len(target_ids) else 1
    disc_map   = np.full(max(V, top_tokens.max() + 1), np.nan, dtype=np.float32)
    disc_map[top_tokens] = disc_scores

    d_per_pos  = disc_map[target_ids]
    valid_mask = ~np.isnan(d_per_pos)
    threshold  = np.median(disc_scores)           # median over top-N vocab

    labels     = np.full(len(target_ids), "other", dtype=object)
    labels[valid_mask & (d_per_pos <  threshold)] = "concordant"
    labels[valid_mask & (d_per_pos >= threshold)] = "discordant"
    return labels, threshold


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_corpus(corpus_name, stats_df, tau_df, model_names):
    """
    Left column  : scatter CRPS vs energy for concordant (blue) / discordant
                   (red) tokens, one subplot per model.
    Right column : Kendall tau comparison bar chart; one bar per model.
    """
    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 2,
                             figsize=(12, 4 * n_models),
                             squeeze=False)

    sub_stats = stats_df[stats_df["corpus"] == corpus_name]
    sub_tau   = tau_df[tau_df["corpus"]   == corpus_name]

    for row_idx, model_name in enumerate(model_names):
        # ── Scatter: CRPS vs energy by group ─────────────────────────────────
        ax_sc = axes[row_idx, 0]
        for grp in ["concordant", "discordant"]:
            mask_path = os.path.join(
                TOKEN_SCORE_DIR,
                f"exp4_{model_name}__{corpus_name}_{grp}_scores.npz")
            if not os.path.exists(mask_path):
                continue
            d = np.load(mask_path)
            crps_g   = d["crps"]
            energy_g = d["energy"]
            # Subsample for readability
            idx = np.random.choice(len(crps_g),
                                   min(500, len(crps_g)), replace=False)
            ax_sc.scatter(crps_g[idx], energy_g[idx],
                          alpha=0.35, s=8,
                          color=GROUP_COLORS[grp], label=grp)
        ax_sc.set_xlabel("CRPS")
        ax_sc.set_ylabel("Energy score")
        ax_sc.set_title(f"{model_name}  —  CRPS vs Energy", fontweight="bold")
        ax_sc.legend(fontsize=9)

        # ── Bar: Kendall tau per group ────────────────────────────────────────
        ax_tau = axes[row_idx, 1]
        m_tau  = sub_tau[sub_tau["model"] == model_name]
        groups = ["concordant", "discordant"]
        taus   = [m_tau[m_tau["group"] == g]["kendall_tau"].values
                  for g in groups]
        vals   = [t[0] if len(t) else np.nan for t in taus]
        bars   = ax_tau.bar(groups, vals,
                            color=[GROUP_COLORS[g] for g in groups],
                            alpha=0.8, edgecolor="k", linewidth=0.7)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax_tau.text(bar.get_x() + bar.get_width() / 2,
                            v + 0.005, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=10)
        ax_tau.set_ylim(0, 1)
        ax_tau.set_ylabel("Kendall tau (CRPS, energy)")
        ax_tau.set_title(f"{model_name}  —  Agreement between CRPS & Energy",
                         fontweight="bold")
        ax_tau.axhline(0.5, color="k", linestyle="--", linewidth=0.8,
                       label="tau = 0.5")
        ax_tau.legend(fontsize=8)

    fig.suptitle(
        f"Experiment 4 — CRPS vs Energy on Concordant/Discordant tokens"
        f"  [{corpus_name}]",
        fontsize=13)
    plt.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR,     exist_ok=True)
    os.makedirs(TOKEN_SCORE_DIR, exist_ok=True)

    stat_records = []
    tau_records  = []

    for model_name, model_id in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        # Load model + embeddings (needed for discordance map)
        model, tokenizer = load_model_and_tokenizer(model_id, DEVICE)
        embeddings       = get_embedding_matrix(model, DEVICE)
        vocab_size       = embeddings.shape[0]
        del model        # embeddings are all we need

        for corpus_name, corpus_cfg in CORPUS_CONFIGS.items():
            print(f"\n  Corpus: {corpus_name}")

            score_path = os.path.join(
                TOKEN_SCORE_DIR, f"{model_name}__{corpus_name}.npz")
            if not os.path.exists(score_path):
                print(f"    [SKIP] exp1 scores not found.")
                continue

            # ── Discordance map ──────────────────────────────────────────────
            disc_cache = os.path.join(
                DISC_CACHE_DIR,
                f"exp4_disc_{model_name}__{corpus_name}.npz")

            if os.path.exists(disc_cache):
                print(f"    Loading discordance map from cache.")
                d_cache    = np.load(disc_cache)
                top_tokens = d_cache["top_tokens"]
                disc_scores = d_cache["disc_scores"]
            else:
                texts = load_corpus(corpus_name, corpus_cfg)
                print(f"    Loaded {len(texts)} documents.")
                vocab_counts = build_vocab_counts(
                    texts, tokenizer, MAX_SEQ_LEN, vocab_size)
                print(f"    Computing discordance map "
                      f"(top_n={TOP_N}, k={K_NN})...")
                top_tokens, disc_scores = compute_discordance(
                    embeddings, vocab_counts, TOP_N, K_NN)
                np.savez_compressed(disc_cache,
                                    top_tokens=top_tokens,
                                    disc_scores=disc_scores)
                print(f"    Discordance map cached.")

            disc_median = float(np.median(disc_scores))
            print(f"    Discordance range: "
                  f"[{disc_scores.min():.3f}, {disc_scores.max():.3f}]  "
                  f"median={disc_median:.3f}")
            n_concordant  = (disc_scores <  disc_median).sum()
            n_discordant  = (disc_scores >= disc_median).sum()
            print(f"    Vocab split: {n_concordant} concordant / "
                  f"{n_discordant} discordant (of top-{TOP_N})")

            # ── Target IDs (replay tokenisation) ────────────────────────────
            texts = load_corpus(corpus_name, corpus_cfg)
            print(f"    Replaying tokenisation...")
            target_ids = collect_target_ids(
                texts, tokenizer, MAX_SEQ_LEN, MAX_TOKENS, CHUNK_SIZE)

            # ── Load scores ──────────────────────────────────────────────────
            data   = np.load(score_path)
            crps   = data["crps"]
            energy = data["energy"]

            n = min(len(target_ids), len(crps))
            target_ids = target_ids[:n]
            crps       = crps[:n]
            energy     = energy[:n]

            # ── Assign groups ────────────────────────────────────────────────
            group_labels, threshold = label_positions(
                target_ids, top_tokens, disc_scores)

            print(f"    Position split: "
                  f"concordant={(group_labels=='concordant').sum()}, "
                  f"discordant={(group_labels=='discordant').sum()}, "
                  f"other={(group_labels=='other').sum()}")

            # ── Per-group statistics ─────────────────────────────────────────
            for grp in ["concordant", "discordant"]:
                mask    = group_labels == grp
                c_grp   = crps[mask]
                e_grp   = energy[mask]
                n_grp   = mask.sum()

                if n_grp < 10:
                    print(f"    [WARN] too few {grp} positions ({n_grp}); skipping.")
                    continue

                # Save raw scores for scatter plot
                scat_path = os.path.join(
                    TOKEN_SCORE_DIR,
                    f"exp4_{model_name}__{corpus_name}_{grp}_scores.npz")
                np.savez_compressed(scat_path, crps=c_grp, energy=e_grp)

                stat_records.append({
                    "model":       model_name,
                    "corpus":      corpus_name,
                    "group":       grp,
                    "n_tokens":    int(n_grp),
                    "mean_crps":   float(c_grp.mean()),
                    "mean_energy": float(e_grp.mean()),
                    "disc_threshold": float(threshold),
                })

                # Kendall tau — subsample for speed
                n_sub = min(n_grp, 10_000)
                idx   = np.random.choice(n_grp, n_sub, replace=False)
                tau, pval = kendalltau(c_grp[idx], e_grp[idx])
                tau_records.append({
                    "model":       model_name,
                    "corpus":      corpus_name,
                    "group":       grp,
                    "kendall_tau": round(float(tau), 4),
                    "p_value":     round(float(pval), 6),
                    "n_tokens":    int(n_grp),
                })

                print(f"    [{grp:12s}] "
                      f"n={n_grp:5d}  "
                      f"CRPS={c_grp.mean():.4f}  "
                      f"energy={e_grp.mean():.4f}  "
                      f"tau(CRPS,energy)={tau:.4f}  p={pval:.4e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────────
    df_stats = pd.DataFrame(stat_records)
    df_tau   = pd.DataFrame(tau_records)
    df_stats.to_csv(os.path.join(RESULTS_DIR, "exp4_group_stats.csv"),
                    index=False)
    df_tau.to_csv(os.path.join(RESULTS_DIR, "exp4_kendall_tau.csv"),
                  index=False)
    print(f"\nSaved CSVs to {RESULTS_DIR}/")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nKendall tau (CRPS, energy) by group:")
    print(df_tau.to_string(index=False))

    print("\nMean scores by group (averaged across models):")
    print(df_stats.groupby(["corpus", "group"])[["mean_crps", "mean_energy"]]
          .mean().to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    model_names = list(MODEL_CONFIGS.keys())
    for corpus_name in CORPUS_CONFIGS:
        fig = plot_corpus(corpus_name, df_stats, df_tau, model_names)
        png = os.path.join(RESULTS_DIR, f"exp4_{corpus_name}.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {png}")


if __name__ == "__main__":
    main()
