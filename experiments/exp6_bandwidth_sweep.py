"""
Experiment 6: Kernel vs Energy Reversal — Bandwidth Sensitivity Sweep
======================================================================

Background
----------
From Experiment 1, OPT-125m ranks #1 under energy but #3 under the RBF kernel
score — the sharpest rule reversal in the data.  Energy grows roughly linearly
with embedding distance, while the Gaussian RBF kernel k(x,y) = exp(−d²/2σ²)
decays to zero once d exceeds a few bandwidths σ.

Hypothesis
----------
OPT's prediction errors land in a "middle distance" band of embedding space:
close enough to the truth for the (linear) energy score to give credit, but
beyond the RBF kernel's locality window at the median-heuristic σ, so the
kernel sees them as total misses.

If true, OPT's *rank under the kernel rule should improve as σ grows*: a wider
kernel window behaves more like energy.  Somewhere along the σ axis the model
ranking should flip from the kernel ordering (gpt2-medium > gpt2 > opt) to the
energy ordering (opt > gpt2-medium > gpt2).  The crossing point localises the
distance band where OPT's errors live.

Method
------
For each model × corpus we run one inference pass.  At every token position we
draw MC samples X, X' ~ P and record the squared embedding distances

    d²(X, y)   and   d²(X, X')

These do not depend on σ, so a single pass supports the entire sweep: for each
bandwidth multiplier m in SIGMA_MULTIPLIERS we evaluate the kernel score

    S_m = 1 − 2·E[exp(−d²(X,y) / 2(mσ₀)²)] + E[exp(−d²(X,X') / 2(mσ₀)²)]

where σ₀ is the model's own median-heuristic bandwidth (as used in Exp 1, so
multiplier m = 1 reproduces the Exp 1 kernel setting).  The energy score is
computed from the same draws as a σ-free reference.

Outputs
-------
  results/exp6_bandwidth/
      <model>__<corpus>.npz       mean kernel score per multiplier +
                                  mean energy score + sigma0
  results/exp6_sweep_scores.csv   mean kernel score per (model, corpus, multiplier)
  results/exp6_sweep_ranks.csv    model rank (1=best) per (corpus, multiplier)
  results/exp6_<corpus>.png       2-panel figure:
                                    (a) mean kernel score vs σ multiplier (log-x)
                                    (b) model rank vs σ multiplier, with the
                                        energy-rule ranking shown at the right edge
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
    MODEL_CONFIGS, CORPUS_CONFIGS, DEVICE,
    MAX_TOKENS, MAX_SEQ_LEN, MC_SAMPLES,
    KERNEL_N_SUBSAMPLE, RESULTS_DIR,
)
from utils.models import load_model_and_tokenizer, get_embedding_matrix
from utils.data import load_corpus
from scoring.rules import compute_kernel_bandwidth

# ── constants ────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(RESULTS_DIR, "exp6_bandwidth")

# Multipliers of each model's own median-heuristic sigma.  m = 1 reproduces
# the Exp 1 kernel setting; large m approaches energy-like behaviour.
SIGMA_MULTIPLIERS = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

MODEL_COLORS = {
    "gpt2":        "#1f77b4",
    "gpt2-medium": "#ff7f0e",
    "opt-125m":    "#2ca02c",
}
MODEL_LABELS = {
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2-med",
    "opt-125m":    "OPT-125m",
}


# ── core computation ─────────────────────────────────────────────────────────

def evaluate_sweep(model, tokenizer, embeddings, texts, sigma0,
                   device, max_tokens, n_samples, multipliers, label=""):
    """
    One inference pass; kernel score evaluated at every bandwidth multiplier.

    Returns:
      kernel_means : (len(multipliers),) — corpus-mean kernel score per multiplier
      energy_mean  : float               — corpus-mean energy score (reference)
      n_tokens     : int
    """
    mults    = torch.as_tensor(multipliers, dtype=torch.float32, device=device)
    sigmas   = mults * sigma0                                  # (M,)
    inv2s2   = 1.0 / (2.0 * sigmas ** 2)                       # (M,)

    # Running sums of per-token scores (avoid storing (N, M) matrices)
    kernel_sum  = torch.zeros(len(multipliers), dtype=torch.float64, device=device)
    energy_sum  = 0.0
    n_tokens    = 0

    pbar = tqdm(texts, desc=label, dynamic_ncols=True)
    with torch.no_grad():
        for text in pbar:
            if n_tokens >= max_tokens:
                break
            if not text.strip():
                continue

            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LEN, add_special_tokens=True).to(device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            outputs = model(**enc)
            logits  = outputs.logits[0, :-1, :]
            targets = input_ids[0, 1:]

            remaining = max_tokens - n_tokens
            logits    = logits[:remaining]
            targets   = targets[:remaining]
            T_prime   = logits.shape[0]

            probs = torch.softmax(logits, dim=-1)

            # MC draws shared by every sigma and by the energy reference
            s1 = torch.multinomial(probs, n_samples, replacement=True)  # (T', S)
            s2 = torch.multinomial(probs, n_samples, replacement=True)
            e1  = embeddings[s1]                       # (T', S, d)
            e2  = embeddings[s2]
            e_y = embeddings[targets].unsqueeze(1)     # (T', 1, d)

            # σ-independent squared distances
            d2_xy = ((e1 - e_y) ** 2).sum(dim=-1)      # (T', S)
            d2_xx = ((e1 - e2) ** 2).sum(dim=-1)       # (T', S)

            # Energy reference: E[d(X,y)] − ½ E[d(X,X')]
            energy = d2_xy.sqrt().mean(dim=-1) - 0.5 * d2_xx.sqrt().mean(dim=-1)
            energy_sum += energy.double().sum().item()

            # Kernel score at every multiplier in one broadcast:
            #   (M, 1, 1) × (T', S) → (M, T', S) → mean over S → (M, T')
            k_xy = torch.exp(-d2_xy.unsqueeze(0) * inv2s2.view(-1, 1, 1)).mean(dim=-1)
            k_xx = torch.exp(-d2_xx.unsqueeze(0) * inv2s2.view(-1, 1, 1)).mean(dim=-1)
            scores = 1.0 - 2.0 * k_xy + k_xx           # (M, T')
            kernel_sum += scores.double().sum(dim=-1)

            n_tokens += T_prime
            pbar.set_postfix({"tokens": n_tokens})

    kernel_means = (kernel_sum / n_tokens).cpu().numpy()
    energy_mean  = energy_sum / n_tokens
    return kernel_means, energy_mean, n_tokens


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_corpus(corpus_name, results, out_path):
    """
    results: model_name → dict(kernel_means, energy_mean, sigma0)
    2 panels: (a) mean kernel score vs multiplier; (b) rank vs multiplier.
    """
    models = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"Kernel Bandwidth Sweep  [{corpus_name}]   "
        f"(multiplier 1 = Exp 1 median-heuristic σ)",
        fontsize=12, fontweight="bold",
    )

    # ── (a) mean kernel score ────────────────────────────────────────────────
    ax = axes[0]
    for m in models:
        ax.plot(SIGMA_MULTIPLIERS, results[m]["kernel_means"],
                marker="o", markersize=4, linewidth=2,
                color=MODEL_COLORS[m], label=MODEL_LABELS[m])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("σ multiplier  (× median-heuristic σ₀)")
    ax.set_ylabel("Mean kernel score  (lower = better)")
    ax.set_title("(a) Score level")
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9)

    # ── (b) rank trajectories ────────────────────────────────────────────────
    ax = axes[1]
    score_mat = np.stack([results[m]["kernel_means"] for m in models])  # (n_models, M)
    # rank 1 = lowest score at each multiplier
    ranks = score_mat.argsort(axis=0).argsort(axis=0) + 1               # (n_models, M)

    for i, m in enumerate(models):
        ax.plot(SIGMA_MULTIPLIERS, ranks[i],
                marker="o", markersize=5, linewidth=2,
                color=MODEL_COLORS[m], label=MODEL_LABELS[m])

    # Energy-rule ranking shown as reference markers at the right edge
    energy_scores = np.array([results[m]["energy_mean"] for m in models])
    energy_ranks  = energy_scores.argsort().argsort() + 1
    x_ref = SIGMA_MULTIPLIERS[-1] * 1.6
    for i, m in enumerate(models):
        ax.scatter([x_ref], [energy_ranks[i]], marker="*", s=140,
                   color=MODEL_COLORS[m], zorder=5)
    ax.text(x_ref, 0.55, "energy\nrank", ha="center", va="top", fontsize=8,
            color="gray")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("σ multiplier")
    ax.set_ylabel("Rank under kernel rule  (1 = best)")
    ax.set_title("(b) Ranking vs bandwidth")
    ax.set_yticks(range(1, len(models) + 1))
    ax.invert_yaxis()
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9, loc="center left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}   # (model_name, corpus_name) → dict

    for model_name, model_id in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")

        all_cached = all(
            os.path.exists(os.path.join(OUTPUT_DIR, f"{model_name}__{c}.npz"))
            for c in CORPUS_CONFIGS
        )

        if all_cached:
            print(f"Model: {model_name}  [all corpora cached, skipping model load]")
            for corpus_name in CORPUS_CONFIGS:
                tag  = f"{model_name}__{corpus_name}"
                d    = np.load(os.path.join(OUTPUT_DIR, f"{tag}.npz"))
                all_results[(model_name, corpus_name)] = {
                    "kernel_means": d["kernel_means"],
                    "energy_mean":  float(d["energy_mean"]),
                    "sigma0":       float(d["sigma0"]),
                    "n_tokens":     int(d["n_tokens"]),
                }
            continue

        print(f"Loading model: {model_name}  ({model_id})")
        model, tokenizer = load_model_and_tokenizer(model_id, DEVICE)
        embeddings       = get_embedding_matrix(model, DEVICE)

        print("  Computing kernel bandwidth (median heuristic)...")
        sigma0 = compute_kernel_bandwidth(embeddings, KERNEL_N_SUBSAMPLE)
        print(f"  sigma0 = {sigma0:.4f}")

        for corpus_name, corpus_cfg in CORPUS_CONFIGS.items():
            tag        = f"{model_name}__{corpus_name}"
            cache_path = os.path.join(OUTPUT_DIR, f"{tag}.npz")

            if os.path.exists(cache_path):
                print(f"\n  Corpus: {corpus_name}  [cached]")
                d = np.load(cache_path)
                all_results[(model_name, corpus_name)] = {
                    "kernel_means": d["kernel_means"],
                    "energy_mean":  float(d["energy_mean"]),
                    "sigma0":       float(d["sigma0"]),
                    "n_tokens":     int(d["n_tokens"]),
                }
                continue

            print(f"\n  Corpus: {corpus_name}")
            texts = load_corpus(corpus_name, corpus_cfg)

            kernel_means, energy_mean, n_tokens = evaluate_sweep(
                model, tokenizer, embeddings, texts, sigma0,
                DEVICE, MAX_TOKENS, MC_SAMPLES, SIGMA_MULTIPLIERS,
                label=f"{model_name}/{corpus_name}",
            )

            np.savez_compressed(
                cache_path,
                kernel_means=kernel_means,
                energy_mean=energy_mean,
                sigma0=sigma0,
                n_tokens=n_tokens,
                multipliers=SIGMA_MULTIPLIERS,
            )
            all_results[(model_name, corpus_name)] = {
                "kernel_means": kernel_means,
                "energy_mean":  energy_mean,
                "sigma0":       sigma0,
                "n_tokens":     n_tokens,
            }

            print(f"  {n_tokens} tokens.  Kernel means across multipliers:")
            for mult, km in zip(SIGMA_MULTIPLIERS, kernel_means):
                print(f"    m={mult:6.3f}  σ={mult*sigma0:8.3f}  score={km:.5f}")
            print(f"  energy mean = {energy_mean:.5f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Tables ───────────────────────────────────────────────────────────────
    score_records, rank_records = [], []
    for corpus_name in CORPUS_CONFIGS:
        per_model = {
            m: all_results[(m, corpus_name)]
            for m in MODEL_CONFIGS if (m, corpus_name) in all_results
        }
        models    = list(per_model.keys())
        score_mat = np.stack([per_model[m]["kernel_means"] for m in models])
        ranks     = score_mat.argsort(axis=0).argsort(axis=0) + 1

        for i, m in enumerate(models):
            for j, mult in enumerate(SIGMA_MULTIPLIERS):
                score_records.append({
                    "corpus": corpus_name, "model": m,
                    "sigma_multiplier": mult,
                    "kernel_score": round(float(score_mat[i, j]), 6),
                })
                rank_records.append({
                    "corpus": corpus_name, "model": m,
                    "sigma_multiplier": mult,
                    "rank": int(ranks[i, j]),
                })

    df_scores = pd.DataFrame(score_records)
    df_ranks  = pd.DataFrame(rank_records)
    df_scores.to_csv(os.path.join(RESULTS_DIR, "exp6_sweep_scores.csv"), index=False)
    df_ranks.to_csv(os.path.join(RESULTS_DIR, "exp6_sweep_ranks.csv"), index=False)

    # ── Figures ──────────────────────────────────────────────────────────────
    for corpus_name in CORPUS_CONFIGS:
        per_model = {
            m: all_results[(m, corpus_name)]
            for m in MODEL_CONFIGS if (m, corpus_name) in all_results
        }
        plot_corpus(
            corpus_name, per_model,
            os.path.join(RESULTS_DIR, f"exp6_{corpus_name}.png"),
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 6 RESULTS")
    print("="*60)
    print("\nRank under kernel rule by σ multiplier (1 = best):")
    pivot = df_ranks.pivot_table(
        index=["corpus", "sigma_multiplier"], columns="model",
        values="rank", aggfunc="first")
    print(pivot.to_string())

    # Locate the crossing point for OPT
    print("\nOPT rank trajectory:")
    for corpus_name in CORPUS_CONFIGS:
        sub = df_ranks[(df_ranks["corpus"] == corpus_name) &
                       (df_ranks["model"] == "opt-125m")].sort_values("sigma_multiplier")
        traj = list(zip(sub["sigma_multiplier"], sub["rank"]))
        print(f"  {corpus_name}: {traj}")
        improving = [m for (m, r), (m2, r2) in zip(traj, traj[1:]) if r2 < r]
        if improving:
            print(f"    → rank improves after multiplier(s): {improving}")
            print(f"      OPT's errors sit just beyond the σ window at these scales —")
            print(f"      consistent with the middle-distance-band hypothesis.")
        else:
            print(f"    → rank never improves with σ: reversal is NOT a locality effect.")

    print(f"\nAll outputs written to  {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
