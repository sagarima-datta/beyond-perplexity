"""
Experiment 5: OPT Energy Advantage — Centroid-Distance Analysis
================================================================

Background
----------
From Experiment 1, OPT-125m ranks #1 under the energy scoring rule on both
corpora (mean energy ≈ 0.84 on WikiText-103) yet ranks #3 under every other
rule, including log-score (mean ≈ 4.05).  GPT-2-medium, by contrast, ranks
#1 on log-score but only #2 on energy.

Hypothesis
----------
OPT makes "distributionally close" errors: even when it assigns the wrong
token as its argmax (a log-score failure), the full probability mass sits
geometrically close to the correct token in its own embedding space.
This is precisely what the energy score rewards — it measures expected distance
from a random draw to the true token, not the probability assigned to the
exact true token.

Method
------
At every evaluated token position t we compute the probability-weighted centroid
of the predicted distribution in the model's own embedding space:

    c_t = Σ_v p(v | context) · e_v  =  probs_t @ E          (V × d multiply)

and its Euclidean distance to the true token's embedding:

    d_c(t) = ‖ c_t − e(y_t) ‖

We then compare d_c distributions between models, focusing on positions where
the model's top-1 prediction is wrong (log-score failures).

Smoking-gun prediction: on wrong positions, OPT's centroid-distance distribution
should be shifted LEFT (closer to truth) relative to GPT-2 variants.

We also compute, for each position:
  • top1_dist : ‖ e(argmax p) − e(y) ‖   — semantic distance of the argmax error
  • dispersion : mean pairwise distance among MC draws from p  (≈ spread of p in emb space)

This lets us distinguish two mechanisms:
  A. OPT spreads mass broadly but centred near truth  → low centroid_dist, high dispersion
  B. OPT's top-1 is semantically near truth           → low top1_dist

Outputs
-------
  results/exp5_centroid/
      <model>__<corpus>.npz      per-token arrays:
                                   centroid_dist, top1_dist, dispersion,
                                   wrong (bool: top-1 ≠ true),
                                   log_score
  results/exp5_summary.csv       mean centroid_dist / top1_dist on wrong positions,
                                 per (model, corpus)
  results/exp5_<corpus>.png      3-panel figure per corpus:
                                   (a) violin of centroid_dist on wrong positions
                                   (b) CDF of centroid_dist on wrong positions
                                   (c) scatter: centroid_dist vs log_score (sample)
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
    MAX_TOKENS, MAX_SEQ_LEN, MC_SAMPLES, RESULTS_DIR,
)
from utils.models import load_model_and_tokenizer, get_embedding_matrix
from utils.data import load_corpus

# ── constants ────────────────────────────────────────────────────────────────

OUTPUT_DIR  = os.path.join(RESULTS_DIR, "exp5_centroid")
SEED        = 591
SCATTER_N   = 2_000    # points shown per model in the scatter panel
DISPERSION_SAMPLES = 200   # MC draws for dispersion estimate

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

def evaluate_centroid(model, tokenizer, embeddings, texts,
                      device, max_tokens, n_disp_samples, label=""):
    """
    Pass over `texts` and compute per-token centroid statistics.

    Returns a dict with numpy arrays:
      centroid_dist : ‖ probs @ E − e_y ‖             (N,)  float32
      top1_dist     : ‖ e(argmax) − e_y ‖             (N,)  float32
      dispersion    : MC estimate of E[‖X−X'‖]/2       (N,)  float32
      wrong         : argmax ≠ true token              (N,)  bool
      log_score     : −log p(y)                        (N,)  float32
    """
    centroid_dist_list = []
    top1_dist_list     = []
    dispersion_list    = []
    wrong_list         = []
    log_score_list     = []
    tokens_processed   = 0

    pbar = tqdm(texts, desc=label, dynamic_ncols=True)
    with torch.no_grad():
        for text in pbar:
            if tokens_processed >= max_tokens:
                break
            if not text.strip():
                continue

            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LEN, add_special_tokens=True).to(device)
            input_ids = enc["input_ids"]           # (1, T)
            if input_ids.shape[1] < 2:
                continue

            outputs   = model(**enc)
            logits    = outputs.logits[0, :-1, :]  # (T-1, V)
            targets   = input_ids[0, 1:]           # (T-1,)

            # Trim to remaining budget
            remaining = max_tokens - tokens_processed
            logits    = logits[:remaining]          # (T', V)
            targets   = targets[:remaining]         # (T',)

            T_prime = logits.shape[0]
            probs   = torch.softmax(logits, dim=-1)  # (T', V)

            # ── 1. Centroid distance: ‖ E[e(X)] − e(y) ‖ ──────────────────
            # centroid shape: (T', d)
            centroid = probs @ embeddings           # exact weighted mean embedding
            e_y      = embeddings[targets]          # (T', d)
            c_dist   = (centroid - e_y).norm(dim=-1).cpu().numpy().astype(np.float32)

            # ── 2. Top-1 distance: ‖ e(argmax) − e(y) ‖ ──────────────────
            top1     = logits.argmax(dim=-1)        # (T',)
            e_top1   = embeddings[top1]             # (T', d)
            t1_dist  = (e_top1 - e_y).norm(dim=-1).cpu().numpy().astype(np.float32)

            # ── 3. Dispersion: ½ · E[‖X − X'‖] via MC paired draws ────────
            s1 = torch.multinomial(probs, n_disp_samples, replacement=True)  # (T', S)
            s2 = torch.multinomial(probs, n_disp_samples, replacement=True)
            e1 = embeddings[s1]                    # (T', S, d)
            e2 = embeddings[s2]
            disp = 0.5 * (e1 - e2).norm(dim=-1).mean(dim=-1)  # (T',)
            disp = disp.cpu().numpy().astype(np.float32)

            # ── 4. Wrong (top-1 != true token) ───────────────────────────
            wr = (top1 != targets).cpu().numpy()   # (T',)

            # ── 5. Log-score: −log p(y) ───────────────────────────────────
            p_y = probs[torch.arange(T_prime, device=device), targets]
            ls  = -torch.log(p_y.clamp(min=1e-40)).cpu().numpy().astype(np.float32)

            centroid_dist_list.append(c_dist)
            top1_dist_list.append(t1_dist)
            dispersion_list.append(disp)
            wrong_list.append(wr)
            log_score_list.append(ls)

            tokens_processed += T_prime
            pbar.set_postfix({"tokens": tokens_processed})

    return {
        "centroid_dist": np.concatenate(centroid_dist_list),
        "top1_dist":     np.concatenate(top1_dist_list),
        "dispersion":    np.concatenate(dispersion_list),
        "wrong":         np.concatenate(wrong_list),
        "log_score":     np.concatenate(log_score_list),
    }


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_corpus(corpus_name, per_model_data, out_path):
    """
    3-panel figure:
      (a) Violin of centroid_dist on wrong positions across models
      (b) CDF of centroid_dist on wrong positions
      (c) Scatter: centroid_dist vs log_score (sample of SCATTER_N points each)
    """
    rng     = np.random.default_rng(SEED)
    models  = list(per_model_data.keys())
    colors  = [MODEL_COLORS[m] for m in models]
    labels  = [MODEL_LABELS[m] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"OPT Energy Advantage — Centroid Analysis  [{corpus_name}]",
        fontsize=13, fontweight="bold",
    )

    # ── Panel (a): Violin ────────────────────────────────────────────────────
    ax = axes[0]
    violin_data = []
    for m in models:
        d = per_model_data[m]
        violin_data.append(d["centroid_dist"][d["wrong"]])
    vp = ax.violinplot(violin_data, positions=range(len(models)), showmedians=True,
                       showextrema=False)
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.65)
    vp["cmedians"].set_color("black")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Centroid distance  ‖ E[e(X)] − e(y) ‖")
    ax.set_title("(a) Distribution on wrong positions")

    # annotate medians
    for i, (m, data) in enumerate(zip(models, violin_data)):
        med = np.median(data)
        ax.text(i, med + 0.01 * ax.get_ylim()[1], f"{med:.3f}",
                ha="center", va="bottom", fontsize=8)

    # ── Panel (b): CDF ───────────────────────────────────────────────────────
    ax = axes[1]
    x_max = max(
        np.percentile(per_model_data[m]["centroid_dist"][per_model_data[m]["wrong"]], 95)
        for m in models
    )
    x_grid = np.linspace(0, x_max, 500)
    for m, col, lbl in zip(models, colors, labels):
        d = per_model_data[m]
        vals = np.sort(d["centroid_dist"][d["wrong"]])
        cdf  = np.searchsorted(vals, x_grid, side="right") / len(vals)
        ax.plot(x_grid, cdf, color=col, linewidth=2, label=lbl)
    ax.set_xlabel("Centroid distance")
    ax.set_ylabel("CDF")
    ax.set_title("(b) CDF on wrong positions")
    ax.legend(fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

    # ── Panel (c): Scatter centroid_dist vs log_score ────────────────────────
    ax = axes[2]
    for m, col, lbl in zip(models, colors, labels):
        d  = per_model_data[m]
        # subsample wrong positions only for readability
        idx_wrong = np.where(d["wrong"])[0]
        n   = min(SCATTER_N, len(idx_wrong))
        idx = rng.choice(idx_wrong, n, replace=False)
        ax.scatter(d["centroid_dist"][idx], d["log_score"][idx],
                   color=col, alpha=0.25, s=6, label=lbl)
    ax.set_xlabel("Centroid distance")
    ax.set_ylabel("Log-score  (−log p(y))")
    ax.set_title("(c) Centroid dist vs log-score\n(wrong positions only)")
    ax.legend(fontsize=9, markerscale=3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def plot_mechanism(corpus_name, per_model_data, out_path):
    """
    Mechanism diagnostic figure:
      2-panel: centroid_dist vs dispersion scatter (all positions) per model,
    showing whether the distribution is tightly peaked near truth (low centroid,
    low dispersion) or broad but centred near truth (low centroid, high dispersion).
    """
    rng    = np.random.default_rng(SEED + 1)
    models = list(per_model_data.keys())
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS[m] for m in models]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharey=True)
    fig.suptitle(
        f"Centroid Distance vs Dispersion  [{corpus_name}]",
        fontsize=12, fontweight="bold",
    )

    for ax, m, col, lbl in zip(axes, models, colors, labels):
        d = per_model_data[m]
        wrong = d["wrong"]
        # subsample
        idx   = rng.choice(len(d["centroid_dist"]), min(2000, len(d["centroid_dist"])),
                            replace=False)
        sc = ax.scatter(
            d["dispersion"][idx],
            d["centroid_dist"][idx],
            c=[col if wrong[i] else "#cccccc" for i in idx],
            alpha=0.3, s=5,
        )
        ax.set_xlabel("Dispersion  ½·E[‖X−X'‖]")
        ax.set_title(lbl)
    axes[0].set_ylabel("Centroid distance  ‖ E[e(X)] − e(y) ‖")

    # legend patches
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#888888", alpha=0.5, label="correct (top-1 = true)"),
        Patch(facecolor="#dd4444", alpha=0.5, label="wrong (top-1 ≠ true)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Inference pass — one model at a time ────────────────────────────────
    all_data = {}   # (model_name, corpus_name) → dict of arrays

    for model_name, model_id in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")

        # Check if all corpora already cached for this model
        all_cached = all(
            os.path.exists(os.path.join(OUTPUT_DIR, f"{model_name}__{c}.npz"))
            for c in CORPUS_CONFIGS
        )

        if all_cached:
            print(f"Model: {model_name}  [all corpora cached, skipping model load]")
            for corpus_name in CORPUS_CONFIGS:
                tag  = f"{model_name}__{corpus_name}"
                data = np.load(os.path.join(OUTPUT_DIR, f"{tag}.npz"))
                all_data[(model_name, corpus_name)] = {k: data[k] for k in data}
            continue

        print(f"Loading model: {model_name}  ({model_id})")
        model, tokenizer = load_model_and_tokenizer(model_id, DEVICE)
        embeddings       = get_embedding_matrix(model, DEVICE)

        for corpus_name, corpus_cfg in CORPUS_CONFIGS.items():
            tag        = f"{model_name}__{corpus_name}"
            cache_path = os.path.join(OUTPUT_DIR, f"{tag}.npz")

            if os.path.exists(cache_path):
                print(f"\n  Corpus: {corpus_name}  [cached]")
                data = np.load(cache_path)
                all_data[(model_name, corpus_name)] = {k: data[k] for k in data}
                continue

            print(f"\n  Corpus: {corpus_name}")
            texts = load_corpus(corpus_name, corpus_cfg)
            print(f"  Loaded {len(texts)} documents.")

            data = evaluate_centroid(
                model, tokenizer, embeddings, texts,
                DEVICE, MAX_TOKENS, DISPERSION_SAMPLES,
                label=f"{model_name}/{corpus_name}",
            )

            np.savez_compressed(cache_path, **data)
            all_data[(model_name, corpus_name)] = data

            n    = len(data["centroid_dist"])
            n_wr = data["wrong"].sum()
            print(f"  {n} positions, {n_wr} wrong ({100*n_wr/n:.1f}%)")
            print(f"    centroid_dist (wrong): mean={data['centroid_dist'][data['wrong']].mean():.4f}  "
                  f"median={np.median(data['centroid_dist'][data['wrong']]):.4f}")
            print(f"    top1_dist     (wrong): mean={data['top1_dist'][data['wrong']].mean():.4f}  "
                  f"median={np.median(data['top1_dist'][data['wrong']]):.4f}")
            print(f"    dispersion    (all  ): mean={data['dispersion'].mean():.4f}")

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXPERIMENT 5 SUMMARY")
    print("="*60)

    records = []
    for (model_name, corpus_name), data in all_data.items():
        wr = data["wrong"]
        n  = len(wr)
        records.append({
            "model":               model_name,
            "corpus":              corpus_name,
            "n_positions":         int(n),
            "pct_wrong":           round(100 * wr.sum() / n, 2),
            # centroid_dist on wrong positions
            "centroid_dist_mean":  round(float(data["centroid_dist"][wr].mean()), 5),
            "centroid_dist_med":   round(float(np.median(data["centroid_dist"][wr])), 5),
            # top1_dist on wrong positions
            "top1_dist_mean":      round(float(data["top1_dist"][wr].mean()), 5),
            "top1_dist_med":       round(float(np.median(data["top1_dist"][wr])), 5),
            # dispersion (all positions — measures how spread each model's dist is)
            "dispersion_mean":     round(float(data["dispersion"].mean()), 5),
            "dispersion_med":      round(float(np.median(data["dispersion"])), 5),
        })

    df_summary = pd.DataFrame(records).set_index(["model", "corpus"])
    df_summary.to_csv(os.path.join(RESULTS_DIR, "exp5_summary.csv"))
    print("\nSummary (sorted by centroid_dist_mean on wrong positions):")
    print(df_summary.sort_values("centroid_dist_mean").to_string())

    # ── Figures ─────────────────────────────────────────────────────────────
    for corpus_name in CORPUS_CONFIGS:
        per_model = {
            m: all_data[(m, corpus_name)]
            for m in MODEL_CONFIGS
            if (m, corpus_name) in all_data
        }

        plot_corpus(
            corpus_name, per_model,
            os.path.join(RESULTS_DIR, f"exp5_{corpus_name}.png"),
        )
        plot_mechanism(
            corpus_name, per_model,
            os.path.join(RESULTS_DIR, f"exp5_{corpus_name}_mechanism.png"),
        )

    # ── Narrative interpretation ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("""
Panel (a) / summary table — centroid_dist on wrong positions:
  IF OPT centroid_dist << GPT-2 centroid_dist
    → Hypothesis CONFIRMED: OPT spreads mass near the true token even
      when its argmax is wrong.  Energy rewards this; log-score does not.

  IF centroid_dist ≈ equal across models
    → The advantage is NOT about where the mass sits, but about the
      geometry of OPT's embedding space itself (proceed to Exp 7 / shared-
      embedding check).

Panel (c) — scatter centroid_dist vs log_score:
  For OPT you expect a NEGATIVE correlation on wrong positions: high log-score
  errors (model very unsure about true token) go with LOW centroid_dist
  (mass still near truth).  For GPT-2 this correlation should be weaker or
  absent.

Panel (mechanism) — centroid_dist vs dispersion:
  Low centroid, high dispersion → broad but centred distribution  (Mechanism A)
  Low centroid, low dispersion  → peaked near correct token       (Mechanism B)
  High centroid, any dispersion → mass pointing away from truth
""")

    print(f"\nAll outputs written to  {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
