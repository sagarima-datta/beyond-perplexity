"""
Experiment 4: Bootstrap uncertainty for Experiment 1 rankings.

Reuses Experiment 1's per-token score caches and adds bootstrap uncertainty
estimates for the global ranking results:

Outputs
-------
  results/exp4_bootstrap_cell_ci.csv       - per-(model, corpus, rule) mean/SE/95% CI
  results/exp4_bootstrap_rank_agreement.csv - P(rule ranking matches log ranking)
  results/exp4_bootstrap_headline.csv      - OPT energy/log headline event probabilities
  results/exp4_bootstrap_kendall_tau.csv   - bootstrap mean/SE/95% CI for Kendall tau
  results/exp4_bootstrap_kendall_tau_mean_matrix_<corpus>.csv - lower-triangular tau matrix
  results/exp4_bootstrap_kendall_tau_se_matrix_<corpus>.csv - lower-triangular SE matrix
  results/exp4_score_ci_<corpus>.png       - mean score with 95% CI by rule
  results/exp4_kendall_tau_bootstrap_<corpus>.png - lower-triangular tau heatmap
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
from scipy.stats import kendalltau

from config import MODEL_CONFIGS, CORPUS_CONFIGS, RESULTS_DIR

TOKEN_SCORE_DIR = os.path.join(RESULTS_DIR, "exp1_token_scores")

RULE_NAMES = ["log", "quadratic", "crps", "energy", "kernel"]
REPORT_RULE_NAMES = [r for r in RULE_NAMES if r != "crps"]
RULE_LABELS = {
    "log":       "Log",
    "quadratic": "Quadratic",
    "energy":    "Energy",
    "kernel":    "Kernel",
}

MODEL_COLORS = {
    "gpt2":        "#1f77b4",
    "gpt2-medium": "#ff7f0e",
    "opt-125m":    "#2ca02c",
}

N_BOOT = 1_000
N_BOOT_TAU = 200
SAMPLE_TAU = 3_000
BOOT_BATCH_SIZE = 100
SEED = 591
SKIP_KENDALL_TAU = os.environ.get("SKIP_EXP4_TAU", "").lower() in {"1", "true", "yes"}


# -- cache loading ------------------------------------------------------------

def load_score_cache(model_name, corpus_name):
    """Load one Experiment 1 per-token score cache."""
    path = os.path.join(TOKEN_SCORE_DIR, f"{model_name}__{corpus_name}.npz")
    if not os.path.exists(path):
        return None

    with np.load(path) as data:
        missing = [r for r in RULE_NAMES if r not in data.files]
        if missing:
            raise ValueError(f"{path} is missing score arrays: {missing}")

        scores = {r: data[r].astype(float, copy=False) for r in RULE_NAMES}
    n = min(len(v) for v in scores.values())
    if any(len(v) != n for v in scores.values()):
        print(f"  [WARN] length mismatch in {path}; truncating all rules to {n}.")
        scores = {r: v[:n] for r, v in scores.items()}

    return scores


def load_all_score_caches():
    """Return {(model, corpus): {rule: scores}} for all available Exp 1 caches."""
    scores_by_cell = {}
    missing = []

    for model_name in MODEL_CONFIGS:
        for corpus_name in CORPUS_CONFIGS:
            scores = load_score_cache(model_name, corpus_name)
            if scores is None:
                missing.append((model_name, corpus_name))
                continue
            scores_by_cell[(model_name, corpus_name)] = scores
            n_tok = len(scores[RULE_NAMES[0]])
            print(f"Loaded {model_name}/{corpus_name}: {n_tok} token positions.")

    if missing:
        print("\nMissing Experiment 1 score caches:")
        for model_name, corpus_name in missing:
            path = os.path.join(TOKEN_SCORE_DIR, f"{model_name}__{corpus_name}.npz")
            print(f"  {path}")

    if not scores_by_cell:
        raise FileNotFoundError(
            "No Experiment 1 per-token caches found. Run "
            "`python -m experiments.exp1_global_ranking` first."
        )

    return scores_by_cell


# -- bootstrap cell means and rankings ---------------------------------------

def bootstrap_cell_means(scores_by_cell, n_boot=N_BOOT, seed=SEED):
    """
    Resample token positions within each (model, corpus) cell and return
    bootstrap means keyed by (model, corpus, rule).
    """
    rng = np.random.default_rng(seed)
    boot_means = {}

    for (model_name, corpus_name), scores in scores_by_cell.items():
        n = len(scores[RULE_NAMES[0]])
        print(f"\nBootstrapping means: {model_name}/{corpus_name}  (n={n})")

        cell_boots = {
            rule: np.empty(n_boot, dtype=np.float64)
            for rule in RULE_NAMES
        }

        for start in range(0, n_boot, BOOT_BATCH_SIZE):
            stop = min(start + BOOT_BATCH_SIZE, n_boot)
            idx = rng.integers(0, n, size=(stop - start, n))
            for rule in RULE_NAMES:
                cell_boots[rule][start:stop] = scores[rule][idx].mean(axis=1)

        for rule in RULE_NAMES:
            boot_means[(model_name, corpus_name, rule)] = cell_boots[rule]

    return boot_means


def cell_ci_table(boot_means, scores_by_cell):
    """Build the per-cell bootstrap mean/SE/CI table."""
    rows = []
    for (model_name, corpus_name, rule), boots in boot_means.items():
        if rule not in REPORT_RULE_NAMES:
            continue
        point = scores_by_cell[(model_name, corpus_name)][rule].mean()
        rows.append({
            "model":      model_name,
            "corpus":     corpus_name,
            "rule":       rule,
            "label":      RULE_LABELS[rule],
            "n_tokens":   len(scores_by_cell[(model_name, corpus_name)][rule]),
            "mean":       float(point),
            "boot_mean":  float(boots.mean()),
            "se":         float(boots.std(ddof=1)),
            "ci_lo":      float(np.percentile(boots, 2.5)),
            "ci_hi":      float(np.percentile(boots, 97.5)),
        })

    return pd.DataFrame(rows).sort_values(["corpus", "model", "rule"])


def rank_agreement_table(boot_means):
    """
    For each corpus and rule, compute the fraction of bootstrap replicates
    where that rule's model ordering exactly matches log-score's ordering.
    """
    rows = []
    corpora = sorted({corpus for _, corpus, _ in boot_means})

    for corpus_name in corpora:
        models = sorted({model for model, corpus, _ in boot_means
                         if corpus == corpus_name})
        if not all((m, corpus_name, "log") in boot_means for m in models):
            continue

        log_boots = np.stack([boot_means[(m, corpus_name, "log")]
                              for m in models])
        log_orders = log_boots.argsort(axis=0)

        for rule in REPORT_RULE_NAMES:
            if not all((m, corpus_name, rule) in boot_means for m in models):
                continue
            rule_boots = np.stack([boot_means[(m, corpus_name, rule)]
                                   for m in models])
            rule_orders = rule_boots.argsort(axis=0)
            agree = float((log_orders == rule_orders).all(axis=0).mean())
            rows.append({
                "corpus":              corpus_name,
                "rule":                rule,
                "label":               RULE_LABELS[rule],
                "p_rank_matches_log":  agree,
            })

    return pd.DataFrame(rows).sort_values(["corpus", "rule"])


def headline_event_table(boot_means):
    """
    Track the notebook's headline event for each available corpus:
    OPT is best under Energy and worst under Log.
    """
    opt_models = [m for m in MODEL_CONFIGS if "opt" in m.lower()]
    if not opt_models:
        return pd.DataFrame()

    opt_model = opt_models[0]
    rows = []
    corpora = sorted({corpus for _, corpus, _ in boot_means})

    for corpus_name in corpora:
        models = sorted({model for model, corpus, _ in boot_means
                         if corpus == corpus_name})
        if opt_model not in models:
            continue
        if not all((m, corpus_name, r) in boot_means
                   for m in models for r in ["log", "energy"]):
            continue

        opt_idx = models.index(opt_model)
        energy_boots = np.stack([boot_means[(m, corpus_name, "energy")]
                                 for m in models])
        log_boots = np.stack([boot_means[(m, corpus_name, "log")]
                              for m in models])

        opt_best_energy = energy_boots.argmin(axis=0) == opt_idx
        opt_worst_log = log_boots.argmax(axis=0) == opt_idx

        rows.append({
            "corpus":                         corpus_name,
            "model":                          opt_model,
            "p_opt_best_energy":              float(opt_best_energy.mean()),
            "p_opt_worst_log":                float(opt_worst_log.mean()),
            "p_best_energy_and_worst_log":    float((opt_best_energy & opt_worst_log).mean()),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["corpus"])


# -- bootstrap Kendall tau ----------------------------------------------------

def pooled_scores_by_corpus(scores_by_cell, corpus_name):
    """Concatenate per-token scores across models for one corpus."""
    pooled = {rule: [] for rule in REPORT_RULE_NAMES}

    for model_name in MODEL_CONFIGS:
        cell = scores_by_cell.get((model_name, corpus_name))
        if cell is None:
            continue
        for rule in REPORT_RULE_NAMES:
            pooled[rule].append(cell[rule])

    return {
        rule: np.concatenate(values)
        for rule, values in pooled.items()
        if values
    }


def bootstrap_kendall_tau_for_corpus(pooled, n_boot=N_BOOT_TAU,
                                     sample_tau=SAMPLE_TAU, seed=SEED):
    """
    Resample token positions with replacement and recompute the rule-pair
    Kendall tau matrix on each bootstrap replicate.
    """
    n_total = len(pooled[REPORT_RULE_NAMES[0]])
    n_sample = min(sample_tau, n_total)
    rng = np.random.default_rng(seed)

    tau_boots = np.full((n_boot, len(REPORT_RULE_NAMES), len(REPORT_RULE_NAMES)),
                        np.nan, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n_total, size=n_sample)
        for i, rule_i in enumerate(REPORT_RULE_NAMES):
            tau_boots[b, i, i] = 1.0
            for j in range(i + 1, len(REPORT_RULE_NAMES)):
                rule_j = REPORT_RULE_NAMES[j]
                tau, _ = kendalltau(pooled[rule_i][idx], pooled[rule_j][idx])
                tau_boots[b, i, j] = tau_boots[b, j, i] = float(tau)

    return tau_boots, n_sample


def kendall_tau_table(scores_by_cell):
    """Build the bootstrap Kendall tau table for every corpus."""
    rows = []
    tau_mats = {}

    for corpus_idx, corpus_name in enumerate(CORPUS_CONFIGS):
        pooled = pooled_scores_by_corpus(scores_by_cell, corpus_name)
        if not pooled:
            continue

        print(f"\nBootstrapping Kendall tau: {corpus_name}")
        tau_boots, n_sample = bootstrap_kendall_tau_for_corpus(
            pooled, seed=SEED + corpus_idx)
        tau_mean = np.nanmean(tau_boots, axis=0)
        tau_se = np.nanstd(tau_boots, axis=0, ddof=1)

        tau_mats[corpus_name] = (tau_mean, tau_se)

        for i, rule_i in enumerate(REPORT_RULE_NAMES):
            for j in range(i + 1, len(REPORT_RULE_NAMES)):
                rule_j = REPORT_RULE_NAMES[j]
                boots = tau_boots[:, i, j]
                rows.append({
                    "corpus":      corpus_name,
                    "rule_1":      rule_i,
                    "rule_2":      rule_j,
                    "tau_mean":    float(np.nanmean(boots)),
                    "tau_se":      float(np.nanstd(boots, ddof=1)),
                    "ci_lo":       float(np.nanpercentile(boots, 2.5)),
                    "ci_hi":       float(np.nanpercentile(boots, 97.5)),
                    "n_boot":      N_BOOT_TAU,
                    "n_sample":    n_sample,
                })

    df = pd.DataFrame(rows).sort_values(["corpus", "rule_1", "rule_2"])
    return df, tau_mats


def plot_tau_heatmap(corpus_name, tau_mean, tau_se):
    """Save a lower-triangular heatmap of bootstrap Kendall tau."""
    labels = [RULE_LABELS[r] for r in REPORT_RULE_NAMES]
    mean_df = pd.DataFrame(tau_mean, index=labels, columns=labels)
    se_df = pd.DataFrame(tau_se, index=labels, columns=labels)
    lower_mask = np.triu(np.ones_like(mean_df, dtype=bool), k=1)
    lower_mean = mean_df.mask(lower_mask)
    lower_se = se_df.mask(lower_mask)

    mean_csv = os.path.join(
        RESULTS_DIR, f"exp4_bootstrap_kendall_tau_mean_matrix_{corpus_name}.csv")
    se_csv = os.path.join(
        RESULTS_DIR, f"exp4_bootstrap_kendall_tau_se_matrix_{corpus_name}.csv")
    lower_mean.to_csv(mean_csv)
    lower_se.to_csv(se_csv)

    masked = np.ma.array(tau_mean, mask=np.triu(np.ones_like(tau_mean, dtype=bool), k=1))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(masked, vmin=-1, vmax=1, cmap="coolwarm")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Kendall's tau")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"Bootstrap Kendall's tau between rules [{corpus_name}]")

    for i in range(len(labels)):
        for j in range(i + 1):
            ax.text(j, i, f"{tau_mean[i, j]:.3f}\n+/-{tau_se[i, j]:.3f}",
                    ha="center", va="center", fontsize=8, color="black")

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    plt.tight_layout()

    png_path = os.path.join(RESULTS_DIR, f"exp4_kendall_tau_bootstrap_{corpus_name}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")


def plot_score_ci_by_corpus(corpus_name, ci_df):
    """Save a faceted mean-score plot with bootstrap 95% CIs."""
    sub = ci_df[ci_df["corpus"] == corpus_name]
    if sub.empty:
        return

    model_names = [m for m in MODEL_CONFIGS if m in set(sub["model"])]
    x = np.arange(len(model_names))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for ax, rule in zip(axes, REPORT_RULE_NAMES):
        rule_df = (
            sub[sub["rule"] == rule]
            .set_index("model")
            .reindex(model_names)
        )
        mean = rule_df["mean"].to_numpy(dtype=float)
        ci_lo = rule_df["ci_lo"].to_numpy(dtype=float)
        ci_hi = rule_df["ci_hi"].to_numpy(dtype=float)
        yerr = np.vstack([
            np.maximum(mean - ci_lo, 0.0),
            np.maximum(ci_hi - mean, 0.0),
        ])

        colors = [MODEL_COLORS.get(m, "#4c78a8") for m in model_names]
        ax.errorbar(x, mean, yerr=yerr, fmt="none", ecolor="#333333",
                    elinewidth=1.2, capsize=4, zorder=2)
        ax.scatter(x, mean, s=48, color=colors, edgecolor="black",
                   linewidth=0.5, zorder=3)

        ax.set_title(RULE_LABELS[rule], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=25, ha="right")
        ax.set_ylabel("Mean score")
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

    for ax in axes[len(REPORT_RULE_NAMES):]:
        ax.axis("off")

    fig.suptitle(f"Exp 4 - Bootstrap mean score 95% CI [{corpus_name}]",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = os.path.join(RESULTS_DIR, f"exp4_score_ci_{corpus_name}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")


# -- main ---------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    scores_by_cell = load_all_score_caches()

    boot_means = bootstrap_cell_means(scores_by_cell)
    ci_df = cell_ci_table(boot_means, scores_by_cell)
    agree_df = rank_agreement_table(boot_means)
    headline_df = headline_event_table(boot_means)

    ci_path = os.path.join(RESULTS_DIR, "exp4_bootstrap_cell_ci.csv")
    agree_path = os.path.join(RESULTS_DIR, "exp4_bootstrap_rank_agreement.csv")
    headline_path = os.path.join(RESULTS_DIR, "exp4_bootstrap_headline.csv")

    ci_df.to_csv(ci_path, index=False)
    agree_df.to_csv(agree_path, index=False)
    headline_df.to_csv(headline_path, index=False)

    for corpus_name in CORPUS_CONFIGS:
        plot_score_ci_by_corpus(corpus_name, ci_df)

    tau_path = os.path.join(RESULTS_DIR, "exp4_bootstrap_kendall_tau.csv")
    if SKIP_KENDALL_TAU:
        print("\nSKIP_EXP4_TAU=1; skipping bootstrap Kendall tau.")
    else:
        tau_df, tau_mats = kendall_tau_table(scores_by_cell)
        tau_df.to_csv(tau_path, index=False)

        for corpus_name, (tau_mean, tau_se) in tau_mats.items():
            plot_tau_heatmap(corpus_name, tau_mean, tau_se)

    print("\nBootstrap cell CIs:")
    print(ci_df.pivot_table(index=["model", "corpus"], columns="rule",
                            values="mean").round(4).to_string())

    print("\nP(rule ranking exactly matches log-score ranking):")
    print(agree_df.pivot(index="corpus", columns="rule",
                         values="p_rank_matches_log").round(3).to_string())

    if not headline_df.empty:
        print("\nOPT headline bootstrap events:")
        print(headline_df.round(3).to_string(index=False))

    print(f"\nSaved CSVs:")
    print(f"  {ci_path}")
    print(f"  {agree_path}")
    print(f"  {headline_path}")
    if not SKIP_KENDALL_TAU:
        print(f"  {tau_path}")


if __name__ == "__main__":
    main()
