"""
Robustness check — bootstrap confidence intervals on Experiment 1 rankings.

Experiment 1 reports a single point estimate for every (model, corpus, rule)
cell (the mean of MAX_TOKENS per-token scores).  This script asks: are the
headline rankings actually stable under resampling, or could a different
token sample have produced a different ordering?

For each (model, corpus) cell we draw `N_BOOT` non-parametric bootstrap
resamples of the per-token score arrays cached by exp1, recompute the
five rule means on each replicate, and derive three artefacts:

    1. results/robustness_bootstrap_means.csv
         per-(model, corpus, rule) bootstrap mean, SE, and 95% percentile CI

    2. results/robustness_bootstrap_rank_agreement.csv
         per-(corpus, rule) fraction of replicates whose full model ranking
         exactly matches the log-score ranking.  Values near 0 certify that
         the cross-rule disagreement in exp1 is statistically robust.

    3. results/robustness_bootstrap_headline.csv
         For each corpus, all (rule, model) pairs whose bootstrap-best or
         bootstrap-worst model differs from log's, plus the joint probability
         of the "flip" event.  Headline cross-rule disagreements emerge
         data-driven, without hardcoding any particular model name.

    4. results/robustness_bootstrap_<corpus>.png
         Forest plot — one panel per rule, models on the x-axis, mean and
         95% CI as error bars.

Prerequisite
------------
Experiment 1 must have been run first.  This script reads the per-token
score caches from `results/exp1_token_scores/` and never loads a model.

    python -m experiments.exp1_global_ranking
    python -m experiments.robustness_bootstrap
"""

import os, sys
# Force UTF-8 output on Windows consoles
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

from config import (
    MODEL_CONFIGS, CORPUS_CONFIGS,
    N_BOOT, BOOT_SEED, RESULTS_DIR,
)

TOKEN_SCORE_DIR = os.path.join(RESULTS_DIR, "exp1_token_scores")
RULE_NAMES      = ["log", "quadratic", "crps", "energy", "kernel"]

MODEL_COLORS = {
    "gpt2":        "#1f77b4",
    "gpt2-medium": "#ff7f0e",
    "opt-125m":    "#2ca02c",
}

RULE_COLORS = {
    "log":       "#e41a1c",
    "quadratic": "#377eb8",
    "crps":      "#4daf4a",
    "energy":    "#ff7f00",
    "kernel":    "#984ea3",
}


# ── bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_cell_means(scores_dict, n_boot, rng):
    """
    Resample with replacement WITHIN one (model, corpus) cell and recompute
    each rule's mean on every replicate.

    Parameters
    ----------
    scores_dict : dict[str, np.ndarray]  rule_name → per-token scores (N,)
    n_boot      : int                    number of bootstrap replicates
    rng         : np.random.Generator

    Returns
    -------
    dict[str, np.ndarray]  rule_name → bootstrap means (n_boot,)
    """
    # All rules share the same N (same token positions in exp1), so a single
    # index matrix suffices and the rules' resamples stay row-aligned.
    n = len(next(iter(scores_dict.values())))
    boot_idx = rng.integers(0, n, size=(n_boot, n))    # (n_boot, n)
    return {
        rule: arr[boot_idx].mean(axis=1)               # (n_boot,)
        for rule, arr in scores_dict.items()
    }


# ── rank agreement ──────────────────────────────────────────────────────────

def rank_agreement_vs_log(boot_means_by_cell, corpus_name, model_names):
    """
    For one corpus, compute P( rule's full ranking == log's full ranking )
    over the bootstrap replicates, for each rule.
    """
    log_stack = np.stack([
        boot_means_by_cell[(m, corpus_name, "log")] for m in model_names
    ])                                                 # (M, N_BOOT)
    log_orders = log_stack.argsort(axis=0)             # best (lowest) first

    rows = []
    for rule in RULE_NAMES:
        rule_stack = np.stack([
            boot_means_by_cell[(m, corpus_name, rule)] for m in model_names
        ])
        rule_orders = rule_stack.argsort(axis=0)
        agree = float((log_orders == rule_orders).all(axis=0).mean())
        rows.append({
            "corpus": corpus_name,
            "rule":   rule,
            "p_rank_equals_log": round(agree, 4),
        })
    return rows


# ── headline disagreements (data-driven) ────────────────────────────────────

def headline_disagreements(boot_means_by_cell, corpus_name, model_names):
    """
    For each non-log rule, check whether the bootstrap-best (argmin) or
    bootstrap-worst (argmax) model differs from log's, on the point-estimate.
    If so, report the joint bootstrap probability of the "flip" — i.e.
    P( model X is best under rule R  AND  model X is worst under log )
    or the symmetric worst-vs-best version.

    This generalises the user-notebook's hardcoded "OPT best under Energy AND
    OPT worst under Log on WikiText" story to whichever (rule, model, corpus)
    combination this setup actually produces.
    """
    # Point-estimate means: just average each bootstrap-mean vector
    point = {
        (m, rule): float(boot_means_by_cell[(m, corpus_name, rule)].mean())
        for m in model_names for rule in RULE_NAMES
    }

    def best(rule):  # argmin model
        return min(model_names, key=lambda m: point[(m, rule)])
    def worst(rule):  # argmax model
        return max(model_names, key=lambda m: point[(m, rule)])

    log_best  = best("log")
    log_worst = worst("log")

    log_stack = np.stack([
        boot_means_by_cell[(m, corpus_name, "log")] for m in model_names
    ])
    log_best_per_rep  = log_stack.argmin(axis=0)
    log_worst_per_rep = log_stack.argmax(axis=0)

    rows = []
    for rule in RULE_NAMES:
        if rule == "log":
            continue

        rule_best  = best(rule)
        rule_worst = worst(rule)

        rule_stack = np.stack([
            boot_means_by_cell[(m, corpus_name, rule)] for m in model_names
        ])
        rule_best_per_rep  = rule_stack.argmin(axis=0)
        rule_worst_per_rep = rule_stack.argmax(axis=0)

        # Flip type 1: a model that is best under `rule` is worst under log.
        # We score this only when the point estimates already show that flip
        # (otherwise the joint probability is trivially small).
        if rule_best == log_worst and rule_best != log_best:
            idx = model_names.index(rule_best)
            p_best_rule  = float((rule_best_per_rep  == idx).mean())
            p_worst_log  = float((log_worst_per_rep  == idx).mean())
            p_joint      = float(((rule_best_per_rep  == idx) &
                                  (log_worst_per_rep  == idx)).mean())
            rows.append({
                "corpus": corpus_name,
                "rule":   rule,
                "model":  rule_best,
                "flip":   f"best under {rule} AND worst under log",
                "p_best_under_rule":   round(p_best_rule, 4),
                "p_worst_under_log":   round(p_worst_log, 4),
                "p_joint":             round(p_joint, 4),
            })

        # Flip type 2: a model that is worst under `rule` is best under log.
        if rule_worst == log_best and rule_worst != log_worst:
            idx = model_names.index(rule_worst)
            p_worst_rule = float((rule_worst_per_rep == idx).mean())
            p_best_log   = float((log_best_per_rep   == idx).mean())
            p_joint      = float(((rule_worst_per_rep == idx) &
                                  (log_best_per_rep   == idx)).mean())
            rows.append({
                "corpus": corpus_name,
                "rule":   rule,
                "model":  rule_worst,
                "flip":   f"worst under {rule} AND best under log",
                "p_worst_under_rule":  round(p_worst_rule, 4),
                "p_best_under_log":    round(p_best_log, 4),
                "p_joint":             round(p_joint, 4),
            })

    return rows


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_forest(corpus_name, ci_df, model_names):
    """
    Forest plot: one panel per rule, models on the x-axis, mean ± 95% CI.
    Each rule lives on its own y-axis (the five rules have wildly different
    natural scales — log ~ 4, brier ~ 0.8, crps ~ 1e3, energy ~ 1, kernel ~ 0.2).
    """
    fig, axes = plt.subplots(1, len(RULE_NAMES), figsize=(3.4 * len(RULE_NAMES), 4.5))
    if len(RULE_NAMES) == 1:
        axes = [axes]

    sub = ci_df[ci_df["corpus"] == corpus_name]
    x = np.arange(len(model_names))

    for ax, rule in zip(axes, RULE_NAMES):
        rsub = sub[sub["rule"] == rule].set_index("model").reindex(model_names)
        means = rsub["mean"].values.astype(float)
        lo    = rsub["ci_lo"].values.astype(float)
        hi    = rsub["ci_hi"].values.astype(float)
        err   = np.vstack([means - lo, hi - means])

        colors = [MODEL_COLORS.get(m, "#444444") for m in model_names]
        for xi, mi, lo_i, hi_i, c in zip(x, means, lo, hi, colors):
            ax.errorbar(xi, mi, yerr=[[mi - lo_i], [hi_i - mi]],
                        fmt="o", color=c, capsize=4, markersize=7,
                        linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
        ax.set_title(rule, color=RULE_COLORS[rule], fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        if rule == RULE_NAMES[0]:
            ax.set_ylabel("Mean score (lower = better)", fontsize=10)

    fig.suptitle(
        f"Robustness check — bootstrap 95% CIs on Experiment 1  [{corpus_name}]",
        fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.isdir(TOKEN_SCORE_DIR):
        print(f"[ERROR] Token-score cache directory not found: {TOKEN_SCORE_DIR}")
        print("        Run experiment 1 first:")
        print("        python -m experiments.exp1_global_ranking")
        sys.exit(1)

    rng = np.random.default_rng(BOOT_SEED)

    # ── Load all available .npz caches ───────────────────────────────────────
    boot_means_by_cell = {}    # (model, corpus, rule) → (N_BOOT,)
    ci_rows = []

    available_cells = []
    for model_name in MODEL_CONFIGS:
        for corpus_name in CORPUS_CONFIGS:
            path = os.path.join(TOKEN_SCORE_DIR,
                                f"{model_name}__{corpus_name}.npz")
            if os.path.exists(path):
                available_cells.append((model_name, corpus_name, path))

    if not available_cells:
        print(f"[ERROR] No .npz caches found in {TOKEN_SCORE_DIR}")
        print("        Run experiment 1 first.")
        sys.exit(1)

    print(f"Bootstrapping with N_BOOT={N_BOOT}, seed={BOOT_SEED}")
    print(f"Found {len(available_cells)} cached (model, corpus) cells.\n")

    for model_name, corpus_name, path in available_cells:
        data = np.load(path)
        scores = {r: data[r] for r in RULE_NAMES if r in data.files}
        if len(scores) != len(RULE_NAMES):
            missing = set(RULE_NAMES) - set(scores)
            print(f"  [WARN] {model_name}__{corpus_name}: missing rules {missing}, skipping.")
            continue

        n_tokens = len(scores["log"])
        print(f"  {model_name:12s}  {corpus_name:12s}  n={n_tokens}")

        boots = bootstrap_cell_means(scores, N_BOOT, rng)

        for rule, b in boots.items():
            boot_means_by_cell[(model_name, corpus_name, rule)] = b
            ci_rows.append({
                "model":  model_name,
                "corpus": corpus_name,
                "rule":   rule,
                "mean":   round(float(b.mean()), 6),
                "se":     round(float(b.std(ddof=1)), 6),
                "ci_lo":  round(float(np.percentile(b, 2.5)), 6),
                "ci_hi":  round(float(np.percentile(b, 97.5)), 6),
                "n_tokens": int(n_tokens),
            })

    ci_df = pd.DataFrame(ci_rows)

    # Models / corpora actually present in the cache
    models_present  = sorted({m for (m, _, _) in boot_means_by_cell})
    corpora_present = sorted({c for (_, c, _) in boot_means_by_cell})

    # ── Rank agreement vs log score ─────────────────────────────────────────
    agree_rows = []
    for corpus_name in corpora_present:
        cell_models = sorted({m for (m, c, _) in boot_means_by_cell if c == corpus_name})
        if len(cell_models) < 2:
            print(f"  [WARN] corpus {corpus_name}: only {len(cell_models)} model(s) — "
                  f"rank-agreement is degenerate, skipping.")
            continue
        agree_rows.extend(rank_agreement_vs_log(
            boot_means_by_cell, corpus_name, cell_models))
    agree_df = pd.DataFrame(agree_rows)

    # ── Headline disagreements ──────────────────────────────────────────────
    head_rows = []
    for corpus_name in corpora_present:
        cell_models = sorted({m for (m, c, _) in boot_means_by_cell if c == corpus_name})
        if len(cell_models) < 2:
            continue
        head_rows.extend(headline_disagreements(
            boot_means_by_cell, corpus_name, cell_models))
    head_df = pd.DataFrame(head_rows)

    # ── Save CSVs ────────────────────────────────────────────────────────────
    ci_path     = os.path.join(RESULTS_DIR, "robustness_bootstrap_means.csv")
    agree_path  = os.path.join(RESULTS_DIR, "robustness_bootstrap_rank_agreement.csv")
    head_path   = os.path.join(RESULTS_DIR, "robustness_bootstrap_headline.csv")

    ci_df.to_csv(ci_path, index=False)
    agree_df.to_csv(agree_path, index=False)
    head_df.to_csv(head_path, index=False)

    # ── Plots ────────────────────────────────────────────────────────────────
    for corpus_name in corpora_present:
        cell_models = sorted({m for (m, c, _) in boot_means_by_cell if c == corpus_name})
        fig = plot_forest(corpus_name, ci_df, cell_models)
        png = os.path.join(RESULTS_DIR, f"robustness_bootstrap_{corpus_name}.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {png}")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ROBUSTNESS BOOTSTRAP — SUMMARY")
    print("="*60)

    print(f"\nBootstrap means and 95% CIs (N_BOOT={N_BOOT}):")
    pivot = ci_df.pivot_table(
        index=["model", "corpus"], columns="rule",
        values=["mean", "ci_lo", "ci_hi"])
    print(pivot.round(4).to_string())

    if not agree_df.empty:
        print("\nP(rule's full ranking == log's ranking), per corpus:")
        pa = agree_df.pivot(index="corpus", columns="rule",
                            values="p_rank_equals_log")
        print(pa.round(3).to_string())
        print("\nValues near 0 ⇒ this rule essentially never agrees with log's")
        print("ranking on a bootstrap replicate — strong cross-rule disagreement.")

    if not head_df.empty:
        print("\nHeadline cross-rule flips (data-driven, no hardcoded models):")
        print(head_df.to_string(index=False))
    else:
        print("\nNo headline cross-rule flips detected in the point-estimate ranks.")

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"  - {ci_path}")
    print(f"  - {agree_path}")
    print(f"  - {head_path}")


if __name__ == "__main__":
    main()
