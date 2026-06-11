# Beyond Perplexity: Evaluating LLM Token Distributions with Proper Scoring Rules

Code accompanying the paper *Beyond Perplexity: Evaluating LLM Token Distributions with Proper Scoring Rules* (`report/report.pdf`). Evaluates GPT-2, GPT-2-Medium, and OPT-125M on WikiText-103 and Penn Treebank (PTB) using strictly proper scoring rules: log-score, quadratic (Brier), energy score, and kernel score (MMD).

CRPS is also computed and cached, but it is excluded from reported tables and figures.

---

## Repository layout

```
beyond-perplexity/
├── config.py                    # Models, corpora, and all hyperparameters
├── requirements.txt
├── scoring/
│   └── rules.py                 # All scoring rules (log, quadratic, CRPS, energy, kernel)
├── utils/
│   ├── data.py                  # Corpus loaders (WikiText-103, PTB via NLTK)
│   └── models.py                # Model + tokenizer loading, embedding extraction
├── experiments/
│   ├── exp1_global_ranking.py   # Experiment 1 — global score comparison & model ranking
│   ├── exp2_token_frequency.py  # Experiment 2 — scores by token frequency decile
│   ├── exp3_entropy_bins.py     # Experiment 3 — scores by prediction entropy
│   ├── exp4_bootstrap_intervals.py # Experiment 4 — bootstrap uncertainty for Exp 1 rankings
│   └── exp5_opt_energy_advantage.py # Experiment 5 — centroid analysis of the energy reversal
├── results/                     # All outputs land here (created automatically)
│   ├── *.csv                    # Numeric results
│   └── *.png                    # Plots
└── report/
    ├── report.tex               # Paper source (NeurIPS 2025 format)
    ├── report.pdf               # Compiled paper
    ├── neurips_2025.sty         # Style file
    └── figures/                 # Figures included in the paper
```

> **Binary caches** (`results/exp1_token_scores/*.npz`, `results/exp5_centroid/*.npz`, entropy `.npy` files, etc.) are excluded from git via `.gitignore`. They are generated automatically on first run and reused on subsequent runs.

---

## Setup

### 1. Prerequisites

- Python 3.10 or later  
- No GPU required — all experiments run on CPU (slower but correct)  
- A GPU with ≥ 8 GB VRAM speeds things up ~10–20×

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

On Windows, use `py -m pip install -r requirements.txt` if `pip` is not on PATH.

### 3. Internet access

The first run downloads model weights and datasets from HuggingFace automatically. The PTB corpus is downloaded via NLTK (also automatic). Subsequent runs use the local cache.

| Model | Size |
|---|---|
| GPT-2 (117M) | ~500 MB |
| GPT-2-Medium (345M) | ~1.4 GB |
| OPT-125M | ~500 MB |

---

## Running the experiments

Run all experiments as Python modules from the **root of the repository**.

### Experiment 1 — Global score comparison and model ranking

```bash
python -m experiments.exp1_global_ranking
```

Outputs:
- `results/exp1_scores.csv` — mean scores per (model, corpus, rule)
- `results/exp1_rankings.csv` — model rank per (corpus, rule)
- `results/exp1_kendall_tau.csv` — pairwise Kendall τ between all rule pairs
- `results/exp1_kendall_tau_matrix_<corpus>.csv`
- `results/exp1_kendall_tau_matrix_<corpus>.png`
- `results/exp1_token_scores/<model>__<corpus>.npz` — per-token score cache (reused by Exp 2–4)

### Experiment 2 — Score behaviour by token frequency

> **Requires Experiment 1 to have been run first.**

```bash
python -m experiments.exp2_token_frequency
```

Groups each evaluated token by its unigram frequency decile (0 = most common, 9 = rarest) and plots each rule's score relative to decile 0.

Outputs:
- `results/exp2_stratified_scores.csv`
- `results/exp2_wikitext103.png`
- `results/exp2_ptb.png`

### Experiment 3 — Score behaviour by prediction entropy

> **Requires Experiment 1 to have been run first.**

```bash
python -m experiments.exp3_entropy_bins
```

Groups each token position by the Shannon entropy H(p) of the model's softmax distribution (low / medium / high tertiles) and plots each rule's score relative to the low-entropy bin.

Outputs:
- `results/exp3_entropy_bins.csv`
- `results/exp3_wikitext103.png`
- `results/exp3_ptb.png`

### Experiment 4 — Bootstrap uncertainty for Experiment 1 rankings

> **Requires Experiment 1 to have been run first.**

```bash
python -m experiments.exp4_bootstrap_intervals
```

Resamples token positions within each `(model, corpus)` cell to quantify uncertainty in Experiment 1's mean scores and model rankings. It also bootstraps the per-token Kendall τ matrix between scoring rules.

Outputs:
- `results/exp4_bootstrap_cell_ci.csv`
- `results/exp4_bootstrap_rank_agreement.csv`
- `results/exp4_bootstrap_headline.csv`
- `results/exp4_bootstrap_kendall_tau.csv`
- `results/exp4_bootstrap_kendall_tau_mean_matrix_<corpus>.csv`
- `results/exp4_bootstrap_kendall_tau_se_matrix_<corpus>.csv`
- `results/exp4_score_ci_<corpus>.png`
- `results/exp4_kendall_tau_bootstrap_<corpus>.png`

To generate only the score/CI tables and plots, set `SKIP_EXP4_TAU=1`.

### Experiment 5 — Centroid analysis of the OPT energy reversal

```bash
python -m experiments.exp5_opt_energy_advantage
```

Explains why OPT-125M ranks first under the energy score while ranking last under the log-score. At every scored position it computes the probability-weighted centroid of the predicted distribution in the model's own embedding space and its Euclidean distance to the true token's embedding. Runs its own forward passes, so it does not depend on Experiment 1's cache.

Outputs:
- `results/exp5_summary.csv` — mean centroid and top-1 distances on wrongly predicted positions
- `results/exp5_wikitext103.png`, `results/exp5_ptb.png` — violin + CDF figures
- `results/exp5_centroid/<model>__<corpus>.npz` — per-position distance cache

---

## Running all experiments in order

```bash
python -m experiments.exp1_global_ranking
python -m experiments.exp2_token_frequency
python -m experiments.exp3_entropy_bins
python -m experiments.exp4_bootstrap_intervals
python -m experiments.exp5_opt_energy_advantage
```

Each experiment checks for cached results and skips model inference if already computed. Re-running is safe and fast after the first pass.

---

## Configuration

All key settings live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_MODE` | `False` | `True` = fast sanity-check run (2 000 tokens, 50 docs, validation splits) |
| `MAX_TOKENS` | `20_000` | Token positions evaluated per (model, corpus) — the setting used in the paper |
| `MAX_SEQ_LEN` | `512` | Maximum context window / truncation length |
| `MC_SAMPLES` | `200` | Monte Carlo draws for energy and kernel score expectations |
| `DEVICE` | auto | `"cuda"` if a GPU is available, else `"cpu"` |

### Quick sanity-check run

```python
# in config.py
SAMPLE_MODE = True   # 2 000 tokens, validation split, 50 docs max
```

Sample-mode outputs are written to `results_sample/` so they never overwrite paper-scale results.

### Paper-scale run (defaults)

```python
# in config.py
SAMPLE_MODE = False   # 20 000 tokens per (model, corpus), test splits
```

All results in the paper were produced with these defaults.

---

## Caching and resuming

Experiment 1 writes its per-token scores to `results/exp1_token_scores/` as `.npz` files; Experiment 5 caches per-position distances in `results/exp5_centroid/`. If a run is interrupted, simply rerun the same command — it will resume without repeating model inference.

To force a full rerun, delete the relevant cache:

```bash
# Re-run Experiment 1 from scratch for all models
rm results/exp1_token_scores/*.npz

# Re-run only for GPT-2
rm results/exp1_token_scores/gpt2__*.npz
```

---

## Expected runtimes (CPU, `MAX_TOKENS = 20_000`)

| Experiment | Approx. time |
|---|---|
| Exp 1 (all 3 models × 2 corpora) | 4–8 hours |
| Exp 2 | < 5 minutes (reads Exp 1 cache only) |
| Exp 3 | 1–2 hours first run (entropy forward passes); < 5 minutes once cached |
| Exp 4 | < 10 minutes (reads Exp 1 cache only) |
| Exp 5 (own forward passes) | 1–2 hours |

With a GPU these times drop by roughly 10–20×.

---

## Paper

The full write-up lives in `report/`. To rebuild the PDF:

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # second pass resolves references
```
