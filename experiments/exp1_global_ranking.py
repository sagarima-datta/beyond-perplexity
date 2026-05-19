"""
Experiment 1: Global score comparison and ranking.

Produces:
  results/exp1_scores.csv       — per-(model, corpus) mean scores for all 5 rules
  results/exp1_rankings.csv     — model rank (1=best) under each rule
  results/exp1_kendall_tau.csv  — pairwise Kendall τ between rules at token level
  results/exp1_token_scores/    — per-token score arrays (for downstream experiments)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau
from tqdm import tqdm

from config import (
    MODEL_CONFIGS, CORPUS_CONFIGS, DEVICE,
    MAX_TOKENS, MAX_SEQ_LEN, CHUNK_SIZE, MC_SAMPLES,
    KERNEL_N_SUBSAMPLE, RESULTS_DIR,
)
from utils.models import load_model_and_tokenizer, get_embedding_matrix
from utils.data import load_corpus, build_frequency_rank
from scoring.rules import (
    log_score_chunk,
    quadratic_score_chunk,
    crps_chunk,
    energy_score_chunk,
    kernel_score_chunk,
    compute_kernel_bandwidth,
)

RULE_NAMES = ["log", "quadratic", "crps", "energy", "kernel"]
TOKEN_SCORE_DIR = os.path.join(RESULTS_DIR, "exp1_token_scores")


# ── helpers ──────────────────────────────────────────────────────────────────

def collect_all_token_ids(texts, tokenizer, max_seq_len, max_tokens):
    """First pass: collect all target token ids to build frequency ranks."""
    ids = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_seq_len, add_special_tokens=True)
        toks = enc["input_ids"][0]
        if len(toks) < 2:
            continue
        ids.extend(toks[1:].tolist())           # predict from position 1 onward
        if len(ids) >= max_tokens * 3:          # collect more than needed for freq stats
            break
    return np.array(ids, dtype=np.int64)


def evaluate(model, tokenizer, embeddings, texts,
             freq_rank, sorted_by_freq, sigma,
             device, max_tokens, chunk_size, n_samples, label=""):
    """
    Iterate over texts, run the model, and compute all 5 scores per token.
    Returns a dict: rule_name → np.ndarray of per-token scores.
    """
    all_scores = {r: [] for r in RULE_NAMES}
    tokens_processed = 0

    freq_rank_dev    = freq_rank.to(device)
    sorted_by_freq_dev = sorted_by_freq.to(device)

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

            outputs = model(**enc)
            logits  = outputs.logits[0, :-1, :]   # (T-1, V)  — predict next token
            targets = input_ids[0, 1:]             # (T-1,)

            T = logits.shape[0]

            # Process in chunks to limit memory for CRPS
            for start in range(0, T, chunk_size):
                if tokens_processed >= max_tokens:
                    break
                end = min(start + chunk_size, T)

                lg_chunk  = logits[start:end]      # (chunk, V)
                tgt_chunk = targets[start:end]     # (chunk,)
                probs     = torch.softmax(lg_chunk, dim=-1)

                all_scores["log"].append(
                    log_score_chunk(probs, tgt_chunk))
                all_scores["quadratic"].append(
                    quadratic_score_chunk(probs, tgt_chunk))
                all_scores["crps"].append(
                    crps_chunk(probs, tgt_chunk, freq_rank_dev, sorted_by_freq_dev))
                all_scores["energy"].append(
                    energy_score_chunk(probs, tgt_chunk, embeddings, n_samples))
                all_scores["kernel"].append(
                    kernel_score_chunk(probs, tgt_chunk, embeddings, sigma, n_samples))

                tokens_processed += (end - start)

            pbar.set_postfix({"tokens": tokens_processed})

    return {r: np.concatenate(v) for r, v in all_scores.items() if v}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TOKEN_SCORE_DIR, exist_ok=True)

    mean_scores = {}   # (model_name, corpus_name) → {rule: mean_score}

    for model_name, model_id in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}  ({model_id})")
        model, tokenizer = load_model_and_tokenizer(model_id, DEVICE)
        embeddings = get_embedding_matrix(model, DEVICE)
        vocab_size = embeddings.shape[0]

        # Median-heuristic bandwidth for kernel score (once per model)
        print("  Computing kernel bandwidth (median heuristic)…")
        sigma = compute_kernel_bandwidth(embeddings, KERNEL_N_SUBSAMPLE)
        print(f"  σ = {sigma:.4f}")

        for corpus_name, corpus_cfg in CORPUS_CONFIGS.items():
            print(f"\n  Corpus: {corpus_name}")
            texts = load_corpus(corpus_name, corpus_cfg)
            print(f"  Loaded {len(texts)} documents.")

            # Build frequency ranks from the corpus (first pass over token ids)
            print("  Building token frequency ranks…")
            all_ids = collect_all_token_ids(
                texts, tokenizer, MAX_SEQ_LEN, MAX_TOKENS)
            freq_rank, sorted_by_freq = build_frequency_rank(all_ids, vocab_size)

            # Main evaluation pass
            label = f"{model_name}/{corpus_name}"
            scores = evaluate(
                model, tokenizer, embeddings, texts,
                freq_rank, sorted_by_freq, sigma,
                DEVICE, MAX_TOKENS, CHUNK_SIZE, MC_SAMPLES, label=label,
            )

            # Persist per-token arrays for downstream experiments
            tag = f"{model_name}__{corpus_name}"
            np.savez_compressed(
                os.path.join(TOKEN_SCORE_DIR, f"{tag}.npz"), **scores)

            mean_scores[(model_name, corpus_name)] = {
                r: float(np.mean(v)) for r, v in scores.items()
            }
            n_tok = len(next(iter(scores.values())))
            print(f"  Evaluated {n_tok} token positions.")
            for r, m in mean_scores[(model_name, corpus_name)].items():
                print(f"    {r:12s}: {m:.5f}")

        # Free GPU memory before loading next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Build results tables ─────────────────────────────────────────────────

    # Table 1: mean scores  (rows = model × corpus, cols = rules)
    records = []
    for (model_name, corpus_name), rule_scores in mean_scores.items():
        row = {"model": model_name, "corpus": corpus_name}
        row.update(rule_scores)
        records.append(row)
    df_scores = pd.DataFrame(records).set_index(["model", "corpus"])

    # Table 2: rankings (1 = best = lowest score) per corpus
    ranking_records = []
    for corpus_name in CORPUS_CONFIGS:
        sub = df_scores.xs(corpus_name, level="corpus")
        for rule in RULE_NAMES:
            ranked = sub[rule].rank().astype(int)
            for model_name in MODEL_CONFIGS:
                ranking_records.append({
                    "corpus": corpus_name,
                    "rule": rule,
                    "model": model_name,
                    "rank": ranked[model_name],
                })
    df_rankings = pd.DataFrame(ranking_records)

    # Table 3: pairwise Kendall τ between rules (pooled over all token positions)
    tau_records = []
    for corpus_name in CORPUS_CONFIGS:
        # Load all token score arrays for this corpus
        token_arrays = {}
        for model_name in MODEL_CONFIGS:
            tag = f"{model_name}__{corpus_name}"
            path = os.path.join(TOKEN_SCORE_DIR, f"{tag}.npz")
            if not os.path.exists(path):
                continue
            data = np.load(path)
            for rule in RULE_NAMES:
                token_arrays.setdefault(rule, []).append(data[rule])

        # Concatenate across models (compare rule behavior at token level)
        pooled = {r: np.concatenate(v) for r, v in token_arrays.items() if v}

        for i, r1 in enumerate(RULE_NAMES):
            for r2 in RULE_NAMES[i+1:]:
                if r1 not in pooled or r2 not in pooled:
                    continue
                # Subsample for speed (Kendall τ is O(n log n))
                n = min(len(pooled[r1]), 10_000)
                idx = np.random.choice(len(pooled[r1]), n, replace=False)
                tau, pval = kendalltau(pooled[r1][idx], pooled[r2][idx])
                tau_records.append({
                    "corpus": corpus_name,
                    "rule_1": r1, "rule_2": r2,
                    "kendall_tau": round(tau, 4),
                    "p_value": round(pval, 6),
                })

    df_tau = pd.DataFrame(tau_records)

    # ── Save ─────────────────────────────────────────────────────────────────
    df_scores.to_csv(os.path.join(RESULTS_DIR, "exp1_scores.csv"))
    df_rankings.to_csv(os.path.join(RESULTS_DIR, "exp1_rankings.csv"), index=False)
    df_tau.to_csv(os.path.join(RESULTS_DIR, "exp1_kendall_tau.csv"), index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 1 RESULTS")
    print("="*60)
    print("\nMean scores (lower = better):")
    print(df_scores.to_string())
    print("\nModel rankings by rule (1 = best):")
    pivot = df_rankings.pivot_table(
        index=["corpus", "rule"], columns="model", values="rank", aggfunc="first")
    print(pivot.to_string())
    print("\nPairwise Kendall τ between rules (token level):")
    print(df_tau.to_string(index=False))

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
