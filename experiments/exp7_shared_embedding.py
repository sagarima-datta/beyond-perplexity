"""
Experiment 7: DistilBERT Shared-Embedding Check
================================================

Background
----------
The energy and kernel scores in Experiments 1–6 use each model's OWN input
embedding matrix.  OPT's #1 energy ranking could therefore reflect favourable
*geometry of its embedding space* (e.g. globally tighter clustering shrinks all
distances) rather than genuinely better-placed probability mass.

Hypothesis
----------
If OPT still ranks #1 on energy when all three models are scored in a single
SHARED embedding space — DistilBERT's, which none of them trained — then the
advantage is intrinsic to OPT's predicted distributions.  If OPT's rank drops,
the Exp 1 result was an artefact of its own embedding geometry ("home-field
advantage").

Method
------
1. For each evaluated model, build a shared-space embedding matrix
   E_shared ∈ R^(V_model × d_bert):
   every vocab token id is decoded to its surface string, re-tokenised with
   DistilBERT's WordPiece tokenizer, and the DistilBERT *input* embeddings of
   the resulting subwords are mean-pooled.  (Input embeddings only — no forward
   pass — so the map is deterministic and cheap.)
   Tokens that produce no subwords (rare control bytes) get the DistilBERT
   [UNK] embedding.

2. Run the usual inference pass per model × corpus, computing energy and
   kernel scores exactly as in Exp 1 but with E_shared instead of the model's
   own matrix.  Kernel bandwidth σ is re-estimated per model on its E_shared
   via the same median heuristic (the matrices differ slightly because the
   vocabs differ).

3. Compare model rankings: own-embedding (from Exp 1 results) vs shared.

Robustness extension (not implemented here): repeat step 2 fixing the space
   to each evaluated model's own embedding matrix in turn, mapped via surface
   strings as above.  If rankings agree across all fixed spaces, the
   conclusion is geometry-independent.

Outputs
-------
  results/exp7_shared/
      shared_emb_<model>.pt           cached shared-space embedding matrices
      <model>__<corpus>.npz           per-token energy & kernel scores (shared space)
  results/exp7_shared_summary.csv     mean scores + own vs shared ranks
  results/exp7_shared_<corpus>.png    grouped bar chart: rank under
                                      own-embedding energy/kernel vs shared
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
from scoring.rules import (
    energy_score_chunk,
    kernel_score_chunk,
    compute_kernel_bandwidth,
)

# ── constants ────────────────────────────────────────────────────────────────

OUTPUT_DIR     = os.path.join(RESULTS_DIR, "exp7_shared")
SHARED_MODEL   = "distilbert-base-uncased"
EMB_BATCH_SIZE = 512     # vocab tokens per batch when building the shared matrix

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


# ── shared-space embedding construction ──────────────────────────────────────

def build_shared_embedding_matrix(model_tokenizer, bert_tokenizer, bert_emb,
                                  device, vocab_size=None, cache_path=None):
    """
    Map every token of `model_tokenizer`'s vocab into DistilBERT's input
    embedding space by surface string.

    `vocab_size` must match the MODEL's output dimension (its embedding-matrix
    row count), not len(tokenizer): OPT pads its vocab (50272 logits vs 50265
    tokenizer entries), and sampled ids can land in the padded range.  Padded
    ids map to DistilBERT's [UNK] vector.

    Returns E_shared : (vocab_size, d_bert) float32 tensor on `device`.
    """
    if vocab_size is None:
        vocab_size = len(model_tokenizer)

    if cache_path is not None and os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location=device)
        if cached.shape[0] == vocab_size:
            print(f"  [cached shared matrix: {cache_path}]")
            return cached
        print(f"  [cached matrix has {cached.shape[0]} rows, need {vocab_size} "
              f"— rebuilding]")

    tok_vocab_size = len(model_tokenizer)
    d_bert     = bert_emb.shape[1]
    unk_id     = bert_tokenizer.unk_token_id
    E_shared   = torch.empty(vocab_size, d_bert, dtype=torch.float32, device=device)

    # Ids beyond the tokenizer's range (vocab padding) have no surface string
    if vocab_size > tok_vocab_size:
        E_shared[tok_vocab_size:] = bert_emb[unk_id]

    print(f"  Building shared matrix: {vocab_size} tokens → {SHARED_MODEL} space...")
    for start in tqdm(range(0, tok_vocab_size, EMB_BATCH_SIZE),
                      desc="  shared-emb", dynamic_ncols=True):
        ids = list(range(start, min(start + EMB_BATCH_SIZE, tok_vocab_size)))
        # Decode each vocab id individually so BPE space-markers (Ġ, ▁) are
        # resolved to plain surface strings.
        strings = model_tokenizer.batch_decode(
            [[i] for i in ids], clean_up_tokenization_spaces=False)

        # WordPiece-encode every surface string (no special tokens)
        enc = bert_tokenizer(
            strings, add_special_tokens=False,
            padding=True, return_tensors="pt",
        )
        sub_ids  = enc["input_ids"].to(device)        # (B, L)
        sub_mask = enc["attention_mask"].to(device)   # (B, L)

        # Mean-pool input embeddings over real subwords
        sub_emb = bert_emb[sub_ids]                            # (B, L, d)
        mask    = sub_mask.unsqueeze(-1).float()               # (B, L, 1)
        summed  = (sub_emb * mask).sum(dim=1)                  # (B, d)
        counts  = mask.sum(dim=1).clamp(min=1.0)               # (B, 1)
        pooled  = summed / counts

        # Tokens that produced zero subwords → [UNK] embedding
        empty = (sub_mask.sum(dim=1) == 0)
        if empty.any():
            pooled[empty] = bert_emb[unk_id]

        E_shared[start:start + len(ids)] = pooled

    if cache_path is not None:
        torch.save(E_shared.cpu(), cache_path)
        E_shared = E_shared.to(device)
    return E_shared


# ── scoring pass ─────────────────────────────────────────────────────────────

def evaluate_shared(model, tokenizer, shared_emb, sigma, texts,
                    device, max_tokens, n_samples, label=""):
    """
    Standard Exp 1-style inference pass, but energy & kernel only, scored in
    the shared embedding space.

    Returns dict: {"energy": (N,), "kernel": (N,)} numpy arrays.
    """
    energy_list, kernel_list = [], []
    tokens_processed = 0

    pbar = tqdm(texts, desc=label, dynamic_ncols=True)
    with torch.no_grad():
        for text in pbar:
            if tokens_processed >= max_tokens:
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

            remaining = max_tokens - tokens_processed
            logits    = logits[:remaining]
            targets   = targets[:remaining]

            probs = torch.softmax(logits, dim=-1)

            energy_list.append(
                energy_score_chunk(probs, targets, shared_emb, n_samples))
            kernel_list.append(
                kernel_score_chunk(probs, targets, shared_emb, sigma, n_samples))

            tokens_processed += logits.shape[0]
            pbar.set_postfix({"tokens": tokens_processed})

    return {
        "energy": np.concatenate(energy_list).astype(np.float32),
        "kernel": np.concatenate(kernel_list).astype(np.float32),
    }


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_rank_comparison(corpus_name, df_summary, out_path):
    """
    Grouped bar chart: for each model, four bars —
      energy rank (own emb), energy rank (shared), kernel rank (own), kernel (shared).
    """
    sub    = df_summary[df_summary["corpus"] == corpus_name]
    models = list(MODEL_CONFIGS.keys())
    n      = len(models)

    cols = [
        ("energy_rank_own",    "Energy (own emb)",    "//"),
        ("energy_rank_shared", "Energy (DistilBERT)", None),
        ("kernel_rank_own",    "Kernel (own emb)",    "//"),
        ("kernel_rank_shared", "Kernel (DistilBERT)", None),
    ]
    bar_w = 0.19
    x     = np.arange(n)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for j, (col, lbl, hatch) in enumerate(cols):
        vals = [int(sub[sub["model"] == m][col].iloc[0]) for m in models]
        base_color = "#7f7f7f" if "energy" in col else "#bcbd22"
        ax.bar(x + (j - 1.5) * bar_w, vals, bar_w,
               label=lbl, color=base_color,
               alpha=0.55 if hatch else 0.95, hatch=hatch,
               edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Rank  (1 = best)")
    ax.set_yticks(range(1, n + 1))
    ax.invert_yaxis()
    ax.set_title(f"Own-embedding vs shared-embedding rankings  [{corpus_name}]")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load DistilBERT input embeddings once ───────────────────────────────
    print(f"Loading shared embedder: {SHARED_MODEL}")
    from transformers import AutoModel, AutoTokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(SHARED_MODEL)
    bert_model     = AutoModel.from_pretrained(SHARED_MODEL)
    bert_emb       = (bert_model.embeddings.word_embeddings
                      .weight.detach().to(DEVICE))
    del bert_model
    print(f"  DistilBERT vocab {bert_emb.shape[0]}, dim {bert_emb.shape[1]}")

    all_scores = {}   # (model_name, corpus_name) → {"energy": ..., "kernel": ...}

    for model_name, model_id in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")

        all_cached = all(
            os.path.exists(os.path.join(OUTPUT_DIR, f"{model_name}__{c}.npz"))
            for c in CORPUS_CONFIGS
        )

        if all_cached:
            print(f"Model: {model_name}  [all corpora cached, skipping model load]")
            for corpus_name in CORPUS_CONFIGS:
                d = np.load(os.path.join(OUTPUT_DIR, f"{model_name}__{corpus_name}.npz"))
                all_scores[(model_name, corpus_name)] = {k: d[k] for k in d}
            continue

        print(f"Loading model: {model_name}  ({model_id})")
        model, tokenizer = load_model_and_tokenizer(model_id, DEVICE)

        # Size the shared matrix from the model's true output dim, not
        # len(tokenizer) — OPT pads its vocab beyond the tokenizer range.
        model_vocab_size = get_embedding_matrix(model, DEVICE).shape[0]

        shared_emb = build_shared_embedding_matrix(
            tokenizer, bert_tokenizer, bert_emb, DEVICE,
            vocab_size=model_vocab_size,
            cache_path=os.path.join(OUTPUT_DIR, f"shared_emb_{model_name}.pt"),
        )

        print("  Computing kernel bandwidth on shared matrix...")
        sigma = compute_kernel_bandwidth(shared_emb, KERNEL_N_SUBSAMPLE)
        print(f"  sigma = {sigma:.4f}")

        for corpus_name, corpus_cfg in CORPUS_CONFIGS.items():
            tag        = f"{model_name}__{corpus_name}"
            cache_path = os.path.join(OUTPUT_DIR, f"{tag}.npz")

            if os.path.exists(cache_path):
                print(f"\n  Corpus: {corpus_name}  [cached]")
                d = np.load(cache_path)
                all_scores[(model_name, corpus_name)] = {k: d[k] for k in d}
                continue

            print(f"\n  Corpus: {corpus_name}")
            texts  = load_corpus(corpus_name, corpus_cfg)
            scores = evaluate_shared(
                model, tokenizer, shared_emb, sigma, texts,
                DEVICE, MAX_TOKENS, MC_SAMPLES,
                label=f"{model_name}/{corpus_name}(shared)",
            )
            np.savez_compressed(cache_path, **scores)
            all_scores[(model_name, corpus_name)] = scores

            print(f"  {len(scores['energy'])} positions.")
            print(f"    energy (shared): {scores['energy'].mean():.5f}")
            print(f"    kernel (shared): {scores['kernel'].mean():.5f}")

        del model, shared_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Own-embedding ranks from Exp 1 token scores ──────────────────────────
    exp1_dir = os.path.join(RESULTS_DIR, "exp1_token_scores")
    own_means = {}   # (model, corpus) → {"energy": float, "kernel": float}
    for model_name in MODEL_CONFIGS:
        for corpus_name in CORPUS_CONFIGS:
            path = os.path.join(exp1_dir, f"{model_name}__{corpus_name}.npz")
            if os.path.exists(path):
                d = np.load(path)
                own_means[(model_name, corpus_name)] = {
                    "energy": float(d["energy"].mean()),
                    "kernel": float(d["kernel"].mean()),
                }

    # ── Summary table with ranks ─────────────────────────────────────────────
    records = []
    for corpus_name in CORPUS_CONFIGS:
        models = [m for m in MODEL_CONFIGS if (m, corpus_name) in all_scores]

        shared_energy = {m: float(all_scores[(m, corpus_name)]["energy"].mean())
                         for m in models}
        shared_kernel = {m: float(all_scores[(m, corpus_name)]["kernel"].mean())
                         for m in models}

        def ranks_of(d):
            order = sorted(d, key=d.get)          # ascending: lower = better
            return {m: order.index(m) + 1 for m in d}

        r_e_shared = ranks_of(shared_energy)
        r_k_shared = ranks_of(shared_kernel)

        have_own = all((m, corpus_name) in own_means for m in models)
        if have_own:
            r_e_own = ranks_of({m: own_means[(m, corpus_name)]["energy"] for m in models})
            r_k_own = ranks_of({m: own_means[(m, corpus_name)]["kernel"] for m in models})
        else:
            r_e_own = r_k_own = {m: np.nan for m in models}

        for m in models:
            records.append({
                "model": m, "corpus": corpus_name,
                "energy_mean_shared": round(shared_energy[m], 5),
                "kernel_mean_shared": round(shared_kernel[m], 5),
                "energy_rank_own":    r_e_own[m],
                "energy_rank_shared": r_e_shared[m],
                "kernel_rank_own":    r_k_own[m],
                "kernel_rank_shared": r_k_shared[m],
            })

    df_summary = pd.DataFrame(records)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "exp7_shared_summary.csv"), index=False)

    # ── Figures ──────────────────────────────────────────────────────────────
    for corpus_name in CORPUS_CONFIGS:
        plot_rank_comparison(
            corpus_name, df_summary,
            os.path.join(RESULTS_DIR, f"exp7_shared_{corpus_name}.png"),
        )

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 7 RESULTS")
    print("="*60)
    print("\nSummary:")
    print(df_summary.to_string(index=False))

    print("\nVerdict on the home-field-advantage hypothesis:")
    for corpus_name in CORPUS_CONFIGS:
        sub = df_summary[(df_summary["corpus"] == corpus_name) &
                         (df_summary["model"] == "opt-125m")]
        if sub.empty:
            continue
        row = sub.iloc[0]
        own, shared = row["energy_rank_own"], row["energy_rank_shared"]
        if pd.isna(own):
            print(f"  {corpus_name}: own-embedding ranks unavailable "
                  f"(run exp1 first); OPT shared energy rank = {shared}")
        elif shared == 1:
            print(f"  {corpus_name}: OPT keeps energy rank #1 in DistilBERT space "
                  f"(own #{int(own)} → shared #{int(shared)})")
            print(f"    → advantage is INTRINSIC to OPT's predicted distributions.")
        else:
            print(f"  {corpus_name}: OPT drops from energy rank #{int(own)} (own) "
                  f"to #{int(shared)} (shared)")
            print(f"    → the Exp 1 energy win was at least partly an artefact of "
                  f"OPT's own embedding geometry.")

    print(f"\nAll outputs written to  {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
