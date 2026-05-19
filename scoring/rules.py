"""
All five proper scoring rules adapted to the discrete token setting.

Convention: every score is *negatively oriented* (lower = better, like a loss),
matching Gneiting & Raftery (2007).

All functions accept:
  probs   : (N, V) float32 tensor — softmax probabilities
  targets : (N,)   int64  tensor  — indices of realized tokens
and return:
  scores  : (N,)   numpy float64 array
"""

import numpy as np
import torch


# ── 1. Log-score (NLL) ───────────────────────────────────────────────────────

def log_score_chunk(probs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """−log p(y).  Associated divergence: KL."""
    p_y = probs[torch.arange(len(targets), device=probs.device), targets]
    return -torch.log(p_y.clamp(min=1e-40)).cpu().numpy()


# ── 2. Quadratic / Brier score ───────────────────────────────────────────────

def quadratic_score_chunk(probs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """‖p − e_y‖² = ‖p‖² − 2p_y + 1.  Associated divergence: L² distance."""
    p_y = probs[torch.arange(len(targets), device=probs.device), targets]
    brier = (probs ** 2).sum(dim=-1) - 2.0 * p_y + 1.0
    return brier.cpu().numpy()


# ── 3. CRPS (discrete, frequency-rank ordered) ────────────────────────────────

def crps_chunk(
    probs: torch.Tensor,
    targets: torch.Tensor,
    freq_rank: torch.Tensor,       # (V,)  token_id → frequency rank (0 = most frequent)
    sorted_indices: torch.Tensor,  # (V,)  argsort of tokens by frequency
) -> np.ndarray:
    """
    Adapted CRPS for a discrete distribution ordered by unigram frequency rank.

    CRPS(P, y) = Σ_t (F_P(t) − 𝟏{rank(y) ≤ t})²

    where F_P is the CDF of P over the frequency-ranked vocabulary.
    Associated divergence: Cramér–von Mises distance.
    """
    N, V = probs.shape
    device = probs.device

    # Reorder columns by frequency rank (most frequent first)
    probs_sorted = probs[:, sorted_indices]          # (N, V)

    # CDF along the frequency-ranked axis
    cdf = probs_sorted.cumsum(dim=-1)                # (N, V)

    # Rank of each target token
    target_ranks = freq_rank[targets].long()         # (N,)

    # Indicator: indicator[i, t] = 1  iff  t >= target_rank[i]
    t_idx = torch.arange(V, device=device).unsqueeze(0)   # (1, V)
    indicator = (t_idx >= target_ranks.unsqueeze(1)).float()  # (N, V)

    crps = ((cdf - indicator) ** 2).sum(dim=-1)      # (N,)
    return crps.cpu().numpy()


# ── 4. Energy score ──────────────────────────────────────────────────────────

def energy_score_chunk(
    probs: torch.Tensor,
    targets: torch.Tensor,
    embeddings: torch.Tensor,   # (V, d) — model's own input embedding matrix
    n_samples: int = 200,
) -> np.ndarray:
    """
    E_{X~P}[‖e(X)−e(y)‖] − ½ E_{X,X'~P}[‖e(X)−e(X')‖].

    Both expectations estimated by MC with paired independent draws,
    giving an O(N · n_samples · d) algorithm.
    Associated divergence: energy distance.
    """
    # Sample two independent sets from each row's distribution
    s1 = torch.multinomial(probs, n_samples, replacement=True)   # (N, n_samples)
    s2 = torch.multinomial(probs, n_samples, replacement=True)

    e1 = embeddings[s1]                              # (N, n_samples, d)
    e2 = embeddings[s2]
    e_y = embeddings[targets].unsqueeze(1)           # (N, 1, d)

    # E[d(X, y)]
    term1 = (e1 - e_y).norm(dim=-1).mean(dim=-1)    # (N,)

    # E[d(X, X')] via paired samples (unbiased)
    term2 = (e1 - e2).norm(dim=-1).mean(dim=-1)     # (N,)

    return (term1 - 0.5 * term2).cpu().numpy()


# ── 5. Kernel score (MMD-based, Gaussian RBF) ────────────────────────────────

def kernel_score_chunk(
    probs: torch.Tensor,
    targets: torch.Tensor,
    embeddings: torch.Tensor,   # (V, d)
    sigma: float,               # RBF bandwidth (use median heuristic)
    n_samples: int = 200,
) -> np.ndarray:
    """
    k(y, y) − 2 E_{X~P}[k(e(X), e(y))] + E_{X,X'~P}[k(e(X), e(X'))],
    where k is the Gaussian RBF kernel with bandwidth σ.

    Both expectations estimated by MC with paired independent draws.
    Associated divergence: MMD (Gretton et al., 2012).
    """
    s1 = torch.multinomial(probs, n_samples, replacement=True)   # (N, n_samples)
    s2 = torch.multinomial(probs, n_samples, replacement=True)

    e1 = embeddings[s1]                              # (N, n_samples, d)
    e2 = embeddings[s2]
    e_y = embeddings[targets].unsqueeze(1)           # (N, 1, d)

    inv2s2 = 1.0 / (2.0 * sigma ** 2)

    def rbf(a, b):
        return torch.exp(-((a - b) ** 2).sum(dim=-1) * inv2s2)

    k_xy  = rbf(e1, e_y).mean(dim=-1)   # (N,)  E[k(X,y)]
    k_xx  = rbf(e1, e2).mean(dim=-1)    # (N,)  E[k(X,X')]
    # k(y,y) = 1 for Gaussian kernel

    return (1.0 - 2.0 * k_xy + k_xx).cpu().numpy()


# ── Utility: median-heuristic bandwidth ─────────────────────────────────────

def compute_kernel_bandwidth(embeddings: torch.Tensor, n_subsample: int = 2_000) -> float:
    """Estimate σ as the median pairwise distance on a random subsample."""
    n = min(n_subsample, embeddings.shape[0])
    idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:n]
    sub = embeddings[idx]

    # Random pairs (avoid O(n^2) full matrix)
    n_pairs = 20_000
    i_idx = torch.randint(n, (n_pairs,), device=sub.device)
    j_idx = torch.randint(n, (n_pairs,), device=sub.device)
    dists = (sub[i_idx] - sub[j_idx]).norm(dim=-1)
    sigma = dists.median().item()
    return max(sigma, 1e-6)
