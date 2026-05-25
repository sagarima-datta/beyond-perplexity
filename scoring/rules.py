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
    sorted_indices: torch.Tensor,  # (V,)  unused — kept for API compatibility
    n_samples: int = 200,
) -> np.ndarray:
    """
    Adapted CRPS for a discrete distribution ordered by unigram frequency rank,
    estimated via the equivalent energy/expectation representation:

        CRPS(P, y) = E_{X~P}[|r(X) − r(y)|] − ½ E_{X,X'~P}[|r(X) − r(X')|]

    where r(v) is the frequency rank of token v (0 = most frequent).

    This is mathematically identical to the CDF formulation but avoids
    materialising the full (N, V) CDF matrix — O(N · n_samples) instead of
    O(N · V).  Associated divergence: Cramér–von Mises distance.
    """
    # Sample two independent sets of tokens from each row's distribution
    s1 = torch.multinomial(probs, n_samples, replacement=True)   # (N, n_samples)
    s2 = torch.multinomial(probs, n_samples, replacement=True)

    # Convert sampled token IDs to frequency ranks
    r1 = freq_rank[s1].float()                        # (N, n_samples)
    r2 = freq_rank[s2].float()
    ry = freq_rank[targets].float().unsqueeze(1)      # (N, 1)

    # E[|r(X) - r(y)|]
    term1 = (r1 - ry).abs().mean(dim=-1)              # (N,)

    # E[|r(X) - r(X')|]  via paired samples
    term2 = (r1 - r2).abs().mean(dim=-1)              # (N,)

    return (term1 - 0.5 * term2).cpu().numpy()


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


# ── 6. CRPS-context (cosine-similarity ordering, expectation formulation) ──────

def crps_context_chunk(
    probs: torch.Tensor,
    targets: torch.Tensor,
    hidden_states: torch.Tensor,  # (N, d_model) — last transformer hidden state
    embeddings: torch.Tensor,     # (V, d_model) — vocabulary embedding matrix
    n_samples: int = 200,
) -> np.ndarray:
    """
    CRPS where the vocabulary is ordered by cosine similarity between the
    context hidden state h_t and each token embedding e_v, estimated via
    the equivalent energy/expectation representation:

        CRPS_context(P, y) = E_{X~P}[|cos(h,e_X) - cos(h,e_y)|]
                           - 0.5 * E_{X,X'~P}[|cos(h,e_X) - cos(h,e_X')|]

    Cosine similarity plays the role that frequency rank played in CRPS:
    it is the 1-D "position" of each token in the context-ordered space.

    This avoids materialising the full (N, V) cosine-similarity matrix,
    making chunking unnecessary — O(N · n_samples · d) only.
    """
    import torch.nn.functional as F

    # Sample two independent draws from each row's distribution
    s1 = torch.multinomial(probs, n_samples, replacement=True)   # (N, n_samples)
    s2 = torch.multinomial(probs, n_samples, replacement=True)

    # Normalise hidden states and embeddings once
    h_norm = F.normalize(hidden_states.float(), dim=-1)           # (N, d)
    e_norm = F.normalize(embeddings.float(), dim=-1)              # (V, d)

    # Cosine similarity of each context to sampled tokens
    # e_norm[s1]: (N, n_samples, d)  →  dot with h_norm[i]: (N, 1, d)
    cos_s1 = (h_norm.unsqueeze(1) * e_norm[s1]).sum(-1)          # (N, n_samples)
    cos_s2 = (h_norm.unsqueeze(1) * e_norm[s2]).sum(-1)          # (N, n_samples)

    # Cosine similarity to the correct target token
    cos_y = (h_norm * e_norm[targets]).sum(-1, keepdim=True)     # (N, 1)

    # E[|cos(h, X) - cos(h, y)|]
    term1 = (cos_s1 - cos_y).abs().mean(dim=-1)                  # (N,)

    # E[|cos(h, X) - cos(h, X')|]  via paired samples
    term2 = (cos_s1 - cos_s2).abs().mean(dim=-1)                 # (N,)

    return (term1 - 0.5 * term2).cpu().numpy()


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
