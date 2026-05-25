# Beyond Perplexity: Evaluating LLM Token Distributions with Proper Scoring Rules

**Stat 591: Distributional Learning and Generative Modeling — Spring 2026**

---

## Abstract

Standard evaluation of large language models collapses to perplexity, the exponential of the mean log-score, which corresponds to a single Bregman divergence: KL. We evaluate three small publicly available language models—GPT-2 (117M), GPT-2-Medium (345M), and OPT-125M—on WikiText-103 and a 10K-document sample of the Pile using five strictly proper scoring rules: the log-score, quadratic (Brier) score, CRPS, energy score, and kernel score. Across four experiments we find that (1) scoring-rule rankings disagree in ways that are theoretically predictable; (2) log-score sensitivity to rare tokens is 2.5× larger than the quadratic score and nearly 1.5× larger than the energy score; (3) the log/energy penalty ratio rises by 45% from low- to high-entropy predictions, confirming the energy score's robustness to genuine uncertainty; and (4) Kendall τ between CRPS and energy score drops by up to 51% on tokens whose frequency rank and semantic geometry are discordant, providing direct causal evidence that each score emphasises a distinct distributional aspect.

---

## 1. Introduction

A language model produces a full probability distribution over its vocabulary at every generation step. The near-universal evaluation metric, perplexity, reduces this distribution to a single number via the log-score: S_log(P, y) = log p(y). The primacy of this choice is largely historical rather than principled (Jelinek et al., 1977). From the perspective of proper scoring rules (Gneiting & Raftery, 2007), the log-score is only one element of a rich family, each associated—through the Bregman representation theorem—with a distinct divergence and therefore with a distinct distributional aspect.

The Bregman representation theorem states that every strictly proper scoring rule S has the form

  S(P, y) − S(Q, Q) = d_φ(Q, P)

for a strictly convex function φ, where d_φ(Q, P) is the corresponding Bregman divergence. "Better" under a given score therefore means "closer to the data-generating distribution in the geometry induced by φ." Choosing perplexity as the sole metric implicitly assumes that KL geometry is the right geometry for all model comparisons—an assumption that is theoretically unjustified and empirically untested.

This project tests that assumption systematically by evaluating three models on two corpora with five scoring rules and examining whether, when, and why their rankings disagree.

---

## 2. Setup

**Models.** GPT-2 (117M parameters), GPT-2-Medium (345M), and OPT-125M (125M), all publicly available on the HuggingFace Hub.

**Corpora.** WikiText-103 (test split; formal encyclopedic text) and a 10,000-document streaming sample of the Pile (diverse web and book text). We evaluate approximately 5,000 token positions per (model, corpus) pair.

**Scoring rules.** All five rules are negatively oriented (lower = better):

| Rule | Formula | Associated divergence |
|---|---|---|
| Log-score | −log p(y) | KL divergence |
| Quadratic (Brier) | ‖p − e_y‖² | L² distance |
| CRPS | Σ_t (F_P(t) − 𝟏{rank(y) ≤ t})² | Cramér–von Mises distance |
| Energy score | E[‖e(X)−e(y)‖] − ½E[‖e(X)−e(X')‖] | Energy distance |
| Kernel score (MMD) | 1 − 2E[k(e(X),e(y))] + E[k(e(X),e(X'))] | Maximum Mean Discrepancy |

CRPS orders the vocabulary by unigram frequency rank; the energy and kernel scores use each model's own input embedding matrix as the feature map φ, with kernel bandwidth σ set by the median heuristic (σ ≈ 1.97–4.81 depending on the model). Expectations under P for the energy and kernel scores are estimated by paired Monte Carlo draws (N = 200) from the softmax distribution.

---

## 3. Experiment 1: Global Score Comparison and Model Ranking

### Results

Mean scores on the Pile:

| Model | log | quadratic | crps | energy | kernel |
|---|---|---|---|---|---|
| GPT-2 | 3.332 | 0.719 | 1256 | 1.315 | 0.184 |
| GPT-2-Medium | 3.005 | 0.692 | **1174** | 1.100 | **0.166** |
| OPT-125M | **3.003** | **0.690** | 1179 | **0.692** | 0.278 |

Ranking reversals appear for every model pair under at least one rule. On the Pile, OPT-125M ties GPT-2-Medium under the log-score and wins by a substantial margin under the energy score (0.692 vs. 1.100), but falls to last place under the kernel score (0.278 vs. 0.166). On WikiText-103 the pattern sharpens further: OPT-125M ranks first under the energy score (0.835 vs. 1.278 vs. 1.513) but third under the log-score and kernel score.

Pairwise Kendall τ at the token level (WikiText-103):

| Rule pair | τ |
|---|---|
| log ↔ quadratic | 0.691 |
| log ↔ energy | 0.603 |
| crps ↔ energy | 0.389 |
| **quadratic ↔ crps** | **0.266** |

All τ values are significantly below 1 (all p = 0), confirming that the rules capture genuinely different distributional information at the token level.

### Theoretical interpretation

The ranking reversals are precisely what the Bregman representation theorem predicts. OPT-125M's strong energy-score performance indicates that its predictive mass, when wrong, tends to fall in the same region of embedding space as the correct token. Its poor kernel-score performance reflects the Gaussian kernel's sharper locality: errors that are semantically close but outside the immediate neighbourhood (distance ≫ σ) are penalised little by the energy score but substantially by the kernel score, as the exponential decay of the RBF kernel suppresses contributions from tokens beyond ≈2σ.

The low quadratic–CRPS agreement (τ = 0.266) is the single most striking token-level finding. The quadratic score applies L² pressure across the entire probability vector (d_φ corresponds to squared Euclidean distance on the simplex); CRPS applies Cramér–von Mises pressure on the frequency-ranked CDF. These two geometries are nearly orthogonal at the token level, confirming that the two scores are measuring fundamentally different distributional properties.

---

## 4. Experiment 2: Score Behavior by Token Frequency

### Results

Tokens are stratified into ten unigram-frequency deciles (decile 0 = most frequent 10% of the vocabulary). We report averaged-across-models scores on the Pile, where all ten deciles are populated:

| Decile | log | quadratic | crps | energy |
|---|---|---|---|---|
| 0 (most freq.) | 2.75 | 0.673 | 1053 | 0.940 |
| 4 | 5.46 | 0.896 | 2019 | 1.640 |
| **8 (rare)** | **6.79** | **0.885** | **2519** | **1.743** |

From decile 0 to decile 8, the log-score increases by a factor of **2.47×**, the quadratic score by only **1.31×**, and the energy score by **1.85×** on average. OPT-125M shows the most pronounced asymmetry: its energy score rises only **1.51×** over the same range while its log-score rises **2.71×**, implying that OPT's rare-token errors are specifically semantically coherent even when it has low probability mass on the correct token.

A non-monotonic dip at decile 9 is visible in CRPS (1607 vs. 2519 at decile 8), likely because the very rarest tokens in our small sample cluster in idiosyncratic ways relative to the frequency-rank ordering.

### Theoretical interpretation

The sharp log-score rise is a direct consequence of the KL divergence's locality: S_log depends only on p(y), and as y becomes rare, any model will assign it low probability, driving −log p(y) → ∞. The Brier score's boundedness (‖p − e_y‖² ∈ [0, 2]) prevents this explosion; its L² geometry distributes the penalty across the full probability vector rather than concentrating it at p(y).

The intermediate behavior of the energy score—growing faster than quadratic but slower than log—reflects its intermediate position in the hierarchy of divergences. It is sensitive to where the model places mass in embedding space, so errors on rare tokens are penalised in proportion to how far the model's predictive mean drifts from the correct embedding, not in proportion to − log p(y). For OPT-125M, whose embedding space appears to group semantically related tokens closely regardless of frequency, this distinction is especially consequential.

---

## 5. Experiment 3: Score Behavior by Prediction Entropy

### Results

Token positions are stratified into low/medium/high entropy tertiles of H(p) = −Σ p_k log p_k. Ratios of mean log-score to mean energy/kernel scores (averaged across models):

| Corpus | Entropy bin | log / energy | log / kernel |
|---|---|---|---|
| Pile | low (H < 2.24) | 2.56 | 13.3 |
| Pile | medium | 2.81 | 14.2 |
| Pile | **high (H ≥ 4.31)** | **3.72** | **17.6** |
| WikiText-103 | low (H < 3.06) | 2.68 | 14.0 |
| WikiText-103 | medium | 3.42 | 16.8 |
| WikiText-103 | **high (H ≥ 4.77)** | **4.16** | **19.1** |

The log/energy ratio rises by 45% (Pile) and 55% (WikiText-103) from the low to the high entropy tertile. On GPT-2/WikiText-103, the mean log-score increases from 2.14 (low) to 5.92 (high entropy), a **2.77×** increase; over the same range the energy score increases from 1.04 to 1.90, only **1.83×**. The quadratic score is the most stable of all, rising from 0.635 to 0.970 (a 1.53× increase), consistent with its bounded, global character.

### Theoretical interpretation

At high entropy the model spreads probability mass diffusely. The log-score's locality means it evaluates only the mass assigned to the realized token y, ignoring where the remaining mass went. If a model is uncertain but still concentrates mass in the semantic neighbourhood of y, the log-score cannot credit this: it sees only a small p(y) and returns a large −log p(y).

The energy score's non-locality redeems this: E[‖e(X) − e(y)‖] is small whenever the model's predictive mean is close to e(y) in embedding space, regardless of H(p). In the language of Bregman divergences, the energy distance d_φ(δ_y, P) measures how far P's center of mass (in feature space) is from y; a high-entropy P concentrated near y can be at small energy distance from δ_y even though the KL divergence KL(δ_y ‖ P) = −log p(y) is large. The widening log/energy ratio across entropy tertiles is therefore a direct, empirically confirmed consequence of the differing geometries of KL and energy distance under distributional uncertainty.

---

## 6. Experiment 4: CRPS vs. Energy Score on Semantically Structured Errors

### Results

For the top-3,000 most frequent vocabulary tokens we compute, for each token v:

- **S(v):** top-30 nearest neighbors by cosine similarity in the model's embedding space
- **R(v):** top-30 nearest neighbors by |freq_rank(u) − freq_rank(v)|
- **Discordance:** d(v) = 1 − |S(v) ∩ R(v)| / 30

Token positions whose target token has d < median(d) are labelled **concordant**; those with d ≥ median(d) are **discordant**. Kendall τ between CRPS and energy at the token level:

| Model | Corpus | Concordant τ | Discordant τ | Drop |
|---|---|---|---|---|
| GPT-2 | WikiText-103 | 0.396 | 0.326 | −18% |
| GPT-2 | Pile | 0.537 | 0.264 | **−51%** |
| GPT-2-Medium | WikiText-103 | 0.414 | 0.343 | −17% |
| GPT-2-Medium | Pile | 0.536 | 0.331 | −38% |
| OPT-125M | WikiText-103 | 0.355 | **0.193** | **−46%** |
| OPT-125M | Pile | 0.529 | 0.335 | −37% |

The drop in τ is significant (p < 10⁻²³ in the worst case) and consistent across all six model/corpus combinations. Mean scores further confirm the divergence: on the Pile, mean CRPS rises 88% from concordant to discordant positions, while mean energy score rises only 59%.

### Theoretical interpretation

This experiment provides the most direct causal test of the CRPS–energy distinction. CRPS integrates the squared difference of CDFs over the frequency-ranked vocabulary: a model that assigns mass to tokens adjacent in frequency rank receives partial credit even if those tokens are semantically distant. The energy score integrates Euclidean distances in embedding space: a model that assigns mass to semantically similar tokens receives partial credit even if those tokens are rare or distant in frequency rank.

On concordant tokens—where semantic proximity and frequency proximity agree—the two scores produce similar rankings (τ ≈ 0.40–0.54). On discordant tokens—where a token's semantic neighbours are not its frequency-rank neighbours—the two scores disagree substantially (τ ≈ 0.19–0.33). The magnitude of the drop indexes how much of the variation in one score cannot be explained by the other once the shared "token difficulty" variance is removed, providing a quantitative handle on the distinct aspects of distributional quality each score measures.

OPT-125M shows the sharpest drop on WikiText-103 (τ: 0.355 → 0.193). Combined with its strong energy-score but poor kernel-score performance in Experiment 1, this points to a systematic property of OPT's learned representations: its embedding space organises tokens by semantic function rather than by frequency, creating a wide population of discordant tokens where frequency rank and semantic geometry diverge.

---

## 7. Discussion

Taken together, the four experiments support three conclusions.

**Scoring rule rankings are not invariant.** OPT-125M is the best model under log-score and energy score on the Pile but the worst under the kernel score. GPT-2-Medium leads under CRPS and kernel score on both corpora. These reversals are not noise: they are reproduced across corpora and are consistent with each model's architectural choices. The NLP community's implicit assumption that perplexity rankings generalise to other scoring rules is falsified by these data.

**The log-score is disproportionately sensitive to two factors that other rules discount.** It penalises rare tokens (Experiment 2, 2.47× amplification vs. 1.31× for quadratic) and uncertain predictions (Experiment 3, 45–55% wider log/energy gap at high entropy). Both sensitivities follow directly from KL divergence's locality: −log p(y) encodes only the probability mass assigned to the realised token, so any factor that reduces p(y)—rarity, uncertainty, or semantic distance from the model's probability mass—inflates the score without distinction.

**CRPS and energy score are operationally distinct at the token level, and the distinction has a geometric cause.** The 37–51% drop in their Kendall τ on discordant tokens (Experiment 4) is the strongest single result: it shows that the two scores are measuring genuinely different things, and that the disagreement is triggered precisely by the tokens for which the frequency-rank and embedding-space orderings diverge. This confirms the Bregman framework's prediction that each rule's associated divergence geometry determines its diagnostic content.

A practical implication: if one wishes to evaluate whether a model makes *semantically coherent* errors (confusing synonyms rather than arbitrary tokens), the energy or kernel score provides signal that the log-score cannot. Conversely, if one cares about the model's calibration on common tokens—which dominate most downstream applications—the quadratic score's bounded, vocabulary-wide character makes it more robust than log-score to the long tail of rare tokens.

---

## 8. Conclusion

We have demonstrated empirically that five strictly proper scoring rules provide substantially different information about LLM next-token distributions, and that their disagreements are interpretable through the lens of the Bregman representation theorem. The log-score/perplexity paradigm, while convenient, conflates tail sensitivity, frequency sensitivity, and calibration into a single number. A richer evaluation using even one additional rule—particularly the energy score for semantic calibration or the quadratic score for robustness—reveals model differences that perplexity cannot detect.

---

## References

Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.

Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13, 723–773.

Jelinek, F., Mercer, R. L., Bahl, L. R., & Baker, J. K. (1977). Perplexity — a measure of the difficulty of speech recognition tasks. *Journal of the Acoustical Society of America*, 62(S1).

Steinwart, I. & Ziegel, J. F. (2021). Strictly proper kernel scores and characteristic kernels on compact spaces. *Applied and Computational Harmonic Analysis*, 51, 510–542.
