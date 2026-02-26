# Bayesian Gaussian Mixture Model via Gibbs Sampling

A from-scratch implementation of a fully Bayesian Gaussian Mixture Model (GMM) using Gibbs sampling, written in NumPy. This project demonstrates hands-on understanding of Bayesian inference, probabilistic graphical models, and Markov Chain Monte Carlo (MCMC) methods.

---

## What This Does

Given a 2D dataset, this model infers the posterior distribution over:
- **Cluster assignments** z_i for each datapoint
- **Cluster means** μ_k for each Gaussian component

Rather than finding a single point estimate like EM (Expectation-Maximization), Gibbs sampling approximates the full posterior — giving a richer, more principled picture of uncertainty in the model parameters.

---

## Why Gibbs Sampling?

Exact Bayesian inference over GMMs is intractable. Gibbs sampling is an MCMC method that sidesteps this by iteratively sampling each variable from its conditional distribution, given all others. Over many iterations, the samples converge to the true joint posterior.

**Compared to EM:**
- EM finds a point estimate (MAP); Gibbs approximates the full posterior
- Gibbs is more principled under uncertainty but computationally heavier
- Gibbs naturally handles multimodality in the posterior

---

## Generative Model

```
π ~ Dirichlet(α)          # Mixture weights
z_i ~ Multinomial(π)       # Cluster assignment for datapoint i
μ_k ~ N(0, 10I)            # Prior over cluster means
x_i ~ N(μ_{z_i}, I)       # Observed data
```

**Hyperparameters:**
- α = (1, ..., 1) — uniform Dirichlet prior (no preference over clusters)
- V_0 = 10I — weakly informative prior on cluster means
- Σ_k = I — observation covariance (assumed known)

---

## Gibbs Update Equations

At each iteration, three conditional distributions are sampled in sequence:

**Cluster memberships** (computed in log space for numerical stability):
```
P(z_i = k | x_i, μ, π) ∝ π_k · N(x_i | μ_k, I)
```

**Mixture weights:**
```
P(π | z) = Dirichlet(α_k + N_k)
```

**Posterior cluster means** (Gaussian-Gaussian conjugacy):
```
V_k^{-1} = V_0^{-1} + N_k · Σ^{-1}
m_k      = V_k (Σ^{-1} · N_k · x̄_k + V_0^{-1} · m_0)
μ_k      ~ N(m_k, V_k)
```

---

## Technical Highlights

- **Full Bayesian treatment** — posterior over means and mixture weights, not just point estimates
- **Conjugate priors** — Gaussian-Gaussian and Dirichlet-Multinomial conjugacy allow closed-form conditional updates
- **Numerically stable** — all cluster membership probabilities computed in log space using the logsumexp trick to prevent underflow
- **Efficient caching** — sufficient statistics N_k (cluster counts) and x̄_k (cluster means) are updated incrementally at each step rather than recomputed from scratch, reducing per-iteration cost
- **Burn-in handling** — early samples discarded to allow the chain to converge before collecting results

---

## Usage

```bash
python gmm_gibbs.py <input_file> <cluster_output_file> <means_output_file>
```

**Example:**
```bash
python gmm_gibbs.py data.csv clusters.csv posterior_means.csv
```

**Output:**
- `clusters.csv` — N rows, 3 columns (features + cluster label from final iteration)
- `posterior_means.csv` — K rows, 2 columns (posterior mean for each cluster)

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| k | 2 | Number of Gaussian components |
| T | 100 | Total Gibbs iterations |
| BURN_IN | 30 | Iterations discarded before collecting samples |
| V_0 | 10I | Prior covariance on cluster means |
| Σ | I | Observation covariance (fixed, not estimated) |

---

## Requirements

```
numpy
scipy
pandas
matplotlib
```

```bash
pip install numpy scipy pandas matplotlib
```

---

## Background

This project was built to develop a deep understanding of MCMC methods and Bayesian inference in the context of unsupervised learning. The math draws on conjugate prior theory, the Dirichlet-Multinomial model, and multivariate Gaussian conditioning — all implemented directly without relying on probabilistic programming libraries.
