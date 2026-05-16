# Bayesian Hierarchical Models for Student Math Scores

A Gibbs Sampler implementation for estimating global and school-specific effects on student math scores across 100 US public high schools.

## Overview

This project applies Bayesian hierarchical modelling to analyse math score data from students spread across 100 large urban public high schools in the USA. The Gibbs Sampler (GS) is used as the inference engine because many schools have small sample sizes — a setting where partial pooling allows smaller groups to borrow statistical strength from the full dataset.

Two models are developed and compared:

- **Model 1** — Hierarchical model without covariates
- **Model 2** — Hierarchical model with covariates (`sch_freelunch`, `stu_ses`)

---

## Models

### Model 1: No Covariates

Each school's math scores are modelled as:

```
Y_{i,j} | θ_j, σ² ~ N(θ_j, σ²)       i = 1,…,m,  j = 1,…,n_j
θ_j | μ, τ²        ~ N(μ, τ²)
μ, τ², σ²          ~ p₀
```

**Priors:**
- `1/σ² ~ Gamma(ν₀/2, ν₀σ₀²/2)`
- `1/τ² ~ Gamma(η₀/2, η₀τ₀²/2)`
- `μ ~ Normal(μ₀, γ₀²)`

Full conditionals are derived analytically and sampled iteratively.

### Model 2: With Covariates

Each school's math scores are modelled using school-level regression coefficients:

```
Y_{i,j} | β_j, σ²  ~ N(β_j^T x_{i,j}, σ²)
β_j | θ, Σ          ~ N(θ, Σ)
θ, Σ, σ²            ~ p₀
```

**Covariates:**
- `stu_ses` — student socioeconomic status (school-specific coefficient)
- `sch_freelunch` — school-level free lunch rate (constant coefficient across schools, estimated via OLS on the full dataset)

**Priors:**
- `θ ~ MultivariateNormal(μ₀, Λ₀)`
- `Σ ~ InverseWishart(η₀, S₀⁻¹)`
- `σ² ~ InverseGamma(ν₀/2, ν₀σ₀²/2)`

---

## Implementation

### Files

| File | Description |
|---|---|
| `Final_project_script.ipynb` | Main Jupyter notebook with both Gibbs Sampler implementations, diagnostics, and plots |
| `Scores_without_covariates` | Dataset for Model 1 (space-separated) |
| `Scores_with_covariates` | Dataset for Model 2 (space-separated) |

### Dependencies

```bash
pip install numpy matplotlib pandas scipy statsmodels seaborn
```

### Running the Notebook

```bash
jupyter notebook FInal_project_script.ipynb
```

---

## Results

### Model 1

The algorithm was run for **5,000 iterations** with the following prior parameters:

```python
nu0, eta0 = 1, 1
mu0 = 50
s20, t20, g20 = 100, 100, 25
```

Key findings:
- **Best school:** School 51 — average θ of **61.83**
- **Worst school:** School 5 — average θ of **38.21**
- Trace plots confirm stationarity (chain convergence)
- ACFs show no significant autocorrelation after the first few lags
- Pairwise scatterplots of μ, σ², τ² show no inter-parameter correlation, confirming good sampler mixing

The experiment was repeated with alternative prior parameters (e.g., `mu0 = 0`, `nu0 = 5`) to assess prior sensitivity. Convergence behaviour was compared across both settings.

### Model 2

The model was run for **1,000 iterations** using data-driven initialisations:
- β initialised via OLS per school
- θ and Σ initialised from the empirical mean and covariance of the per-school OLS estimates
- σ² initialised from the average within-school variance

The `sch_freelunch` coefficient is fixed across all schools (estimated globally via OLS on all observations) since it does not vary within schools. Only the `stu_ses` coefficient is updated at each Gibbs iteration.

---

## Diagnostics

For both models, convergence is assessed via:

- **Trace plots** of θ_j, μ, σ², τ² (and β_j, Σ, σ² in Model 2)
- **Autocorrelation function (ACF)** plots
- **Kernel density estimates** (KDE)
- **Pairwise scatterplots** of global parameters

---

## Prior Sensitivity

Model 1 was run under two prior configurations to study the effect of prior choice on posterior inference and chain convergence speed. Informative priors (second configuration) centred away from the data mean (`mu0 = 0`) showed slower initial convergence but ultimately recovered the same stationary distribution.
