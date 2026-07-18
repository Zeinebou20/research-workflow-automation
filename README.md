# PhD Integrator Project — Du Calcul Symbolique à l'IA Scientifique Multi-GPU

[![Scientific Computing CI/CD Pipeline](https://github.com/Zeinebou20/research-workflow-automation/actions/workflows/ci_cd_pipeline.yml/badge.svg)](https://github.com/Zeinebou20/research-workflow-automation/actions/workflows/ci_cd_pipeline.yml)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen)
![Mypy](https://img.shields.io/badge/mypy-strict-blue)
![Ruff](https://img.shields.io/badge/ruff-passing-brightgreen)

> **TP 2 — Grand Projet Intégrateur Doctoral (D1, 2026-2030).**
> Pipeline de recherche reproductible résolvant une équation d'**advection-diffusion**
> `u_t + c·u_x − ν·u_xx = f(x, t)` par un réseau de neurones informé par la physique (PINN),
> orchestré par Snakemake et validé par CI/CD GitHub Actions.

---

## 1. Procédure de déploiement (uv)

Ce projet utilise [`uv`](https://github.com/astral-sh/uv) — aucun droit root requis, environnement scellé.

```bash
# 1. Installer uv (si absent) — sans sudo
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Synchroniser l'environnement verrouillé (déterministe via uv.lock)
uv sync

# 3. Validation qualité (identique à la CI)
uv run ruff check src/ tests/          # lint — 0 erreur
uv run mypy --strict src/              # typage strict — 0 erreur
USE_NUMBA=False uv run pytest --cov=src --cov-report=term-missing tests/

# 4. Exécuter l'intégralité du pipeline scientifique (commande unique)
uv run snakemake --cores all
```

## 2. Architecture du pipeline (DAG Snakemake)

```
derive_symbolic ──► reference.npz ──┐
                                    ├─► train_pinn ──► pinn_weights.pth ──► generate_plots ──► PDF + HTML
ingest_and_vectorize ─► grid.npy ───┘
analyze_stability ──► stability_report.txt
```

| Module | Fichier | Rôle |
|--------|---------|------|
| 3 — Symbolique | `src/symbolic_derivation.py` | Dérive `f(x,t)` exact depuis `u=tanh(x−ct)` (SymPy + lambdify) |
| 4 — NumPy | `src/numerical_core.py` | Layout mémoire (strides, C/F, vues) + ingestion Polars lazy + broadcasting |
| 5 — Stabilité | `src/stability_analysis.py` | κ(A) de Hilbert, précision float16/32/64, propagation de perturbation |
| 6 — HPC | `src/hpc_acceleration.py` | JIT Numba (`@njit parallel/fastmath`), profiling, balayage Joblib |
| 7 — PINN | `src/deep_pinn.py` | MLP + perte physique par autograd + entraînement CPU/CUDA/MPS |
| 8 — Visualisation | `src/visualisation.py` | Figure PDF double-panneau (usetex) + surface 3D Plotly HTML |
| 9 — Orchestration | `Snakefile` | DAG reproductible de bout en bout |

## 3. Fondement physique et mathématique

On adopte la **méthode de la solution manufacturée** : on postule une solution analytique
exacte `u(x, t) = tanh(x − c·t)` (onde solitaire), puis on injecte cette expression dans
l'opérateur d'advection-diffusion pour générer **symboliquement** le terme source exact :

```
f(x, t) = ∂u/∂t + c·∂u/∂x − ν·∂²u/∂x²
```

Comme `∂u/∂t = −c·∂u/∂x`, les termes d'advection s'annulent et `f = −ν·∂²u/∂x²`. Le PINN est
ensuite entraîné à minimiser le **résidu physique** (éq. 1 du TP), calculé par différentiation
automatique (`torch.autograd.grad`), combiné à une perte sur les conditions aux limites
`u = tanh(x − ct)`. La solution exacte connue permet de **quantifier l'erreur absolue** de
prédiction (panneau droit de la figure PDF).

## 4. Analyse critique : précision des flottants et accélérations

- **κ(A) et IEEE 754** : la matrice de Hilbert est extrêmement mal conditionnée
  (κ croît exponentiellement avec `n`). Une perturbation `ε = 10⁻⁷` sur `b` est amplifiée
  d'un facteur ≤ κ(A) sur la solution `α` (borne vérifiée dans `stability_analysis.py`).
  En `float16`, la reconstruction échoue dès `n ≈ 10` ; `float64` reste fiable — d'où le choix
  de la double précision pour la vérité de référence.
- **`fastmath=True`** : autorise des réassociations algébriques violant IEEE 754 (non
  reproductibilité stricte). Gain de performance réel (~2,5×), mais à réserver aux zones
  tolérantes à l'erreur — pas à la validation numérique de référence.
- **PINN sans maillage** : l'autograd calcule les dérivées exactes de û, évitant l'erreur de
  discrétisation des différences finies, au prix d'un entraînement stochastique.

## 5. Artéfacts générés

- `outputs/figures/solution_surface.pdf` — figure vectorielle double-panneau (solution + erreur)
- `outputs/figures/visualisation_pinn_interactive.html` — surface 3D Plotly autonome
- `models/pinn_weights.pth` — poids du modèle entraîné
- `outputs/stability_report.txt` — rapport de conditionnement

Ces artéfacts sont publiés automatiquement par la CI (`actions/upload-artifact`) en cas de succès.

## 6. Qualité et reproductibilité

- **0 erreur** Ruff · **0 erreur** Mypy `--strict` · couverture **≥ 90 %**
- Environnement verrouillé (`uv.lock`), **aucun `sudo`**, matrice CI Python 3.10 / 3.11 / 3.12
