# Mémoire de projet — TP 2

## Du Calcul Symbolique à l'IA Scientifique Multi-GPU avec Automatisation de la Reproductibilité (CI/CD)

**Formation Doctorale : Calcul Scientifique, HPC et Génie Logiciel Avancé — Cycle Ph.D. D1 — Session 2026-2030**

> *Note au lecteur : ce mémoire documente l'intégralité du pipeline développé. Les mesures numériques présentées ont été relevées sur la station de développement (CPU, `USE_NUMBA=True`) ; elles sont à réactualiser si le projet est exécuté sur une autre architecture.*

---

## Table des matières

1. [Introduction et problématique scientifique](#1-introduction-et-problématique-scientifique)
2. [Cadre méthodologique et architecture logicielle](#2-cadre-méthodologique-et-architecture-logicielle)
3. [Module 3 — Dérivation symbolique et solution manufacturée](#3-module-3--dérivation-symbolique-et-solution-manufacturée)
4. [Module 4 — Maîtrise de NumPy et ingestion massive](#4-module-4--maîtrise-de-numpy-et-ingestion-massive)
5. [Module 5 — Stabilité, conditionnement et analyse des erreurs](#5-module-5--stabilité-conditionnement-et-analyse-des-erreurs)
6. [Module 6 — Optimisation, accélération JIT et parallélisme](#6-module-6--optimisation-accélération-jit-et-parallélisme)
7. [Module 7 — Réseau de neurones informé par la physique (PINN)](#7-module-7--réseau-de-neurones-informé-par-la-physique-pinn)
8. [Module 8 — Visualisation scientifique pour publication](#8-module-8--visualisation-scientifique-pour-publication)
9. [Module 9 — Orchestration et intégration continue](#9-module-9--orchestration-et-intégration-continue)
10. [Réponses aux questions théoriques](#10-réponses-aux-questions-théoriques)
11. [Analyse critique : précision, accélération et fidélité du modèle](#11-analyse-critique--précision-accélération-et-fidélité-du-modèle)
12. [Conclusion et perspectives](#12-conclusion-et-perspectives)
13. [Annexe — Procédure de réplication](#13-annexe--procédure-de-réplication)

---

## 1. Introduction et problématique scientifique

La validation des résultats en recherche computationnelle moderne ne peut plus reposer sur
l'affirmation « ça fonctionne sur ma machine ». Ce projet met en œuvre un **pipeline de recherche
reproductible de bout en bout** pour l'étude d'un phénomène de transfert de masse régi par une
**équation d'advection-diffusion** :

$$\frac{\partial u}{\partial t} + c\,\frac{\partial u}{\partial x} - \nu\,\frac{\partial^2 u}{\partial x^2} = f(x, t)$$

où `c` est la vitesse d'advection et `ν` le coefficient de diffusion. L'objectif est de résoudre et
d'analyser ce système en combinant quatre piliers :

- **calcul symbolique** et manipulation de données massives ;
- **accélération de bas niveau** (JIT) et distribution de charge ;
- **apprentissage profond informé par la physique** (PINN) ;
- **automatisation complète** via GitHub Actions et orchestration Snakemake.

La contrainte structurante est la **reproductibilité absolue** : environnement scellé et verrouillé
(`uv.lock`), aucun droit *root* requis, et validation autonome sur infrastructure tierce.

---

## 2. Cadre méthodologique et architecture logicielle

Le projet suit une architecture modulaire à responsabilité unique, chaque module correspondant à un
maillon du DAG scientifique :

```
src/
├── symbolic_derivation.py   # Module 3 — dérivation symbolique (SymPy)
├── numerical_core.py        # Module 4 — NumPy / Polars
├── stability_analysis.py    # Module 5 — conditionnement, IEEE 754
├── hpc_acceleration.py      # Module 6 — Numba + Joblib
├── deep_pinn.py             # Module 7 — PINN (PyTorch)
└── visualisation.py         # Module 8 — Matplotlib / Plotly
```

**Chaîne qualimétrique.** Trois barrières statiques et dynamiques garantissent la robustesse avant
toute simulation coûteuse :

| Outil | Rôle | État |
|-------|------|------|
| `ruff` | Analyse statique / style | **0 erreur** |
| `mypy --strict` | Vérification de types stricte | **0 erreur** |
| `pytest --cov` | Tests unitaires + couverture | **55 tests, 97 %** |

La gestion de l'environnement repose sur **`uv`** (gestionnaire écrit en Rust), sans installation
globale ni privilège root, avec un cache activé pour minimiser l'empreinte carbone des exécutions CI.

---

## 3. Module 3 — Dérivation symbolique et solution manufacturée

### 3.1 Méthode de la solution manufacturée

Pour disposer d'une **vérité terrain analytique**, on adopte la *méthode de la solution manufacturée* :
plutôt que de résoudre l'EDP, on **postule** une solution exacte et on en déduit le terme source qui la
rend valide. On choisit une onde solitaire :

$$u(x, t) = \tanh(x - c\,t)$$

À l'aide de SymPy, on calcule symboliquement les dérivées partielles $\partial_t u$, $\partial_x u$,
$\partial_{xx} u$, puis on **génère automatiquement** le terme source résiduel exact :

$$f(x, t) = \frac{\partial u}{\partial t} + c\,\frac{\partial u}{\partial x} - \nu\,\frac{\partial^2 u}{\partial x^2}$$

### 3.2 Propriété remarquable

Comme $u$ ne dépend que de la variable caractéristique $\xi = x - ct$, on a $\partial_t u = -c\,\partial_x u$.
Les termes d'advection **s'annulent exactement**, et le terme source se réduit à :

$$f(x, t) = -\nu\,\frac{\partial^2 u}{\partial x^2} = 2\nu\,\tanh(\xi)\,\bigl(1 - \tanh^2(\xi)\bigr)$$

Cette identité est vérifiée numériquement dans la suite de tests (`test_solution_satisfies_pde`).
Les expressions symboliques sont exportées en **code NumPy vectorisé** via `sympy.lambdify`, prêtes à
être évaluées sur des grilles entières sans boucle Python.

---

## 4. Module 4 — Maîtrise de NumPy et ingestion massive

### 4.1 Manipulation bas niveau du `ndarray`

Le module explore explicitement les attributs mémoire d'un tableau bidimensionnel : `shape`, `dtype`,
`strides`. Deux points sont démontrés par programme :

- **Layout mémoire (C vs F).** Un stockage *C-contiguous* (ligne par ligne) favorise la localité de
  cache lors des réductions **par ligne** ; un stockage *F-contiguous* (colonne par colonne) favorise
  les réductions **par colonne**. Un accès non aligné sur le layout provoque des défauts de cache.
- **Vues vs copies.** Un découpage `grid[::2, ::2]` renvoie une **vue** partageant la mémoire de base
  (vérifié par `sub_grid.base is grid`), et non une copie — distinction cruciale pour éviter les
  duplications mémoire silencieuses.

### 4.2 Ingestion massive et vectorisation

L'ingestion des mesures capteurs (CSV/Parquet) utilise l'**API lazy de Polars** (`scan_csv` /
`scan_parquet`) : le filtrage des coordonnées valides est poussé au niveau du plan d'exécution, sans
charger l'intégralité du fichier en RAM. Le terme source `f(x, t)` est ensuite appliqué sur la grille
entière par **broadcasting**, sans aucune boucle `for` itérative.

---

## 5. Module 5 — Stabilité, conditionnement et analyse des erreurs

On étudie la sensibilité de la résolution du système linéaire $A\alpha = b$ issu de la discrétisation,
où $A$ est la **matrice de Hilbert**, archétype de matrice mal conditionnée.

### 5.1 Conditionnement et propagation d'erreur

Le nombre de conditionnement $\kappa(A) = \lVert A \rVert \cdot \lVert A^{-1} \rVert$ croît de façon
explosive avec la dimension. En perturbant le second membre d'un bruit $\varepsilon = 10^{-7}$, on
mesure l'amplification de l'erreur sur la solution, bornée théoriquement par :

$$\frac{\lVert \Delta\alpha \rVert}{\lVert \alpha \rVert} \le \kappa(A)\,\frac{\lVert \Delta b \rVert}{\lVert b \rVert}$$

**Résultats mesurés :**

| $n$ | $\kappa(A)$ | Erreur relative $\lVert\Delta\alpha\rVert/\lVert\alpha\rVert$ | Borne théorique |
|----:|------------:|------------------------------:|----------------:|
| 5   | 4.77 × 10⁵  | 1.93 × 10⁻⁵ | 3.73 × 10⁻² |
| 10  | 1.60 × 10¹³ | 5.28 × 10⁻² | 1.66 × 10⁶ |
| 15  | 3.68 × 10¹⁷ | 3.84 × 10¹  | 3.11 × 10¹⁰ |
| 20  | 1.32 × 10¹⁸ | 2.26 × 10¹  | 1.34 × 10¹¹ |
| 25  | 1.33 × 10¹⁸ | 7.66 × 10¹  | 1.29 × 10¹¹ |

L'erreur mesurée reste **systématiquement inférieure à la borne**, confirmant l'inégalité. Dès $n \ge 15$,
$\kappa(A)$ dépasse la précision machine du `float64` ($\approx 10^{16}$) : la solution perd toute
signification, illustrant la limite intrinsèque de la double précision face au mauvais conditionnement.

### 5.2 Précision des types et validation robuste

La comparaison `float16` / `float32` / `float64` montre que la précision réduite échoue dès les
petites dimensions (débordement en `float16`, dont la valeur maximale ~65 504 est dépassée par les
composantes de $\alpha$). La validation numérique proscrit l'**égalité stricte** (`==`) — jamais
satisfaite à cause des arrondis IEEE 754 — au profit de `np.isclose()` avec des tolérances `atol`/`rtol`
adaptées à la précision. Le résidu strict $r = \lVert A\alpha - b \rVert_2$ fournit la métrique de
validation.

---

## 6. Module 6 — Optimisation, accélération JIT et parallélisme

### 6.1 Diagnostic de profiling

Le profiling (`cProfile` + `timeit`) identifie que les boucles imbriquées d'accès séquentiel à la grille
constituent le goulot d'étranglement, justifiant une compilation JIT.

### 6.2 Accélération JIT via Numba

L'opérateur de filtrage local est décoré par `@njit(parallel=True, fastmath=True)`, activant la
vectorisation multi-cœurs (`prange`) et les optimisations mathématiques agressives.

**Résultats mesurés (grille 600 × 600, 10 exécutions) :**

| Implémentation | Temps moyen / appel | Accélération |
|----------------|--------------------:|-------------:|
| NumPy vectorisé | 21.52 ms | 1.00× (référence) |
| Numba JIT       | 12.23 ms | **1.76×** |

L'accélération est réelle bien que modérée sur cette opération déjà bien vectorisée par NumPy ; le gain
serait supérieur sur des noyaux à branchements ou dépendances de données que NumPy ne peut vectoriser.

**Risques de `fastmath=True`.** Ce drapeau autorise des réassociations algébriques (ex. commutativité de
l'addition flottante) qui **violent la norme IEEE 754**. Le gain de performance se paie par une perte de
reproductibilité stricte : deux exécutions peuvent différer aux derniers bits. En modélisation sensible
(qualité de l'air, imagerie médicale), ce mode est à réserver aux zones tolérantes à l'erreur.

### 6.3 Parallélisation de haut niveau

L'exploration de l'espace des paramètres $(c, \nu)$ est distribuée via **Joblib** (`Parallel`/`delayed`)
sur l'ensemble des cœurs logiques, chaque combinaison étant une simulation indépendante (parallélisme
*embarrassingly parallel*). Le temps global décroît proportionnellement au nombre de *workers*.

---

## 7. Module 7 — Réseau de neurones informé par la physique (PINN)

### 7.1 Architecture

On remplace le solveur classique par un **perceptron multicouche** $\hat{u}_\theta(x, t)$ (entrées
$(x, t)$, 3 couches cachées de 32 neurones, activations `tanh`, sortie scalaire). L'entraînement détecte
automatiquement la meilleure architecture disponible : **CUDA > MPS (Apple Silicon) > CPU**.

### 7.2 Perte physique sans maillage

Le cœur de l'approche est le calcul du **résidu de l'EDP par différentiation automatique**
(`torch.autograd.grad`), sans discrétisation par différences finies :

$$\mathcal{L}_{\text{physique}} = \frac{1}{N_p}\sum_{i=1}^{N_p}\left(\frac{\partial \hat{u}}{\partial t} + c\,\frac{\partial \hat{u}}{\partial x} - \nu\,\frac{\partial^2 \hat{u}}{\partial x^2} - f(x_i, t_i)\right)^2$$

Les dérivées d'ordre supérieur sont obtenues en chaînant `autograd.grad` avec `create_graph=True`
(nécessaire pour dériver une seconde fois et pour rétropropager à travers l'opérateur différentiel). La
perte totale combine ce terme physique avec une **perte de données/conditions aux limites**
$\mathcal{L}_{\text{données}}$ qui contraint $\hat{u}$ à coïncider avec la solution exacte
$u = \tanh(x - ct)$ sur les bords et la condition initiale :

$$\mathcal{L} = \mathcal{L}_{\text{physique}} + \mathcal{L}_{\text{données}}$$

### 7.3 Résultats d'entraînement

Sur 200 époques (optimiseur Adam, `lr = 10⁻³`, graine fixée à 42 pour la reproductibilité) :

| Grandeur | Valeur |
|----------|-------:|
| Perte initiale | 3.60 × 10⁻¹ |
| Perte finale   | 5.49 × 10⁻⁴ |
| Facteur de réduction | **≈ 655×** |

La décroissance de trois ordres de grandeur confirme que le réseau **apprend effectivement** à satisfaire
l'EDP. La disponibilité de la solution exacte permet en outre de quantifier l'**erreur absolue** de
prédiction (panneau droit de la figure de résultats).

---

## 8. Module 8 — Visualisation scientifique pour publication

Deux artéfacts sont produits :

1. **Figure statique double-panneau** (Matplotlib + Seaborn) : à gauche la solution approchée
   $\hat{u}(x, t)$ en carte de chaleur, à droite l'erreur absolue $|\hat{u} - u_{\text{exact}}|$.
   Export en **PDF vectoriel**. Le rendu LaTeX (`text.usetex`) est activable via la variable
   d'environnement `USE_LATEX=1` lorsqu'une distribution TeX est disponible — désactivé par défaut pour
   garantir l'exécution sur les runners CI dépourvus de TeX.
2. **Surface 3D interactive** (Plotly) de $\hat{u}(x, t)$, exportée en **HTML autonome** (bibliothèque
   Plotly embarquée) pour intégration dans un tableau de bord de suivi.

---

## 9. Module 9 — Orchestration et intégration continue

### 9.1 Orchestration Snakemake

Le pipeline complet est exécutable en une commande (`snakemake --cores all`). Le DAG relie cinq règles :

```
derive_symbolic ──► reference.npz ─────┐
                                       ├─► train_pinn ─► pinn_weights.pth ─► generate_plots ─► PDF + HTML
ingest_and_vectorize ─► processed_grid.npy ─┘
analyze_stability ──► stability_report.txt
```

Snakemake résout automatiquement l'ordre d'exécution à partir du graphe des dépendances entrées/sorties
et ne réexécute que les règles dont les entrées ont changé.

### 9.2 Intégration continue (GitHub Actions)

Le workflow se déclenche sur `push` et `pull_request` vers `main`. Il installe `uv`
(`astral-sh/setup-uv@v5`, cache activé), puis enchaîne : **lint (ruff) → typage (mypy --strict) →
tests + couverture (pytest)**. Une **matrice** valide le code sur Python 3.10, 3.11 et 3.12 en parallèle.
En cas de succès, les artéfacts scientifiques (`outputs/figures/`, `models/`) sont publiés via
`actions/upload-artifact`.

### 9.3 Défi CI/CD — validation expérimentale (Exercice 9.2)

Conformément à l'Exercice 9.2, on a provoqué **volontairement** une défaillance du pipeline pour
vérifier que la barrière `mypy` bloque effectivement une régression de typage *avant* toute exécution
coûteuse. Le scénario a été rejoué sur la branche `feat/residual-pipeline-entrypoint`, puis intégré à
`main` par *pull request*.

**1. Introduction de l'erreur** — commit `feat(core): evaluate physics residual in the ingestion entry point`.
Le point d'accès `numerical_core.main()` passe par erreur le *chemin* du fichier capteur (une `str`) à
`vectorized_residual`, dont la signature attend un `np.ndarray`.

**2. Blocage en CI à l'étape `mypy`** — `ruff` passe (exit 0), `mypy --strict` échoue (exit 1) :

```
src/numerical_core.py:96: error: Argument 1 to "vectorized_residual" has incompatible type
    "str"; expected "ndarray[...]"  [arg-type]
src/numerical_core.py:96: error: Argument 2 to "vectorized_residual" has incompatible type
    "str"; expected "ndarray[...]"  [arg-type]
Found 2 errors in 1 file (checked 8 source files)
```

> **Note méthodologique.** Un premier essai passant directement `args.sensors` n'était **pas** détecté :
> les attributs d'un `argparse.Namespace` sont typés `Any`, compatible avec tout type. Il a fallu
> matérialiser la `str` dans une variable explicitement typée (`sensor_path: str`) pour que le contrôle
> statique la rejette — illustration concrète de la portée *et des limites* du typage graduel.

**3. Correction + test de régression** — commit `fix(core): guard vectorized_residual against non-ndarray inputs`.
Trois mesures :
- le résidu est désormais évalué sur la grille de discrétisation `(X, T)`, qui sont des `ndarray` ;
- un **garde-fou runtime** lève `TypeError` si `x`/`t` ne sont pas des `ndarray` (défense en profondeur,
  au-delà du seul contrôle statique) ;
- le test unitaire `test_vectorized_residual_rejects_non_array` **verrouille ce cas limite** contre toute
  réapparition.

La chaîne complète repasse au vert : `ruff` ✅ · `mypy --strict` ✅ · `pytest` ✅ (56 tests).

Ce cycle **rouge → vert** démontre que la CI remplit son rôle : aucune régression de typage ne peut
atteindre `main`, garantissant qu'une simulation de plusieurs heures ne sera jamais lancée sur un code
au contrat de types rompu. Les messages respectent la norme *Conventional Commits*
(`feat:` / `fix:` / `chore:`).

---

## 10. Réponses aux questions théoriques

**Q1 — Utilité de `enable-cache: true` (uv) et impact carbone.**
Le cache conserve les paquets et environnements résolus entre exécutions. Sans cache, chaque
déclenchement de CI retélécharge et recompile l'intégralité des dépendances (torch, numba, scipy…),
gaspillant bande passante, temps CPU et énergie. Le cache réduit drastiquement la durée des *runs* et
donc le **coût computationnel** et l'**empreinte carbone** de l'infrastructure — enjeu majeur quand des
milliers de *runs* s'accumulent à l'échelle d'un laboratoire.

**Q2 — Nécessité de `mypy --strict` avant des simulations longues.**
Sur un supercalculateur, une simulation peut tourner plusieurs heures avant qu'une erreur de type
(ex. passer une chaîne là où un `np.ndarray` est attendu) ne provoque un plantage — gaspillant des
heures de calcul coûteuses et une allocation de *nœuds* rare. La vérification statique stricte détecte
ces erreurs **avant** le lancement, à coût quasi nul, garantissant que les contrats de types entre
modules mathématiques sont respectés. C'est une assurance indispensable en HPC.

**Q3 — Justification de `np.isclose()` plutôt que `==` (Module 5).**
L'arithmétique flottante IEEE 754 introduit des erreurs d'arrondi à chaque opération : $A\alpha$ n'est
donc quasi jamais **exactement** égal à $b$. Une assertion `A @ alpha == b` échouerait presque toujours,
même pour une solution correcte. `np.isclose(atol, rtol)` teste l'égalité **à une tolérance près**,
calibrée sur la précision du type utilisé (plus large en `float16`, plus fine en `float64`), ce qui
constitue la seule validation numérique robuste.

---

## 11. Analyse critique : précision, accélération et fidélité du modèle

Le projet met en tension **performance** et **fidélité numérique** :

- **Précision des flottants.** La double précision (`float64`) est indispensable pour la vérité de
  référence : dès $\kappa(A) > 10^{16}$, même le `float64` échoue, et le `float16`/`float32` sont
  disqualifiés pour tout calcul mal conditionné. Le choix du type n'est jamais neutre.
- **Accélérations logicielles.** `fastmath` et la parallélisation apportent un gain réel (≈ 1.76× ici),
  mais `fastmath` sacrifie la conformité IEEE 754 et la reproductibilité bit-à-bit. Le compromis doit
  être **conscient et documenté** selon la sensibilité de l'application.
- **Fidélité de l'IA scientifique.** Le PINN, entraîné en `float32` sur GPU, hérite de ces compromis :
  sa précision (~10⁻² sur l'erreur absolue) est excellente pour de la modélisation, mais inférieure à un
  solveur direct `float64` bien conditionné. Son avantage est l'**absence de maillage** et la
  différentiation exacte, non la précision ultime.

En résumé, la chaîne « symbolique (vérité exacte) → PINN (approximation apprise) → validation contre la
vérité » constitue une méthodologie rigoureuse et moderne, à condition d'assumer explicitement les
compromis de précision introduits à chaque étape.

---

## 12. Conclusion et perspectives

Ce projet démontre la faisabilité d'un **pipeline de recherche scientifique entièrement reproductible**,
de la dérivation symbolique à l'IA multi-GPU, validé automatiquement par CI/CD. Les objectifs sont
atteints : environnement scellé sans *root*, qualité stricte (0 ruff, 0 mypy strict, 97 % de couverture),
et pipeline exécutable en une commande produisant des artéfacts vérifiables.

**Perspectives :**
- passage à l'échelle du PINN via le parallélisme de données (DDP) sur plusieurs GPU ;
- substitution de la matrice de Hilbert par des noyaux RBF pour l'interpolation ;
- extension à des EDP non linéaires à coefficients variables ;
- intégration de bibliothèques HPC sous-jacentes (PETSc) pour les solveurs linéaires.

---

## 13. Annexe — Procédure de réplication

```bash
# Environnement (sans sudo)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Chaîne qualité (identique à la CI)
uv run ruff check src/ tests/
uv run mypy --strict src/
USE_NUMBA=False uv run pytest --cov=src --cov-report=term-missing tests/

# Pipeline scientifique complet
uv run snakemake --cores all
```

**Artéfacts produits :** `outputs/reference.npz`, `data/processed_grid.npy`,
`outputs/stability_report.txt`, `models/pinn_weights.pth`,
`outputs/figures/solution_surface.pdf`, `outputs/figures/visualisation_pinn_interactive.html`.
