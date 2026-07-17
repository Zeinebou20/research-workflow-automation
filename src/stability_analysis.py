"""Module 5 : Stabilité, conditionnement et analyse des erreurs.

Étude de la sensibilité de A·α = b (A = matrice de Hilbert mal conditionnée) :
conditionnement κ(A), comparaison float16/32/64, propagation d'une perturbation
ε sur b reliée à κ(A), et validation robuste du résidu via np.isclose.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import hilbert


def condition_number(n: int) -> float:
    """Nombre de conditionnement κ(A) = ‖A‖·‖A⁻¹‖ de la matrice de Hilbert n×n."""
    return float(np.linalg.cond(hilbert(n)))


def compute_residual(A: np.ndarray, alpha: np.ndarray, b: np.ndarray) -> float:
    """Résidu strict r = ‖A·α − b‖₂ (Exercice 5.1.4)."""
    return float(np.linalg.norm(A @ alpha - b))


def solve_and_validate(
    A: np.ndarray, b: np.ndarray, dtype: type, atol: float, rtol: float
) -> tuple[np.ndarray, bool]:
    """Résout le système dans la précision `dtype` et valide via np.isclose.

    L'égalité stricte (==) est proscrite car les erreurs d'arrondi IEEE 754 rendent
    Aα rarement exactement égal à b ; np.isclose avec (atol, rtol) adaptés est robuste.
    """
    A_dt: np.ndarray = A.astype(dtype)
    b_dt: np.ndarray = b.astype(dtype)
    # numpy.linalg ne supporte pas float16 : on résout en float64 puis on projette
    # dans la précision cible pour observer la perte de précision (Exercice 5.1.2).
    solve_dtype = np.float64 if dtype == np.float16 else dtype
    alpha_full = np.linalg.solve(A.astype(solve_dtype), b.astype(solve_dtype))
    with np.errstate(over="ignore", invalid="ignore"):
        # En précision réduite (float16), le débordement est attendu et signale
        # précisément l'instabilité — on le neutralise pour garder une sortie propre.
        alpha: np.ndarray = alpha_full.astype(dtype)
        is_valid = bool(np.all(np.isclose(A_dt @ alpha, b_dt, atol=atol, rtol=rtol)))
    return alpha, is_valid


def perturbation_analysis(n: int, eps: float = 1e-7) -> dict[str, float]:
    """Perturbe b de ε et relie l'amplification de l'erreur à κ(A) (Exercice 5.1.3).

    Borne théorique : ‖Δα‖/‖α‖ ≤ κ(A) · ‖Δb‖/‖b‖.
    """
    A = hilbert(n)
    b = np.ones(n)

    alpha = np.linalg.solve(A, b)
    delta_b = eps * np.random.randn(n)
    alpha_pert = np.linalg.solve(A, b + delta_b)

    rel_error_alpha = float(np.linalg.norm(alpha_pert - alpha) / np.linalg.norm(alpha))
    rel_error_b = float(np.linalg.norm(delta_b) / np.linalg.norm(b))
    kappa = float(np.linalg.cond(A))

    return {
        "kappa": kappa,
        "rel_error_b": rel_error_b,
        "rel_error_alpha": rel_error_alpha,
        "theoretical_bound": kappa * rel_error_b,
    }


def analyze_stability(n_values: list[int] | None = None) -> None:
    """Balaye les dimensions et affiche κ(A) et la validité selon la précision."""
    if n_values is None:
        n_values = [5, 10, 15, 20, 25]

    for n in n_values:
        A = hilbert(n)
        print(f"\nDimension n={n}, Condition Number={condition_number(n):.2e}")

        b_true = np.ones(n)
        for dtype in [np.float16, np.float32, np.float64]:
            atol = 1e-3 if dtype == np.float16 else 1e-8
            try:
                _, is_valid = solve_and_validate(A, b_true, dtype, atol=atol, rtol=0.0)
                print(f"  Dtype {dtype.__name__}: Solution valid? {is_valid}")
            except Exception:
                print(f"  Dtype {dtype.__name__}: Error in solving system")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse de stabilité")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    lines: list[str] = ["=== Analyse de stabilité (Module 5) ===", ""]
    for n in [5, 10, 15, 20, 25]:
        stats = perturbation_analysis(n)
        lines.append(
            f"n={n:2d} | kappa={stats['kappa']:.2e} | "
            f"err_alpha={stats['rel_error_alpha']:.2e} | "
            f"borne_theorique={stats['theoretical_bound']:.2e}"
        )

    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Rapport de stabilité écrit : {args.output}")


if __name__ == "__main__":
    main()
