"""Module 3 : Calcul symbolique de l'équation d'advection-diffusion.

Solution manufacturée u(x, t) = tanh(x - c*t) ; on en dérive le terme source
exact f(x, t) = u_t + c*u_x - nu*u_xx, exporté en code NumPy via lambdify.
"""
import argparse
import os
from typing import Callable, Tuple

import numpy as np
import sympy as sp


def symbolic_advection_diffusion() -> Tuple[Callable[..., object], Callable[..., object]]:
    """Retourne (u_func, f_func) lambdifiées pour évaluation NumPy vectorisée."""
    x, t, c, nu = sp.symbols("x t c nu")

    u = sp.tanh(x - c * t)

    du_dt = sp.diff(u, t)
    du_dx = sp.diff(u, x)
    d2u_dx2 = sp.diff(u, x, 2)

    f = du_dt + c * du_dx - nu * d2u_dx2

    func_u: Callable[..., object] = sp.lambdify((x, t, c, nu), u, "numpy")
    func_f: Callable[..., object] = sp.lambdify((x, t, c, nu), f, "numpy")

    return func_u, func_f


def main() -> None:
    """Matérialise la solution exacte u et le terme source f sur une grille de référence."""
    parser = argparse.ArgumentParser(description="Dérivation symbolique")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_points", type=int, default=100)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.05)
    args = parser.parse_args()

    u_func, f_func = symbolic_advection_diffusion()
    x = np.linspace(0.0, 1.0, args.n_points)
    t = np.linspace(0.0, 1.0, args.n_points)
    X, T = np.meshgrid(x, t)

    u_exact = np.asarray(u_func(X, T, args.c, args.nu), dtype=np.float64)
    f_source = np.asarray(f_func(X, T, args.c, args.nu), dtype=np.float64)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, X=X, T=T, u_exact=u_exact, f_source=f_source)
    print(f"Référence symbolique sauvegardée : {args.output}")


if __name__ == "__main__":
    main()
