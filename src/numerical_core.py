"""Module 4 : Maîtrise de NumPy et ingestion de données.

Manipulation bas niveau du ndarray (strides, layout mémoire, vues vs copies),
ingestion massive via l'API lazy de Polars, et application vectorisée (broadcasting)
du terme source résiduel sur la grille de discrétisation.
"""
from __future__ import annotations

import argparse
from typing import Any, Callable, cast, tuple 

import numpy as np
import polars as pl

from src.symbolic_derivation import symbolic_advection_diffusion


def explore_ndarray_properties() -> np.ndarray[Any, Any]:
    """Exercice 4.1 : explore shape/dtype/strides, contiguïté C/F et vues vs copies."""
    N, M = 100, 50
    grid = np.zeros((N, M), dtype=np.float64)

    print("")
    print(f"Shape: {grid.shape}")
    print(f"Dtype: {grid.dtype}")
    print(f"Strides: {grid.strides}")

    grid_c = np.array(grid, order="C")
    grid_f = np.array(grid, order="F")

    print(f"Is C-contiguous: {grid_c.flags['C_CONTIGUOUS']}")
    print(f"Is F-contiguous: {grid_f.flags['F_CONTIGUOUS']}")

    sub_grid = grid[::2, ::2]
    print(f"Is sub_grid a view? {sub_grid.base is grid}")

    return grid


def make_discretization_grid(
    n_points: int = 100,
    x_range: tuple[float, float] = (0.0, 1.0),
    t_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Construit la grille spatio-temporelle (X, T) de discrétisation."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    t = np.linspace(t_range[0], t_range[1], n_points)
    return cast(tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]], np.meshgrid(x, t))


def process_massive_data(file_path: str) -> np.ndarray[Any, Any]:
    """Exercice 4.2 : scan lazy Polars d'un CSV/Parquet, extraction des coords valides."""
    df = (
        pl.scan_parquet(file_path)
        if file_path.endswith(".parquet")
        else pl.scan_csv(file_path)
    )

    valid_data = df.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
    coords = valid_data.select(["lat", "lon"]).collect().to_numpy()

    return coords


def vectorized_residual(
    x: np.ndarray[Any, Any],
    t: np.ndarray[Any, Any],
    c: float,
    nu: float,
    func_f: Callable[..., object],
) -> np.ndarray[Any, Any]:
    """Applique la fonction résiduelle f par broadcasting (sans boucle for).

    Garde-fou runtime : x et t doivent être des ndarray. Passer un autre type
    (p. ex. un chemin str au lieu des coordonnées chargées) lève TypeError —
    verrou de régression complémentaire au typage statique de mypy.
    """
    if not isinstance(x, np.ndarray) or not isinstance(t, np.ndarray):
        raise TypeError(
            "vectorized_residual attend des np.ndarray pour x et t, "
            f"reçu ({type(x).__name__}, {type(t).__name__})."
        )
    return np.asarray(func_f(x, t, c, nu), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion et discrétisation")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sensors", type=str, default="")
    parser.add_argument("--n_points", type=int, default=100)
    args = parser.parse_args()

    if args.sensors:
        try:
            coords = process_massive_data(args.sensors)
            print(f"Coordonnées capteurs ingérées : {coords.shape}")
        except Exception as exc:  # noqa: BLE001
            print(f"Ingestion capteurs ignorée ({exc})")

    X, T = make_discretization_grid(args.n_points)
    grid = np.stack([X, T], axis=0)

    # Évaluation du résidu physique sur la grille de discrétisation (ndarray)
    _, f_func = symbolic_advection_diffusion()
    residual = vectorized_residual(X, T, 1.0, 0.05, f_func)
    print(f"Résidu moyen : {float(np.abs(residual).mean()):.4e}")

    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, grid)
    print(f"Grille de discrétisation sauvegardée : {args.output} {grid.shape}")


if __name__ == "__main__":
    main()
