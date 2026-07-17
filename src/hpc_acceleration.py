"""Module 6 : Optimisation, accélération JIT (Numba) et parallélisme (Joblib).

Filtrage local lourd accéléré par @njit, diagnostic de profiling (cProfile + timeit),
et exploration parallèle de l'espace des paramètres (c, nu) via Joblib.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Callable, List, Sequence, Tuple, cast

import numpy as np
from numba import njit, prange
from typing import Any

use_numba = os.environ.get("USE_NUMBA", "True") == "True"

Kernel = Callable[..., Any]


def _identity(func: Kernel) -> Kernel:
    """Décorateur neutre (désactive njit pendant les tests)."""
    return func


_decorator: Callable[[Kernel], Kernel] = (
    njit(parallel=True, fastmath=True) if use_numba else _identity
)


@_decorator
def heavy_computation_optimized(grid:np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Opérateur de filtrage local : result[i, j] = sin(grid) + cos(grid)."""
    n, m = grid.shape
    result: np.ndarray[Any, Any] = np.zeros((n, m))

    for i in prange(n):  # type: ignore[no-untyped-call, attr-defined]
        for j in range(m):
            result[i, j] = np.sin(grid[i, j]) + np.cos(grid[i, j])

    return result


def heavy_computation_numpy(grid: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Référence NumPy vectorisée pour comparaison de performance."""
    return cast(np.ndarray[Any, Any], np.sin(grid) + np.cos(grid))


def profile_computation(size: int = 400) -> str:
    """Exercice 6.1 : profile NumPy vs Numba avec cProfile et timeit."""
    import cProfile
    import io
    import pstats
    import timeit

    grid = np.random.rand(size, size)
    heavy_computation_optimized(grid)  # warm-up JIT

    t_numpy = timeit.timeit(lambda: heavy_computation_numpy(grid), number=5)
    t_numba = timeit.timeit(lambda: heavy_computation_optimized(grid), number=5)

    profiler = cProfile.Profile()
    profiler.enable()
    heavy_computation_numpy(grid)
    profiler.disable()

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(5)

    report = (
        f"=== Profiling (grille {size}x{size}) ===\n"
        f"NumPy vectorisé : {t_numpy:.4f}s (5 exec)\n"
        f"Numba optimisé  : {t_numba:.4f}s (5 exec)\n\n"
        f"{stream.getvalue()}"
    )
    return report


def _simulate_param(params: Tuple[float, float], grid: np.ndarray[Any, Any]) -> float:
    """Simulation pure-NumPy pour un couple (c, nu) : norme d'un champ résiduel."""
    c, nu = params
    field = np.sin(c * grid) - nu * np.cos(grid)
    return float(np.linalg.norm(field))


def parameter_sweep(
    param_grid: Sequence[Tuple[float, float]],
    n_workers: int = -1,
    grid_size: int = 100,
) -> List[float]:
    """Exercice 6.3 : explore l'espace (c, nu) en parallèle via Joblib."""
    from joblib import Parallel, delayed

    grid = np.random.rand(grid_size, grid_size)
    results: List[float] = Parallel(n_jobs=n_workers)(
        delayed(_simulate_param)(p, grid) for p in param_grid
    )
    return list(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Accélération HPC")
    parser.add_argument("--size", type=int, default=400)
    args = parser.parse_args()

    large_grid = np.random.rand(args.size, args.size)

    start = time.time()
    heavy_computation_numpy(large_grid)
    print(f"NumPy vectorization time: {time.time() - start:.4f}s")

    heavy_computation_optimized(large_grid)  # warm-up
    start = time.time()
    heavy_computation_optimized(large_grid)
    print(f"Numba optimized time: {time.time() - start:.4f}s")


if __name__ == "__main__":
    main()
