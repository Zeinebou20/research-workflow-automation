import importlib

import numpy as np

from src import hpc_acceleration
from src.hpc_acceleration import (
    heavy_computation_numpy,
    heavy_computation_optimized,
    parameter_sweep,
    profile_computation,
)


def test_heavy_computation_correctness():
    grid = np.array([[0.0, np.pi / 2], [np.pi, 0.0]])
    result = heavy_computation_optimized(grid)
    expected = np.array([[1.0, 1.0], [-1.0, 1.0]])
    assert np.allclose(result, expected)


def test_optimized_matches_numpy_reference():
    grid = np.random.rand(20, 15)
    assert np.allclose(heavy_computation_optimized(grid), heavy_computation_numpy(grid))


def test_heavy_computation_shape_and_bounds():
    grid = np.random.rand(10, 10)
    res = heavy_computation_optimized(grid)
    assert res.shape == (10, 10)
    assert not np.isnan(res).any()
    assert np.all(res <= 2.0)  # sin+cos borné par sqrt(2)


def test_parameter_sweep_joblib():
    """Exercice 6.3 : le balayage parallèle renvoie un résultat par combinaison."""
    combos = [(1.0, 0.01), (2.0, 0.05), (0.5, 0.1), (1.5, 0.02)]
    results = parameter_sweep(combos, n_workers=2, grid_size=30)
    assert len(results) == len(combos)
    assert all(isinstance(r, float) and r > 0 for r in results)


def test_profile_computation_reports_timings():
    report = profile_computation(size=50)
    assert "NumPy" in report
    assert "Numba" in report


def test_numba_enabled_branch(monkeypatch):
    """Recharge le module avec USE_NUMBA=True pour couvrir la branche njit réelle."""
    monkeypatch.setenv("USE_NUMBA", "True")
    reloaded = importlib.reload(hpc_acceleration)
    try:
        grid = np.random.rand(8, 8)
        result = reloaded.heavy_computation_optimized(grid)
        assert np.allclose(result, np.sin(grid) + np.cos(grid))
    finally:
        monkeypatch.setenv("USE_NUMBA", "False")
        importlib.reload(hpc_acceleration)


def test_main_block(monkeypatch, capsys):
    import runpy

    monkeypatch.setenv("USE_NUMBA", "False")
    import sys
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["hpc_acceleration.py", "--size", "40"])
        runpy.run_module("src.hpc_acceleration", run_name="__main__")
    out = capsys.readouterr().out
    assert "time" in out.lower()
    importlib.reload(hpc_acceleration)
