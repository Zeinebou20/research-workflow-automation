import argparse
from unittest.mock import patch

import numpy as np

from src.symbolic_derivation import main, symbolic_advection_diffusion


def test_symbolic_advection_diffusion_execution():
    u_func, f_func = symbolic_advection_diffusion()
    x, t, c, nu = 0.0, 0.0, 1.0, 0.1
    u_val = u_func(x, t, c, nu)
    f_val = f_func(x, t, c, nu)
    assert np.isclose(u_val, 0.0)  # tanh(0) = 0
    assert isinstance(u_val, (float, np.floating))
    assert isinstance(f_val, (float, np.floating))


def test_functionality_with_arrays():
    u_func, _ = symbolic_advection_diffusion()
    x = np.array([0.0, 1.0])
    t = np.array([0.0, 0.0])
    u_vals = u_func(x, t, 1.0, 0.1)
    assert u_vals.shape == (2,)


def test_solution_satisfies_pde():
    """Pour u=tanh(x-ct) : u_t = -c*u_x, donc f = -nu*u_xx."""
    _, f_func = symbolic_advection_diffusion()
    x, t, c, nu = 0.5, 0.2, 1.3, 0.07
    z = x - c * t
    d2u_dx2 = -2 * np.tanh(z) * (1 - np.tanh(z) ** 2)
    expected_f = -nu * d2u_dx2
    assert np.isclose(f_func(x, t, c, nu), expected_f)


def test_main_saves_reference(tmp_path):
    out = tmp_path / "reference.npz"
    mock_args = argparse.Namespace(output=str(out), n_points=20, c=1.0, nu=0.05)
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        main()
    assert out.exists()
    data = np.load(out)
    assert set(data.files) == {"X", "T", "u_exact", "f_source"}
    assert data["u_exact"].shape == (20, 20)
