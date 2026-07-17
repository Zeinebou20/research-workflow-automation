import argparse
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from src.numerical_core import (
    explore_ndarray_properties,
    main,
    make_discretization_grid,
    process_massive_data,
    vectorized_residual,
)


def test_explore_ndarray_properties():
    grid = explore_ndarray_properties()
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (100, 50)
    assert grid.dtype == np.float64
    assert np.all(grid == 0.0)


def test_make_discretization_grid():
    X, T = make_discretization_grid(n_points=20)
    assert X.shape == (20, 20)
    assert T.shape == (20, 20)
    assert X.min() == 0.0 and X.max() == 1.0


def test_process_massive_data_csv(tmp_path):
    csv_path = tmp_path / "coords.csv"
    csv_path.write_text("lat,lon\n1.0,2.0\n,3.0\n4.0,\n5.0,6.0\n")
    coords = process_massive_data(str(csv_path))
    assert coords.shape == (2, 2)
    np.testing.assert_array_equal(coords, np.array([[1.0, 2.0], [5.0, 6.0]]))


def test_process_massive_data_parquet(tmp_path):
    parquet_path = tmp_path / "coords.parquet"
    pl.DataFrame({"lat": [10.0, None, 20.0], "lon": [11.0, 12.0, None]}).write_parquet(parquet_path)
    coords = process_massive_data(str(parquet_path))
    assert coords.shape == (1, 2)
    np.testing.assert_array_equal(coords, np.array([[10.0, 11.0]]))


def test_process_massive_data_real_sample():
    """Le fichier capteur fourni dans data/raw_sensors doit être ingéré."""
    coords = process_massive_data("data/raw_sensors/sensors_sample.csv")
    assert isinstance(coords, np.ndarray)
    assert coords.shape[1] == 2
    # 1 ligne a lat manquante -> filtrée ; 9 lignes conservent lat & lon
    assert coords.shape[0] == 9


def test_vectorized_residual_broadcast():
    x = np.array([0.0, 1.0, 2.0])
    t = np.array([0.0, 0.5, 1.0])
    result = vectorized_residual(x, t, 1.0, 0.1, lambda x, t, c, nu: c * x + nu * t)
    np.testing.assert_allclose(result, 1.0 * x + 0.1 * t)


def test_vectorized_residual_rejects_non_array():
    """Cas limite (régression défi CI/CD 9.2) : une str au lieu d'un ndarray lève TypeError."""
    with pytest.raises(TypeError):
        vectorized_residual("bad_path", "bad_path", 1.0, 0.05, lambda x, t, c, nu: x)


def test_main_saves_grid(tmp_path):
    out = tmp_path / "grid.npy"
    mock_args = argparse.Namespace(
        output=str(out), sensors="data/raw_sensors/sensors_sample.csv", n_points=15
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        main()
    assert out.exists()
    grid = np.load(out)
    assert grid.shape == (2, 15, 15)


def test_main_handles_missing_sensors(tmp_path):
    out = tmp_path / "grid2.npy"
    mock_args = argparse.Namespace(output=str(out), sensors="data/does_not_exist.csv", n_points=8)
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        main()  # ne doit pas lever malgré le fichier capteur absent
    assert out.exists()
