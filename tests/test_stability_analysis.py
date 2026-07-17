import argparse
from unittest.mock import patch

import numpy as np
from scipy.linalg import hilbert

from src.stability_analysis import (
    analyze_stability,
    compute_residual,
    condition_number,
    main,
    perturbation_analysis,
    solve_and_validate,
)


def test_condition_number_grows_with_n():
    conds = [condition_number(n) for n in (5, 10, 15)]
    assert conds[0] < conds[1] < conds[2]
    assert conds[0] > 1000  # Hilbert est mal conditionnée


def test_compute_residual_zero_for_exact_solution():
    A = hilbert(5)
    b = np.ones(5)
    alpha = np.linalg.solve(A, b)
    assert compute_residual(A, alpha, b) < 1e-8


def test_solve_and_validate_float64():
    A = hilbert(5)
    b = np.ones(5)
    alpha, is_valid = solve_and_validate(A, b, np.float64, atol=1e-8, rtol=0.0)
    assert alpha.shape == (5,)
    assert is_valid is True


def test_solve_and_validate_float16_less_accurate():
    """En float16, la validation stricte échoue là où float64 réussit."""
    A = hilbert(12)
    b = np.ones(12)
    _, valid16 = solve_and_validate(A, b, np.float16, atol=1e-8, rtol=0.0)
    assert valid16 is False


def test_perturbation_analysis_respects_condition_bound():
    """L'amplification d'erreur doit respecter la borne ‖Δα‖/‖α‖ ≤ κ(A)·‖Δb‖/‖b‖."""
    np.random.seed(0)
    stats = perturbation_analysis(10, eps=1e-7)
    assert stats["rel_error_alpha"] <= stats["theoretical_bound"] * 1.0001
    assert stats["kappa"] > 0
    assert stats["rel_error_b"] > 0


def test_analyze_stability_runs(capsys):
    analyze_stability(n_values=[5, 10])
    out = capsys.readouterr().out
    assert "Condition Number" in out


def test_analyze_stability_default_reaches_25(capsys):
    analyze_stability()
    assert "n=25" in capsys.readouterr().out


def test_main_writes_report(tmp_path):
    out = tmp_path / "report.txt"
    mock_args = argparse.Namespace(output=str(out), seed=42)
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        main()
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "kappa" in content
    assert "n=25" in content
