import runpy

import sympy as sp

from src.symbolic_engine import get_derivative_of_sin_x2


def test_derivative():
    x = sp.symbols('x')
    expected_derivative = 2 * x * sp.cos(x**2)

    result = get_derivative_of_sin_x2(x)

    # Comparaison symbolique robuste (simplification de la différence)
    assert sp.simplify(result - expected_derivative) == 0


def test_derivative_returns_expr():
    x = sp.symbols('x')
    result = get_derivative_of_sin_x2(x)
    assert isinstance(result, sp.Expr)


def test_derivative_numeric_value():
    """La dérivée évaluée en x=1 doit valoir 2*cos(1)."""
    x = sp.symbols('x')
    result = get_derivative_of_sin_x2(x)
    value = float(result.subs(x, 1))
    assert abs(value - 2 * sp.cos(1)) < 1e-9


def test_symbolic_engine_main_block(capsys):
    runpy.run_module("src.symbolic_engine", run_name="__main__")
    captured = capsys.readouterr()
    # Le bloc __main__ imprime la dérivée
    assert "cos" in captured.out
