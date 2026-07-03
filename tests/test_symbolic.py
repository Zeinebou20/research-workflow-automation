import sympy as sp
from src.symbolic_engine import get_derivative_of_sin_x2

def test_derivative():
    x = sp.symbols('x')
    expected_derivative = 2 * x * sp.cos(x**2)
    
    result = get_derivative_of_sin_x2()
    
    assert sp.simplify(result - expected_derivative) == 0