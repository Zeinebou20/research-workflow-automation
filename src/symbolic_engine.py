# mypy: disable-error-code="import-untyped"
from sympy import Expr, Symbol, sin, diff

def get_derivative_of_sin_x2(x: Symbol) -> Expr:
    return diff(sin(x**2), x)

if __name__ == "__main__":
    x_sym: Symbol = Symbol('x')
    derivative: Expr = get_derivative_of_sin_x2(x_sym)
    print(f": {derivative}")