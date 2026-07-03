import sympy as sp
from sympy import Expr, Symbol, sin, diff

def get_derivative_of_sin_x2(x: Symbol) -> Expr:
   
    f: Expr = sin(x**2)
    df: Expr = diff(f, x)
    return df

if __name__ == "__main__":
    x_sym: Symbol = Symbol('x')
    
    derivative: Expr = get_derivative_of_sin_x2(x_sym)
    
    print(f"المشتقة هي: {derivative}")