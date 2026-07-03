import sympy as sp
# type: ignore[import-untyped]

def get_derivative_of_sin_x2(x: Expr) -> Expr:

    x = sp.symbols('x')
    f = sp.sin(x**2)
    
    df = sp.diff(f, x)
    
    return df

if __name__ == "__main__":
    derivative = get_derivative_of_sin_x2()
    print(f" {derivative}")