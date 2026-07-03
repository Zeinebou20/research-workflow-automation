# type: ignore[import-untyped]
from sympy import Expr, Symbol, sin, diff

def get_derivative_of_sin_x2(x: Symbol) -> Expr:
    # نقوم بإجراء الحساب مباشرة
    return diff(sin(x**2), x)

if __name__ == "__main__":
    # تعريف الرمز
    x_sym: Symbol = Symbol('x')
    derivative: Expr = get_derivative_of_sin_x2(x_sym)
    print(f"المشتقة هي: {derivative}")