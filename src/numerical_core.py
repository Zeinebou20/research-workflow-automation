import numpy as np

def square_matrix(x: np.ndarray) -> np.ndarray:
    # Handling numpy's dynamic return type for Mypy strict mode
    return np.square(x)  # type: ignore[no-any-return]
