import numpy as np
from typing import Any

# تحديد نوع المصفوفة بشكل صريح ليرضى عنه Mypy
def square_matrix(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.square(x)  # type: ignore[no-any-return]