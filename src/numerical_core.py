import numpy as np
from numpy.typing import NDArray

def square_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.square(x)  # type: ignore[no-any-return]