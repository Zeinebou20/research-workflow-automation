import numpy as np
from src.numerical_core import square_matrix

def test_square_matrix():
    
    x = np.array([1, 2, 3])
    expected = np.array([1, 4, 9])
    
    result = square_matrix(x)
    
    np.testing.assert_array_equal(result, expected)