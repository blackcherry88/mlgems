import numpy as np

def xor_swizzling(A: np.array):
    O = np.zeros_like(A)
    S, _ = A.shape

    for i in range(S):
        for j in range(S):
            O[i, j] = A[i, i^j]
    return O
