# Generalized form of Lp Norm
import numpy as np


def lp_norm(x, p):
    if p >= 1:
        return np.sum(np.abs(x) ** p) ** (1 / p)
    

def test_lp_norm():
    x = np.array([1, 2, 3, 4, 5])
    for i in range(1, 6):
        c = lp_norm(x, i)
        if c== np.linalg.norm(x, i):
            print("Success")
        else:
            print("Failure")
        
test_lp_norm()