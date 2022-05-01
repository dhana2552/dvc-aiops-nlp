import numpy as np
from scipy.sparse import csr_matrix

A = np.array([
    [1,0,0,1,0,1],
    [0,0,2,0,0,4],
    [5,0,0,0,0,6]
])

print(A)
S = csr_matrix(A) #tuples the position of the non-zero values
print(S)
print(type(S))

B = S.todense()
print(B)