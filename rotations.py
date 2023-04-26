import numpy as np
from math import cos, sin

def construct_sub_matrix(i, j, dim, theta):
    """
    Sub-matrix used to construct a full rotation matrix
    """
    sub_matrix = np.diag(np.ones(dim))
    sub_matrix[i][j] = -sin(theta)
    sub_matrix[j][i] = sin(theta)
    sub_matrix[i][i] = sub_matrix[j][j] = cos(theta)
    return sub_matrix

def construct_rotation(thetas:'list[float]', dim:int):
    """
        constructs a rotation matrix of dimensionality given by the length of `thetas`

        Thetas are
    """
    if not len(thetas)==int(dim*(dim-1)/2):
        print(len(thetas))
        print(dim*(dim-1)/2)
        raise ValueError("Number of thetas should be n*(n-1)/2 where `n` is the number of dimensions")

    M = np.diag(np.ones(dim))

    count = 0
    for i in range(dim):
        for j in range(dim):
            if j>=i:
                continue
            
            M = np.matmul(M, construct_sub_matrix(i, j, dim, thetas[count]))
            count += 1
    return M
