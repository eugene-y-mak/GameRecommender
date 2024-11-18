import numpy as np
from collaborative_filtering import Recommender
R = [
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
    [2, 1, 3, 0],
]

R = np.array(R)
# N: num of User
N = len(R)
# M: num of Movie
M = len(R[0])
# Num of Features
K = 3

P = np.random.rand(N, K)
Q = np.random.rand(M, K)

rec = Recommender()

nP, nQ = rec.matrix_factorization(R, P, Q, K)

nR = np.dot(nP, nQ.T)