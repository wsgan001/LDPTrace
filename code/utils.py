import numpy as np
from typing import Tuple, List
from math import sqrt


def precompute_markov(one_level_mat: np.ndarray, max_level: int):
    """
    Estimate N-level markov probabilities using one-level prob
    :return: list of markov matrices
    """
    # Use 1-level matrix as a placeholder for 0-level matrix
    markov_mats = [one_level_mat, one_level_mat]

    for i in range(2, max_level + 1):
        prev_level_mat = markov_mats[i - 1]

        # Use matrix multiply to calculate next-level matrix
        markov_mats.append(np.matmul(prev_level_mat, one_level_mat))

    return markov_mats


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def dtw_distance(t0: List[Tuple[float, float]], t1: List[Tuple[float, float]]):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : List[Tuple[float,float]]
    param t1 : List[Tuple[float,float]]
    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """

    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = np.inf
    C[0, 1:] = np.inf
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            C[i, j] = euclidean_distance(t0[i - 1], t1[j - 1]) + min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
    dtw = C[n0, n1]
    return dtw


def point_to_line_distance(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]):
    """
    Euclidean distance between p0 to p1p2
    """
    # A = y2 - y1
    A = p2[1] - p1[1]
    # B = x1 - x2
    B = p1[0] - p2[0]
    # C = x1(y1-y2) + y1(x2-x1)
    C = p1[0] * (p1[1] - p2[1]) + p1[1] * (p2[0] - p1[0])

    return np.abs(A * p0[0] + B * p0[1] + C) / (np.sqrt(A ** 2 + B ** 2))


def kl_divergence(prob1, prob2):
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)

    kl = np.log((prob1 + 1e-8) / (prob2 + 1e-8)) * prob1

    return np.sum(kl)


def jensen_shannon_distance(prob1, prob2):
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)

    avg_prob = (prob1 + prob2) / 2

    return 0.5 * kl_divergence(prob1, avg_prob) + 0.5 * kl_divergence(prob2, avg_prob)




