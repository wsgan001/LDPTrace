from grid import Grid, GridMap
from typing import List, Tuple
import numpy as np
from utils import dtw_distance
import trajectory
from scipy.spatial.kdtree import KDTree
from parse import args


def get_sniffed_traj(syn_db: List[List[Grid]], sniffed_region: List[Grid]):
    sniffed_traj = []
    for i in range(len(syn_db)):
        if set(sniffed_region).issubset(set(syn_db[i])):
            sniffed_traj.append(i)

    return sniffed_traj


def get_sniffed_points(trj: List[Tuple[float, float]], sniffed_region: List[Grid]):
    sniffed_points = []
    for point in trj:
        for g in sniffed_region:
            if g.in_cell(point):
                sniffed_points.append(point)
                break

    return sniffed_points


def re_identification_attack(orig_grid_db: List[List[Grid]], syn_grid_db: List[List[Grid]],
                             orig_db: List[List[Tuple[float, float]]],
                             syn_db: List[List[Tuple[float, float]]],
                             grid_map: GridMap):
    # sniffed_region = [grid_map.map[4][3],
    #                   grid_map.map[3][3],
    #                   grid_map.map[3][2],
    #                   grid_map.map[3][1]]
    sniffed_region = [grid_map.map[3][1],
                      grid_map.map[2][1],
                      grid_map.map[1][1],
                      grid_map.map[1][0]]
    if args.dataset == 'taxi':
        sniffed_region = [grid_map.map[4][3],
                          grid_map.map[4][2],
                          grid_map.map[4][1]]

    sniffed_syn_id = get_sniffed_traj(syn_grid_db, sniffed_region)
    sniffed_orig_id = get_sniffed_traj(orig_grid_db, sniffed_region)

    print(f'Orig: {len(sniffed_orig_id)}, Syn: {len(sniffed_syn_id)}')

    if not len(sniffed_orig_id):
        return

    distance_matrix = np.zeros((len(sniffed_syn_id), len(sniffed_orig_id)))
    for i in range(len(sniffed_syn_id)):
        sniffed_syn_points = get_sniffed_points(syn_db[sniffed_syn_id[i]], sniffed_region)
        for j in range(len(sniffed_orig_id)):
            sniffed_orig_points = get_sniffed_points(orig_db[sniffed_orig_id[j]], sniffed_region)
            distance_matrix[i][j] = dtw_distance(sniffed_syn_points, sniffed_orig_points)

    thetas = [100000, 120000, 150000, 200000, 250000]
    # Ks = [1, 2, 3, 4, 5, 10, 15]
    Ks = [2, 4, 6, 8, 10, 12, 14]

    for theta in thetas:
        count = np.zeros_like(Ks)
        for j in range(len(sniffed_orig_id)):
            dist = sorted(distance_matrix[:, j])
            for i, K in enumerate(Ks):
                if dist[0] > theta:
                    count += 1
                    break
                if dist[K - 1] < theta:
                    count[i] += 1
        print(f'Theta: {theta}')
        print(count / len(sniffed_orig_id))


def outlier_attack(orig_db: List[List[Tuple[float, float]]], syn_db: List[List[Tuple[float, float]]],
                   out_type='length'):
    # betas = [500, 1000, 2500, 5000, 10000, 30000, 100000]
    betas = [10, 20, 30, 40, 50, 80, 100, 150]
    if out_type == 'length':

        # Length outlier
        lengths = np.array([trajectory.get_travel_distance(t) for t in syn_db]).reshape(-1, 1)
        length_tree = KDTree(lengths)

        # dist: [n x 20], distance to k-nearest neighbor
        dist, _ = length_tree.query(lengths, k=20)
        # index_dist: (i, d), where i is the index of syn_trajectory, d is the distance to its kth neighbor
        index_dist = [(i, dist[i, -1]) for i in range(len(dist))]

        # find trajectories with largest dist to their kth neighbor
        index_dist = sorted(index_dist, key=lambda x: x[1], reverse=True)
        outliers = index_dist[:200]

        orig_lengths = np.array([trajectory.get_travel_distance(t) for t in orig_db]).reshape(-1, 1)
        orig_length_tree = KDTree(orig_lengths)

        # outlier_lengths = [lengths[outlier] for outlier, _ in outliers]
        outlier_lengths = lengths[[outlier for outlier, _ in outliers]]
        # for each outlier, do similarity search in orig_db
        outlier_dist, _ = orig_length_tree.query(outlier_lengths, k=20)

    elif out_type == 'start':

        # Trip start outlier
        starts = np.array([list(t[0]) for t in syn_db]).reshape(-1, 2)
        start_tree = KDTree(starts)

        dist, _ = start_tree.query(starts, k=20)
        index_dist = [(i, dist[i, -1]) for i in range(len(dist))]

        index_dist = sorted(index_dist, key=lambda x: x[1], reverse=True)
        outliers = index_dist[:200]

        orig_starts = np.array([list(t[0]) for t in orig_db]).reshape(-1, 2)
        orig_start_tree = KDTree(orig_starts)

        outlier_starts = starts[[outlier for outlier, _ in outliers]]
        outlier_dist, _ = orig_start_tree.query(outlier_starts, k=20)

    # Ks = [1, 2, 3, 4, 5, 10, 15]
    Ks = [2, 4, 6, 8, 10, 12, 14]

    for beta in betas:
        count = np.zeros_like(Ks)
        for i in range(len(outliers)):
            for j, k in enumerate(Ks):
                if outlier_dist[i, 0] > beta:
                    count += 1
                    break
                if outlier_dist[i, k - 1] < beta:
                    count[j] += 1

        print(f'Beta: {beta}')
        print(count / len(outliers))
