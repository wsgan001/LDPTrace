import random
from typing import List, Tuple
import utils
import numpy as np
from grid import Grid, GridMap
import trajectory
import map_func
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


class Query:
    def __init__(self):
        pass

    def point_query(self, db):
        raise NotImplementedError


class GridQuery(Query):
    def __init__(self,
                 grid_map: GridMap,
                 size: int):
        super().__init__()
        # Randomly select left-top grid
        start_i = random.randint(0, len(grid_map.map) - size)
        start_j = random.randint(0, len(grid_map.map[0]) - size)

        self.region = []
        for i in range(size):
            for j in range(size):
                self.region.append(grid_map.map[start_i + i][start_j + j])

    def point_query(self, db: List[List[Grid]]):
        count = 0
        for t in db:
            for g in t:
                if g in self.region:
                    count += 1
        return count


class Pattern:
    def __init__(self, grids: List[Grid]):
        self.grids = grids

    @property
    def size(self):
        return len(self.grids)

    def __eq__(self, other):
        if other is None:
            return False
        if not type(other) == Pattern:
            return False
        if not other.size == self.size:
            return False

        for i in range(self.size):
            if not self.grids[i].index == other.grids[i].index:
                return False

        return True

    def __hash__(self):
        prime = 31
        result = 1
        for g in self.grids:
            result = result * prime + g.__hash__()

        return result


def calculate_point_query(orig_db,
                          syn_db,
                          queries: List[Query],
                          sanity_bound=0.01):
    actual_ans = list()
    syn_ans = list()

    total_points = np.sum([len(t) for t in orig_db])

    for q in queries:
        actual_ans.append(q.point_query(orig_db))
        syn_ans.append(q.point_query(syn_db))

    actual_ans = np.asarray(actual_ans)
    syn_ans = np.asarray(syn_ans)

    # Error = |actual-syn| / max{actual, 1% * len(db)}
    numerator = np.abs(actual_ans - syn_ans)
    # numerator = syn_ans - actual_ans
    denominator = np.asarray([max(actual_ans[i], total_points * sanity_bound) for i in range(len(actual_ans))])
    # denominator = actual_ans

    error = numerator / denominator

    return np.mean(error)


def calculate_coverage_kendall_tau(orig_db: List[List[Grid]],
                                   syn_db: List[List[Grid]],
                                   grid_map: GridMap):
    actual_counts = np.zeros(grid_map.size)
    syn_counts = np.zeros(grid_map.size)

    # For each grid, find how many trajectories pass through it
    for i in range(len(grid_map.map)):
        for j in range(len(grid_map.map[0])):
            g = grid_map.map[i][j]
            index = map_func.grid_index_map_func(g, grid_map)
            for t in orig_db:
                actual_counts[index] += trajectory.pass_through(t, g)
            for t in syn_db:
                syn_counts[index] += trajectory.pass_through(t, g)

    concordant_pairs = 0
    reversed_pairs = 0
    for i in range(grid_map.size):
        for j in range(i + 1, grid_map.size):
            if actual_counts[i] > actual_counts[j]:
                if syn_counts[i] > syn_counts[j]:
                    concordant_pairs += 1
                else:
                    reversed_pairs += 1
            if actual_counts[i] < actual_counts[j]:
                if syn_counts[i] < syn_counts[j]:
                    concordant_pairs += 1
                else:
                    reversed_pairs += 1

    denominator = grid_map.size * (grid_map.size - 1) / 2
    return (concordant_pairs - reversed_pairs) / denominator


def calculate_diameter_error(orig_db: List[List[Tuple[float, float]]],
                             syn_db: List[List[Tuple[float, float]]],
                             bucket_num=20, multi=False):
    if multi:
        pool = multiprocessing.Pool(CORES)
        orig_diameter = pool.map(trajectory.get_diameter, orig_db)
        pool.close()
        pool = multiprocessing.Pool(CORES)
        syn_diameter = pool.map(trajectory.get_diameter, syn_db)
        pool.close()
    else:
        orig_diameter = [trajectory.get_diameter(t) for t in orig_db]
        syn_diameter = [trajectory.get_diameter(t) for t in syn_db]

    bucket_size = (max(orig_diameter) - min(orig_diameter)) / bucket_num

    orig_count = np.zeros(bucket_num)
    syn_count = np.zeros(bucket_num)
    for i in range(bucket_num):
        start = i * bucket_size
        end = start + bucket_size

        for d in orig_diameter:
            if start <= d <= end:
                orig_count[i] += 1
        for d in syn_diameter:
            if start <= d <= end:
                syn_count[i] += 1

    # Normalization
    orig_count /= np.sum(orig_count)
    syn_count /= np.sum(syn_count)

    return utils.jensen_shannon_distance(orig_count, syn_count)


def calculate_length_error(orig_db: List[List[Tuple[float, float]]],
                           syn_db: List[List[Tuple[float, float]]],
                           bucket_num=20):
    orig_length = [trajectory.get_travel_distance(t) for t in orig_db]
    syn_length = [trajectory.get_travel_distance(t) for t in syn_db]

    bucket_size = (max(orig_length) - min(orig_length)) / bucket_num

    orig_count = np.zeros(bucket_num)
    syn_count = np.zeros(bucket_num)
    for i in range(bucket_num):
        start = i * bucket_size
        end = start + bucket_size

        for d in orig_length:
            if start <= d <= end:
                orig_count[i] += 1
        for d in syn_length:
            if start <= d <= end:
                syn_count[i] += 1

    # Normalization
    orig_count /= np.sum(orig_count)
    syn_count /= np.sum(syn_count)

    return utils.jensen_shannon_distance(orig_count, syn_count)


def mine_patterns(db: List[List[Grid]], min_size=2, max_size=8):
    """
    Find all patterns of size between min_size and max_size
    :return: Dict[Pattern, int]: count of each pattern
    """
    pattern_dict = {}
    for curr_size in range(min_size, max_size + 1):
        for t in db:
            for i in range(0, len(t) - curr_size + 1):
                p = Pattern(t[i: i + curr_size])
                try:
                    pattern_dict[p] += 1
                except KeyError:
                    pattern_dict[p] = 1

    return pattern_dict


def calculate_pattern_f1_error(orig_pattern,
                               syn_pattern,
                               k=100):
    sorted_orig = sorted(orig_pattern.items(), key=lambda x: x[1], reverse=True)
    sorted_syn = sorted(syn_pattern.items(), key=lambda x: x[1], reverse=True)

    orig_top_k = [x[0] for x in sorted_orig][:k]
    syn_top_k = [x[0] for x in sorted_syn][:k]

    count = 0
    for p1 in syn_top_k:
        if p1 in orig_top_k:
            count += 1

    precision = count / k
    recall = count / k

    return 2 * precision * recall / (precision + recall)


def calculate_pattern_support(orig_pattern, syn_pattern, k=100):
    sorted_orig = sorted(orig_pattern.items(), key=lambda x: x[1], reverse=True)
    orig_top_k = [x[0] for x in sorted_orig][:k]

    error = 0
    for i in range(len(orig_top_k)):
        p: Pattern = orig_top_k[i]
        orig_support = orig_pattern[p]
        try:
            syn_support = syn_pattern[p]
        except KeyError:
            syn_support = 0
        error += np.abs(orig_support-syn_support)/orig_support

    return error / k
