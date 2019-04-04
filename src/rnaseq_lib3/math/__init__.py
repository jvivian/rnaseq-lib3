from itertools import combinations
from typing import Union, List, Tuple, Set, Dict

import numpy as np
from pandas import DataFrame
from statsmodels.stats.stattools import medcouple


# Outlier
def iqr_bounds(ys: List[Union[float, int]], whis: float = 1.5) -> Tuple[float, float]:
    """
    Return upper and lower bound for an array of values
    Lower bound: Q1 - (IQR * 1.5)
    Upper bound: Q3 + (IQR * 1.5)

    Args:
        ys: Array of values
        whis: Constant value for whisker

    Returns:
        Upper and lower bound
    """
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * whis)
    upper_bound = quartile_3 + (iqr * whis)
    return lower_bound, upper_bound


def std_bounds(ys: List[Union[float, int]], num_std: int = 2) -> Tuple[float, float]:
    """Return upper and lower bounds for an array of values"""
    u = np.mean(ys)
    std = np.std(ys)
    upper = float(u + num_std * std)
    lower = float(u - num_std * std)
    return upper, lower


def adjusted_whisker_skew(ys: List[Union[float, int]]) -> float:
    """Calculate IQR whisker modifier based on skew (medcouple)"""
    # Cannot compute medcouple for arrays of 1 or fewer
    if len(ys) <= 1:
        return 1.5

    # Calculate medcouple and adjusted whisker based on skew
    mc = float(medcouple(ys))
    if mc >= 0:
        return 1.5 * np.exp(3 * mc)
    else:
        return 1.5 * np.exp(4 * mc)


# Differential Expression
def log2fc(a: Union[float, np.array], b: Union[float, np.array], pad: float = 0.001) -> Union[float, np.array]:
    """
    Calculate the log2 fold change between two arrays or floats.
    a and b cannot be, nor contain, values less than 0
    """
    return np.log2(a + pad) - np.log2(b + pad)


# Normalization
def l2norm(x: float, pad: float = 1.0) -> float:
    """Log2 normalization function"""
    return np.log2(x + pad)


def min_max_normalize(df: DataFrame) -> DataFrame:
    """Rescale features to the range of [0, 1]"""
    return (df - df.min()) / (df.max() - df.min())


def mean_normalize(df: DataFrame) -> DataFrame:
    """Normalizes data to mean of 0 and std of 1"""
    return (df - df.mean()) / df.std()


def softmax(df: DataFrame) -> DataFrame:
    """Normalizes columns to sum to 1"""
    return df.divide(df.sum())


# GMM Fits
def find_gaussian_intersection(m1, m2, std1, std2):
    """
    Given parameters for two gaussian distributions, identify the intersection(s)
    :param float m1: Mean for first Gaussian
    :param float m2: Mean for second Gaussian
    :param float std1: Standard deviation for first Gaussian
    :param float std2: Standard deviation for second Gaussian
    :return: Intersection(s) between Gaussian distributions
    :rtype: list(float,)
    """
    # Define systems of equations
    m1, m2, std1, std2 = float(m1), float(m2), float(std1), float(std2)
    a = 1.0 / (2 * std1 ** 2) - 1.0 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)

    # Return intersection between means
    mean_min, mean_max = sorted([m1, m2])

    # Only return the intersection if one exists between the means
    roots = [round(x, 2) for x in np.roots([a, b, c])]
    inter = [x for x in np.roots([a, b, c]) if mean_min < x < mean_max]
    if len(inter) == 0:
        return roots
    else:
        return inter


def mutually_exclusive_set_counts(
        sets: Dict[str, Set[str]],
        groups: List[str] = None):
    """
    Given a dictionary of sets, computes the mutually exclusive set intersection counts

    Args:
        sets: Dictionary of sets
        groups: Optional set of keys to subset from set dictionary

    Returns:
        Memberships and their associated counts
    """
    # Collect master set
    groups = sets.keys() if groups is None else groups
    master = set()
    for g in groups:
        for s in sets[g]:
            master.add(s)

    combs = []
    counts = []
    # For decreasing set sizes, identify if outlier belongs to group
    for i in range(len(groups), 0, -1):
        for combination in combinations(groups, i):
            count = 0
            combs.append(combination)
            inter = set.intersection(*[sets[x] for x in combination])

            # Iterate over master and pop items if they belong to group
            for item in list(master):
                if item in inter:
                    count += 1
                    master.remove(item)

            counts.append(count)
    return combs, counts
