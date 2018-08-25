from typing import Union, List, Tuple

import numpy as np
from pandas import DataFrame
from statsmodels.stats.stattools import medcouple


# Outlier
def iqr_bounds(ys: List[Union[float, int]], whis=1.5) -> Tuple[float, float]:
    """
    Return upper and lower bound for an array of values
    Lower bound: Q1 - (IQR * 1.5)
    Upper bound: Q3 + (IQR * 1.5)

    Args:
        ys: Array of values

    Returns:
        Upper and lower bound
    """
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * whis)
    upper_bound = quartile_3 + (iqr * whis)
    return upper_bound, lower_bound


def std_bounds(ys: List[Union[float, int]], num_std: int = 2) -> Tuple[float, float]:
    """Return upper and lower bounds for an array of values"""
    u = np.mean(ys)
    std = np.std(ys)
    upper = u + num_std * std
    lower = u - num_std * std
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
