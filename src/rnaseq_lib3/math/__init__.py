from typing import Union, List, Tuple

import numpy as np
from pandas import DataFrame


# Outlier
def iqr_bounds(ys: List[Union[float, int]]) -> Tuple[float, float]:
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
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return upper_bound, lower_bound


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
