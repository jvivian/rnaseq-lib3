from typing import Union

import numpy as np
from pandas import DataFrame


# Differential Expression
def log2fc(a: Union[float, np.array], b: Union[float, np.array], pad: float = 0.001) -> Union[float, np.array]:
    """
    Calculate the log2 Fold Change between two arrays, floats, or integers
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
