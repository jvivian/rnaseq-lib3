import numpy as np
from pandas import DataFrame


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
