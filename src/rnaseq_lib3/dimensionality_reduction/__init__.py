import trimap
import pandas as pd
import numpy as np
from typing import List

def run_trimap(df: pd.DataFrame, dim_set: List[str] = None, class_col: str = None, **kwargs) -> pd.DataFrame:
    """
    Run TriMap, but allowing for subsetting of features to train on and adding back in of categorical information

    Args:
        df: Input DataFrame
        dim_set: Set of features to train on
        class_col: Categorical column to add back into dataframe
        **kwargs: TriMap hyperparameters to pass

    Returns:
        DataFrame of TriMap coords and optional categorical information
    """
    # Define feature set to reduce and run TriMap
    array = np.array(df[dim_set]) if dim_set else np.array(df)
    reduced = pd.DataFrame(trimap.TRIMMAP(**kwargs).fit_transform(array), columns=['x', 'y'])
    if class_col:
        reduced[class_col] = df[class_col]
    return reduced
