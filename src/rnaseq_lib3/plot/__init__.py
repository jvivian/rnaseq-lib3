from typing import List

import holoviews as hv
import numpy as np
import seaborn as sns
from holoviews import Bars
from pandas import DataFrame

from rnaseq_lib3.exp import sample_counts_df


def regression(df: DataFrame,
               x_val: str,
               y_val: str,
               vdims: List[str] = None,
               ci: int = 95,
               title: str = None) -> hv.NdOverlay:
    """
    Seaborn's regplot implemented in Holoviews

    Args:
        df: Input dataframe
        x_val: Column name for x-variable
        y_val: Column name for y-variable
        vdims: Additional vdims to include that will display using the hover tool (optional)
        ci: Confidence interval to calculate for regression [0, 100]
        title: Title for plot (optional)

    Returns:
        Combined plot of scatter points, regression line, and CI interval
    """
    # Define grid for x-axis
    xmin, xmax = df[x_val].min(), df[x_val].max()
    grid = np.linspace(xmin, xmax)
    # Seaborn Fit and CI via Bootstrapping
    yhat, yhat_boots = _fit_fast(grid, df[x_val], df[y_val])
    ci = sns.utils.ci(yhat_boots, which=ci, axis=0)
    # Define plot Elements
    vdims = [y_val] + vdims if vdims else [y_val]
    scatter = hv.Scatter(data=df, kdims=['P-value'], vdims=vdims).options(tools=['hover'], width=700, height=300,
                                                                          color='blue', size=5, alpha=0.50)
    regline = hv.Curve((grid, yhat)).options(color='red')
    lower = hv.Area((grid, yhat, ci[0]), vdims=['y', 'y2']).options(color='red', alpha=0.15)
    upper = hv.Area((grid, yhat, ci[1]), vdims=['y', 'y2']).options(color='red', alpha=0.15)

    return (scatter * regline * lower * upper).relabel(title)


def _fit_fast(grid: np.array, x: str, y: str):
    """Low-level regression and prediction using linear algebra. Copied from Seaborn's Regression.py"""

    def reg_func(_x, _y):
        return np.linalg.pinv(_x).dot(_y)

    X, y = np.c_[np.ones(len(x)), x], y
    grid = np.c_[np.ones(len(grid)), grid]
    yhat = grid.dot(reg_func(X, y))

    beta_boots = sns.algorithms.bootstrap(X, y, func=reg_func).T
    yhat_boots = grid.dot(beta_boots).T
    return yhat, yhat_boots


def sample_counts(df: DataFrame, groupby: str = 'tissue', title: str = None) -> Bars:
    """Bar graph of tissues or subtypes grouped by dataset"""

    # Convert dataframe
    counts = sample_counts_df(df, groupby=groupby)

    # Define dimensions
    tissue_dim = hv.Dimension(groupby, label=groupby.capitalize())
    label_dim = hv.Dimension('label', label='Label')
    count_dim = hv.Dimension('counts', label='Count')

    # Opts
    sample_count_opts = {'Bars': {'plot': dict(width=875, height=400, xrotation=70, tools=['hover'],
                                               show_legend=False, toolbar='above'),
                                  'style': dict(alpha=0.25, hover_alpha=0.75)}}

    # Return Bars object of sample counts
    return hv.Bars(counts, kdims=[tissue_dim, label_dim], vdims=[count_dim],
                   label=title).opts(sample_count_opts)
