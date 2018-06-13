import holoviews as hv
from holoviews import Bars
from pandas import DataFrame

from rnaseq_lib3.exp import sample_counts_df


def sample_counts(df: DataFrame, groupby: str = 'tissue') -> Bars:
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
                   label='Sample Counts for TCGA and GTEx').opts(sample_count_opts)
