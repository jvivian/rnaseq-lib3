import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame


# These functions are for data stored in this Synapse ID: syn12120302
# Expression: pd.read_hdf(data_path, key='exp')
# Metadata: pd.read_hdf(data_path, key='met')
def add_metadata_to_exp(exp: DataFrame, met: DataFrame) -> DataFrame:
    """Adds metadata to the expression dataframe and returns a combined object"""
    # Remove duplicates from metadata
    samples = [x for x in exp.index if x in met.id]
    met = met[met.id.isin(samples)].drop_duplicates('id')

    # Ensure index dims are the same length
    assert len(exp) == len(met), 'Expression dataframe and metadata do not match index lengths'

    # Add metadata and return resorted dataframe
    df = exp.copy()
    df.insert(0, 'label', _label_vector_from_samples(df.index))
    df.insert(0, 'tumor', met.tumor)
    df.insert(0, 'subtype', met.type)
    df.insert(0, 'tissue', met.tissue)
    df.insert(0, 'id', met.id)
    return df


def _label_vector_from_samples(samples: List[str]) -> List[str]:
    """Produce a vector of TCGA/GTEx labels for the sample vector provided"""
    vector = []
    for x in samples:
        if x.startswith('TCGA'):
            if x.endswith('11'):
                vector.append('tcga-normal')
            elif x.endswith('01'):
                vector.append('tcga-tumor')
            else:
                vector.append('tcga-other')
        else:
            vector.append('gtex')
    return vector


def sample_counts_df(df: DataFrame, groupby: str = 'tissue') -> DataFrame:
    """Return a dataframe of sample counts based on groupby of 'tissue' or 'type'"""
    # Cast value_counts as DataFrame
    vc = pd.DataFrame(df.groupby(groupby).label.value_counts())
    # Relabel column and reset_index to cast multi-index as columns
    vc.columns = ['counts']
    vc.reset_index(inplace=True)
    return vc.sort_values([groupby, 'label'])


def subset_by_dataset(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Subset expression/metadata table by Label"""
    tumor = df[df.label == 'tcga-tumor']
    normal = df[df.label == 'tcga-normal']
    gtex = df[df.label == 'gtex']
    return tumor, normal, gtex


def low_variance_filtering(df: DataFrame, unexpressed: float = 0.8,
                           thresh: int = 0, filter_perc: float = 0.2) -> DataFrame:
    """
    Filter genes not expressed in 80% of samples and then 20% of the lowest varying remaining genes
    Source: https://github.com/UCSC-Treehouse/protocol/blob/master/3_generate-thresholds.ipynb

    Args:
        df: DataFrame of expression values
        unexpressed: Unexpressed ratio required to remove gene
        thresh: Threshold which constitutes zero. May need to adjust to 0.0001443 if genes not removed
        filter_perc: Percentage of low-variance genes to remove

    Returns:
        DataFrame of filtered genes
    """
    # TODO: Eventually fix code to assume samples by gene matrix
    # Transpose matrix because source code assumes genes by samples
    df = df.T

    # Calculate maximum allowed zeroes
    max_allowed_zeroes = len(df.columns) * unexpressed

    # Is the count of items less than threshold within the acceptable count?
    def sufficiently_expressed(series, max_zeroes, threshold):
        return len(series[series <= threshold]) < max_zeroes

    # Gene & whether it is Keep (True) or too many zeroes (False)
    with_zeroes = df.apply(sufficiently_expressed, args=(max_allowed_zeroes, thresh), axis=1)

    # Next, do variance filtering
    expression_filtered_compendium = df[with_zeroes]
    print(str(len(expression_filtered_compendium)) + ' genes remain after expression filter.')

    # Get the standard deviation
    variance = expression_filtered_compendium.apply(np.std, axis=1)
    cut_proportion = int(math.ceil(len(variance) * filter_perc))
    keep_proportion = len(variance) - cut_proportion
    expression_and_variance_filtered = variance.nlargest(keep_proportion)
    print(str(len(expression_and_variance_filtered)) + ' genes remain after variance filter.')

    return df.T[expression_and_variance_filtered.index.values]
