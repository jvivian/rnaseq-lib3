from typing import List, Tuple

import pandas as pd
from pandas import DataFrame


# These functions are for data stored in this Synapse ID:
# Expression: pd.read_hdf(data_path, key='exp')
# Metadata: pd.read_hdf(data_path, key='met')
def add_metadata_to_exp(exp: DataFrame, met: DataFrame) -> DataFrame:
    """Adds metadata to the expression dataframe and returns a combined object"""
    # Copy genes from expression DataFrame
    genes = exp.columns.tolist()

    # Remove duplicates from metadata
    samples = [x for x in exp.index if x in met.id]
    met = met[met.id.isin(samples)].drop_duplicates('id')

    # Ensure index dims are the same length
    assert len(exp) == len(met), 'Expression dataframe and metadata do not match index lengths'

    # Add metadata and return resorted dataframe
    df = exp
    df.loc[:, 'id'] = met.id
    df.loc[:, 'tissue'] = met.tissue
    df.loc[:, 'type'] = met.type
    df.loc[:, 'tumor'] = met.tumor
    df.loc[:, 'label'] = _label_vector_from_samples(df.index)
    return df[['id', 'tissue', 'type', 'label', 'tumor'] + genes]


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
