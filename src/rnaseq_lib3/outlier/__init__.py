import pickle
from collections import defaultdict
from typing import List, Tuple, Dict

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import trimap
from pymc3.backends.base import MultiTrace
from pymc3.model import Model
from sklearn.feature_selection import SelectKBest
from sklearn.manifold import t_sne
from tqdm.autonotebook import tqdm


def run_model(sample: pd.Series,
              background_df: pd.DataFrame,
              class_col: str = 'tissue',
              training_genes: List[str] = None,
              gene_pool: List[str] = None,
              n_genes: int = 50,
              beta_func=pm.Normal,
              draws: int = 500,
              tune: int = 1000,
              n_chains: int = 4) -> Tuple[Model, MultiTrace]:
    """
    Run Bayesian outlier model

    Args:
        sample: N-of-1 sample to run   
        background_df: Background dataframe to use in comparison
        class_col: Column in background dataframe to use as categorical discriminator
        training_genes: Genes to use during training
        gene_pool: Set of genes
        n_genes: Number of genes to use in training if not supplied via training_genes
        beta_func: PyMC3 Distribution to use for beta function. Must accept mu and sd.
        draws: Number of draws during sampling
        tune: Sampling parameter
        n_chains: Sampling parameter

    Returns:
        Model and Trace from PyMC3
    """
    assert any([gene_pool, training_genes]), 'gene_pool or training_genes must be supplied'

    # Create categorical index
    idx = background_df[class_col].astype('category').cat.codes
    n_cats = len(background_df[class_col].unique())

    # Identify gene set to train on
    if not training_genes:
        training_genes = select_k_best_genes(background_df, genes=gene_pool, class_col=class_col, n=n_genes)

    # Define model and sample
    with pm.Model() as model:
        # Alpha in linear model
        a = pm.Normal('a', mu=0, sd=10)

        # If number of categories is 1, we don't need hyperpriors for b
        if n_cats == 1:
            b = beta_func('b', mu=0, sd=10, shape=1)
        else:
            mu_b = pm.Normal('mu_b', mu=0, sd=10)
            sigma_b = pm.InverseGamma('sigma_b', 2.1, 1)
            b = beta_func('b', mu=mu_b, sd=sigma_b, shape=n_cats)

        # Linear model
        mu = {}
        for gene in training_genes:
            mu[gene] = a + b[idx] * background_df[gene]

        # Model estimation
        eps = pm.InverseGamma('eps', 2.1, 1)
        z = {}
        for gene in training_genes:
            z[gene] = pm.Laplace(gene, mu=mu[gene], b=eps, observed=sample[gene])

        trace = pm.sample(draws=draws, tune=tune, n_chains=n_chains)
    return model, trace


def ppc_from_coefs(trace: MultiTrace,
                   genes: List[str],
                   background_df: pd.DataFrame,
                   class_col: str,
                   num_samples: int = None):
    """
    Draws posterior using the linear model coefficients

    Args:
        trace: Trace from PyMC3
        genes: Gene of interest
        background_df: Background dataset of expression where index = Samples
        class_col: Column to use as class discriminator
        num_samples: Number of sampling iterations, defaults to get a total of ~1 mil samples in posterior
    """
    num_samples = 1_000_000 // len(background_df) if num_samples is None else num_samples

    # Categorical code mapping
    codes = {cat: i for i, cat in enumerate(background_df[class_col].unique())}
    code_vec = [codes[x] for x in background_df[class_col]]

    # Calculate posterior from linear model
    df_len = len(background_df)
    zs = {gene: np.zeros(df_len * num_samples) for gene in genes}
    sub = background_df[genes]
    for i in tqdm(range(num_samples), total=num_samples):
        z = trace['a'][i] + sub.mul([trace['b'][i, x] for x in code_vec], axis=0)
        for j, gene in enumerate(z.columns):
            zs[gene][df_len * i: df_len * (i+1)] = np.random.laplace(loc=z[gene], scale=trace['eps'].mean())
    return zs


def posterior_predictive_pvals(sample: pd.Series, ppc: Dict[str, np.array]):
    pvals = {}
    for gene in tqdm(ppc):
        z_true = sample[gene]
        z = ppc[gene]
        pvals[gene] = _ppp_one_gene(z_true, z)
    return pvals


def _ppp_one_gene(z_true, z):
    return round(np.sum(z_true < z) / len(z), 5)


def plot_gene_ppc(sample: pd.Series, ppc: Dict[str, np.array], gene, ax=None):
    z = ppc[gene]
    pval = _ppp_one_gene(sample[gene], z)
    # Plot
    if ax:
        ax.axvline(sample[gene], color='red', label='z-true')
        ax.set_title(f'{gene} - P: {pval}')
        sns.kdeplot(z, label='Linear-Equation', ax=ax)
    else:
        plt.axvline(sample[gene], color='red', label='z-true')
        plt.title(f'{gene} - P: {pval}')
        sns.kdeplot(z, label='Linear-Equation')


def select_k_best_genes(df: pd.DataFrame, genes: List[str], class_col, n=50):
    k = SelectKBest(k=n)
    k.fit_transform(df[genes], df[class_col])
    return [genes[i] for i in k.get_support(indices=True)]


def _pickle(model_name, model, trace):
    with open(model_name, 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace}, buff)


def _load_pickle(pkl_path):
    with open(pkl_path, 'rb') as buff:
        data = pickle.load(buff)
    return data['model'], data['trace']


def calculate_weights(classes, trace):
    weight_by_class = pd.DataFrame({'Class': classes,
                                    'Weights': [np.median(trace['b'][:, x]) for x in range(len(classes))]})
    return weight_by_class.sort_values('Weights', ascending=False)


def plot_weights(classes, trace, output: str = None):
    """Plot model coefficients associated with each class"""
    # Construct weight by class DataFrame
    weights = calculate_weights(classes, trace)

    plt.figure(figsize=(12, 4))
    sns.barplot(data=weights, x='Class', y='Weights')
    plt.xticks(rotation=90)
    plt.title('Median Beta Coefficient Weight by Tissue for N-of-1 Sample')
    if output:
        plt.savefig(output, bbox_inches='tight')


def dimensionality_reduction(sample: pd.Series,
                             background_df: pd.DataFrame,
                             genes: List[str],
                             col: str,
                             method='trimap') -> Tuple[pd.DataFrame, hv.Scatter]:
    """
    Wrapper for returning trimap plot with column for `color_index` and `size_index`

    Args:
        sample: n-of-1 sample. Gets own label
        background_df: Background dataset
        genes: Genes to use in dimensionality reduction
        col: Column to use for color_index
        method: Method of dimensionality reduction. `trimap` or `tsne`

    Returns:
        Holoviews Scatter object of plot with associated vdims
    """
    assert method == 'trimap' or method == 'tsne', '`method` must be either `trimap` or `tsne`'
    combined = background_df.append(sample)
    if method == 'trimap':
        reduced = trimap.TRIMAP().fit_transform(combined[genes])
    else:
        reduced = t_sne.TSNE().fit_transform(combined[genes])
    df = pd.DataFrame(reduced, columns=['x', 'y'])
    df[col] = background_df[col].tolist() + [f'N-of-1 - {sample[col]}']
    df['size'] = [1 for _ in background_df[col]] + [5]
    return df, hv.Scatter(data=df, kdims=['x'], vdims=['y', col, 'size'])
