import pickle
from typing import List, Tuple, Dict

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as st
import seaborn as sns
import trimap
from sklearn.feature_selection import SelectKBest
from sklearn.manifold import t_sne
from sklearn.metrics import pairwise_distances
from tqdm.autonotebook import tqdm


def run_model(sample: pd.Series,
              df: pd.DataFrame,
              training_genes: List[str],
              class_col: str = 'tissue',
              **kwargs):
    """
    Run Bayesian model by prefitting Y-distributions

    Args:
        sample: N-of-1 sample to run
        df: Background dataframe to use in comparison
        training_genes: Genes to use during training
        class_col:
        **kwargs:

    Returns:
        Model and Trace from PyMC3
    """
    classes = sorted(df[class_col].unique())
    df = df[[class_col] + training_genes]

    # Collect fits
    print('Collecting fits')
    ys = {}
    for gene in tqdm(training_genes):
        for i, dataset in enumerate(classes):
            cat_mu, cat_sd = st.norm.fit(df[df[class_col] == dataset][gene])
            # Standard deviation can't be initialized to 0, so set to 0.1
            cat_sd = 0.1 if cat_sd == 0 else cat_sd
            ys[f'{gene}-{dataset}'] = (cat_mu, cat_sd)

    print('Building model')
    with pm.Model() as model:
        # Linear model priors
        a = pm.Normal('a', mu=0, sd=10)
        b = [1] if len(classes) == 1 else pm.Dirichlet('b', a=np.ones(len(classes)))
        # Model error
        eps = pm.InverseGamma('eps', 2.1, 1)

        # TODO: Try tt.stack to declare mu more intelligently via b * y
        # Linear model declaration
        for gene in tqdm(training_genes):
            mu = a
            for i, dataset in enumerate(classes):
                name = f'{gene}-{dataset}'
                y = pm.Normal(name, *ys[name])
                mu += b[i] * y

            # Embed mu in laplacian distribution
            pm.Laplace(gene, mu=mu, b=eps, observed=sample[gene])
        # Sample
        trace = pm.sample(**kwargs)
    return model, trace


def ppc(trace, genes: List[str]) -> Dict[str, np.array]:
    """
    Posterior predictive check for a list of genes trained in the model

    Args:
        trace: PyMC3 trace
        genes: List of genes of interest

    Returns:

    """
    d = {}
    for gene in tqdm(genes):
        d[gene] = _gene_ppc(trace, gene)
    return d


def _gene_ppc(trace, gene: str) -> np.array:
    """
    Calculate posterior predictive for a gene

    Args:
        trace: PyMC3 Trace
        gene: Gene of interest

    Returns:
        Random variates representing PPC of the gene
    """
    y_gene = [x for x in trace.varnames if x.startswith(f'{gene}-')]
    b = trace['a']
    for i, y_name in enumerate(y_gene):
        b += trace['b'][:, i] * trace[y_name]
    return np.random.laplace(loc=b, scale=trace['eps'])


def posterior_predictive_pvals(sample: pd.Series, ppc: Dict[str, np.array]) -> pd.Series:
    pvals = {}
    for gene in tqdm(ppc):
        z_true = sample[gene]
        z = ppc[gene]
        pvals[gene] = _ppp_one_gene(z_true, z)
    return pd.Series(pvals).sort_values()


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


def select_k_best_genes(df: pd.DataFrame, genes: List[str], class_col='tissue', n=50):
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


def pairwise_distance_ranks(sample: pd.Series, df: pd.DataFrame, genes: List[str], group: str):
    """
    Calculate pairwise distance, rank, and normalize by group counts
    Args:
        sample: n-of-1 sample. Gets own label
        df: background dataset
        genes: genes to use for pairwise distance
        group: Column to use as class discriminator

    Returns:
        DataFrame of pairwise distance ranks
    """
    dist = pairwise_distances(np.array(sample[genes]).reshape(1, -1), df[genes])
    dist = pd.DataFrame([dist.ravel(), df[group]]).T
    dist.columns = ['Distance', 'Group']
    dist = dist.sort_values('Distance')

    # Pandas-FU
    dist = dist.reset_index(drop=True).reset_index()
    return dist.groupby('Group').apply(lambda x: x['index'].sum() / len(x)).sort_values().reset_index(name='Rank-Count')


def sample_by_group_pearsonr(sample: pd.Series, df: pd.DataFrame, genes: List[str], class_col: str) -> pd.DataFrame:
    """
    Return pearsonR of the sample against all groups in the `class_col`

    Args:
        sample: N-of-1 sample
        df: background dataset
        genes: genes to use when calculating PearsonR
        class_col: column to use as class discriminator

    Returns:
        2-column DataFrame of pearsonR scores
    """
    rows = []
    for group in df[class_col].unique():
        sub = df[df[class_col] == group]
        pr, pval = st.pearsonr(sample[genes], sub[genes].median())
        rows.append([group, pr])
    return pd.DataFrame(rows, columns=[class_col, 'PR']).sort_values('PR', ascending=False)
