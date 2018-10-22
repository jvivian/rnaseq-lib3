from multiprocessing import cpu_count
from typing import List, Dict, Tuple

from sklearn.feature_selection import SelectKBest
import pandas as pd
import pymc3 as pm
import scipy.stats as st


def train_outlier_model(sample: pd.Series,
                        background_df: pd.DataFrame,
                        class_col: str,
                        genes: List[str],
                        n_samples: int = 200,
                        n_chains: int = None,
                        tune: int = 1000,
                        target_accept: float = 0.90,
                        n_genes: int = 50) -> Tuple[pm.model.Model, pm.backends.base.MultiTrace]:
    """
    Trains Bayesian outlier model of n-of-1 sample against background datasets

    Args:
        sample: N-of-1 expression vector
        genes: Gene pool to selectKBest from
        background_df: Background dataset of expression where index = Samples
        class_col: Column to use as class discriminator
        n_samples: Number of samples to train
        n_chains: Number of chains to run. Chains x n_samples = total samples. Defaults to num_cores
        tune: `tune` parameter for `pm_sample`
        target_accept: `target_accept` parameter for NUTS
        n_genes: Number of genes to select via SelectKBest if `genes` is None

    Returns:
        (Model, Trace) from PyMC3
    """
    # There is both a n_chains and njobs command, but this simplifies things a bit since almost always chains >= 4
    n_chains = cpu_count() if n_chains is None else n_chains
    classes = sorted(background_df[class_col].unique())
    print(f'Running {n_chains} on as many cores (if >= 4)')
    print(f'Number of parameters in model: {num_params(len(genes), len(classes))}')

    # Pick genes if not passed in
    if genes is None:
        print(f'Genes not selected, picking {n_genes} via SelectKBest')
        k = SelectKBest(k=n_genes)
        k.fit_transform(background_df[genes], background_df[class_col])
        kbest = [genes[i] for i in k.get_support(indices=True)]

    # Fits
    fits = fit_genes_gaussian(df=background_df, class_col=class_col, genes=kbest)

    # Define and run model
    with pm.Model() as model:
        # Priors for linear model
        alpha = pm.Normal('alpha', 0, 5)
        beta = pm.Normal('beta', 0, 5, shape=len(classes))

        # Convert fits into Normal RVs
        exp_rvs = {key: pm.Normal(key, *fits[key]) for key in fits}

        # Define linear model for each gene
        mu = {}
        for i, gene in enumerate(kbest):
            mu[gene] = alpha
            for j, name in enumerate(classes):
                mu[gene] += exp_rvs[f'{gene}-{name}'] * beta[j]

        # Single sigma across all genes
        sigma = pm.InverseGamma('sigma', 1)

        # Define z distributions for each mu
        z = {}
        for i, gene in enumerate(genes):
            obs = sample[gene]
            z[gene] = pm.Laplace(gene, mu=mu[gene], b=sigma, observed=obs)

        # Calculate trace
        trace = pm.sample(n_samples,
                          tune=tune,
                          nuts_kwargs={'target_accept': target_accept},
                          njobs=n_chains)
    return model, trace


def fit_genes_gaussian(df: pd.DataFrame, class_col: str, genes: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Fits Gaussian distribution to genes

    Args:
        df: DataFrame of gene expression
        class_col: Column in dataframe to use for class separation
        genes: Genes to train on

    Returns:
        Dictionary of Gaussian fits for genes and given classes
    """
    fits = {}
    classes = sorted(df[class_col].unique())
    for c in classes:
        sub = df[df[class_col] == c]
        for gene in genes:
            key = f'{gene}-{c}'
            fits[key] = st.norm.fit(sub[gene])
    return fits


def num_params(n_genes, n_datasets):
    """Calculates number of parameters in model"""
    return (n_genes * n_datasets) + (n_datasets + 1) + n_genes + 1 + n_genes
