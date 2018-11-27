import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
from pymc3.backends.base import MultiTrace
from pymc3.model import Model
from sklearn.feature_selection import SelectKBest


def run_model(sample: pd.Series,
              background_df: pd.DataFrame,
              class_col: str,
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


def posterior_from_linear(trace: MultiTrace,
                          sample: pd.Series,
                          gene: str,
                          background_df: pd.DataFrame,
                          class_col: str,
                          ax=None):
    """
    Draws posterior using the linear model coefficients

    Args:
        trace: Trace from PyMC3
        sample: N-of-1 sample
        gene: Gene of interest
        background_df: Background dataset of expression where index = Samples
        class_col: Column to use as class discriminator
        ax: Optional ax input if using within subplots
    """
    group = sorted(background_df[class_col].unique())

    # Calculate posterior from linear model
    z = trace['a']
    for i, t in enumerate(group):
        samples = np.random.choice(background_df[background_df[class_col] == t][gene], len(z))
        z += trace['b'][:, i] * samples

    # Calculate PPP
    z_true = sample[gene]
    ppp = round(sum(z_true < z) / len(z), 2)

    # Plot
    if ax:
        ax.axvline(sample[gene], color='red', label='z-true')
        ax.set_title(f'{gene} - P: {ppp}')
        sns.kdeplot(z, label='Linear-Equation', ax=ax)
    else:
        plt.axvline(sample[gene], color='red', label='z-true')
        plt.title(f'{gene} - P: {ppp}')
        sns.kdeplot(z, label='Linear-Equation')


def posterior_pvalues(sample: pd.Series, trace: MultiTrace, model: Model, genes: List[str]) -> pd.DataFrame:
    """
    Calculates posterior pvalues from `pm.sample_ppc` for training genes against an n-of-1 sample

    Args:
        sample: n-of-1 sample to compare
        trace: PyMC3 trace
        model: PyMC3 model
        genes: List of genes that were used in training

    Returns:
        DataFrame containing genes and pvalues
    """
    ppc = pm.sample_ppc(trace, model=model)
    ppp = {}
    # For each gene, calculate posterior estimate and calculate PPP
    for gene in genes:
        z = ppc[gene].ravel()
        z_true = sample[gene]
        ppp[gene] = sum(z_true < z) / len(z)

    return pd.DataFrame({'gene': genes, 'pval': [ppp[g] for g in genes]}).sort_values('pval')


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


def plot_weights(classes, trace, output: str = None):
    """Plot model coefficients associated with each class"""
    # Construct weight by class DataFrame
    weight_by_class = pd.DataFrame({'Class': classes,
                                    'Weights': [np.median(trace['b'][:, x]) for x in range(len(classes))]})
    weight_by_class = weight_by_class.sort_values('Weights', ascending=False)

    plt.figure(figsize=(12, 4))
    sns.barplot(data=weight_by_class, x='Class', y='Weights')
    plt.xticks(rotation=90)
    plt.title('Median Beta Coefficient Weight by Tissue for N-of-1 Sample')
    if output:
        plt.savefig(output, bbox_inches='tight')
