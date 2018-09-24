import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from sklearn import mixture

from rnaseq_lib3.math import find_gaussian_intersection
from rnaseq_lib3.math.dist import best_fit_distribution, make_pdf


def overlay_gmm_to_hist(source_dist, figsize=(12, 4), color='red', label='Tumor'):
    """
    Given a source distribution, fit a 2-component Gaussian mixture model and return plot
    :param np.array source_dist: Source distribution
    :param tuple(int, int) figsize: Figure size
    :return: Fig object and cutoff value
    :rtype: float
    """
    # Fit GMM
    gmm = mixture.GaussianMixture(n_components=2).fit(pd.DataFrame(source_dist))
    m1, m2 = gmm.means_
    std1, std2 = gmm.covariances_

    # Identify intersection between the two Gaussians
    cutoffs = find_gaussian_intersection(m1, m2, std1, std2)

    # Plot source data
    plt.subplots(figsize=figsize)
    sns.kdeplot(source_dist, shade=True, label=label)
    # plt.hist(source_dist, density=True, alpha=0.25, bins=50, label=label, color=color)

    # Plot Gaussian fits and intersection
    x = np.linspace(min(source_dist), max(source_dist), len(source_dist))
    plt.plot(x, *st.norm.pdf(x, m1, std1), label='u={}, o={}'.format(round(m1, 1), round(std1, 1)))
    plt.plot(x, *st.norm.pdf(x, m2, std2), label='u={}, o={}'.format(round(m2, 1), round(std2, 1)))
    # Add intersection lines
    for cutoff in cutoffs:
        plt.vlines(cutoff, *plt.ylim(), label='Cutoff: {}'.format(cutoff), color='red', linestyles='--')
    plt.legend()
    if len(cutoffs) == 1:
        return cutoffs[0]
    else:
        return max(cutoffs)


def plot_fits(data, title, xlabel='Log2(TPM + 1)'):
    # Plot for comparison
    plt.figure(figsize=(12, 8))
    ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5)
    # Save plot limits
    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_fit_name, best_fit_params, sse = best_fit_distribution(data, 200, ax)
    best_dist = getattr(st, best_fit_name)

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(f'{title}\n All Fitted Distributions')
    ax.set_xlabel(u'Log2(TPM + 1)')
    ax.set_ylabel('Frequency')

    # Make PDF with best params
    pdf = make_pdf(best_dist, best_fit_params)

    # Display
    plt.figure(figsize=(12, 8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)

    ax.set_title(f'{title} w/ best fit\n' + dist_str)
    ax.set_xlabel(u'Log2(TPM + 1)')
    ax.set_ylabel('Frequency');
