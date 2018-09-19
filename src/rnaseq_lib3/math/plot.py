from scipy.stats import norm
from sklearn import mixture

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rnaseq_lib3.math import find_gaussian_intersection


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
    #plt.hist(source_dist, density=True, alpha=0.25, bins=50, label=label, color=color)

    # Plot Gaussian fits and intersection
    x = np.linspace(min(source_dist), max(source_dist), len(source_dist))
    plt.plot(x, *norm.pdf(x, m1, std1), label='u={}, o={}'.format(round(m1, 1), round(std1, 1)))
    plt.plot(x, *norm.pdf(x, m2, std2), label='u={}, o={}'.format(round(m2, 1), round(std2, 1)))
    # Add intersection lines
    for cutoff in cutoffs:
        plt.vlines(cutoff, *plt.ylim(), label='Cutoff: {}'.format(cutoff), color='red', linestyles='--')
    plt.legend()
    if len(cutoffs) == 1:
        return cutoffs[0]
    else:
        return max(cutoffs)
