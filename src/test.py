import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import rel_entr
from scipy.stats import ks_2samp

def chi_square_test(real_distribution, sim_distribution):
    """Perform Chi-square test of independence between real and simulated responses."""
    
    categories = sorted(set(real_distribution.index).union(set(sim_distribution.index)))
    real_freqs = np.array([real_distribution.get(cat, 0) for cat in categories])
    sim_freqs = np.array([sim_distribution.get(cat, 0) for cat in categories])

    if np.sum(real_freqs) == 0 or np.sum(sim_freqs) == 0:
        return np.nan, np.nan

    chi2_stat, p_value, _, _ = stats.chi2_contingency([real_freqs, sim_freqs])

    print(f"Chi-square Statistic: {chi2_stat:.4f}, P-value: {p_value:.4f}")
    return chi2_stat, p_value


def ks_test(real_distribution, sim_distribution):
    """Perform Kolmogorov-Smirnov test for distribution shape comparison."""
    
    categories = sorted(set(real_distribution.index).union(set(sim_distribution.index)))
    real_cdf = np.cumsum([real_distribution.get(cat, 0) for cat in categories])
    sim_cdf = np.cumsum([sim_distribution.get(cat, 0) for cat in categories])

    ks_stat = np.max(np.abs(real_cdf - sim_cdf))
    p_value = 1 - ks_stat  # Approximate p-value

    print(f"KS Statistic: {ks_stat:.4f}, P-value: {p_value:.4f}")
    return ks_stat, p_value


def shannon_diversity(distribution):
    """Calculate Shannon's Diversity Index."""
    proportions = np.array(distribution)
    proportions = proportions[proportions > 0]  # Avoid log(0)
    return -np.sum(proportions * np.log2(proportions + 1e-10))


def simpson_index(distribution):
    """Calculate Simpson's Diversity Index."""
    proportions = np.array(distribution)
    return 1 - np.sum(proportions ** 2)


def kl_divergence(real_distribution, sim_distribution):
    """Calculate Kullback-Leibler divergence between real and simulated distributions."""
    
    categories = sorted(set(real_distribution.index).union(set(sim_distribution.index)))
    real_probs = np.array([real_distribution.get(cat, 0) for cat in categories])
    sim_probs = np.array([sim_distribution.get(cat, 0) for cat in categories])

    real_probs /= real_probs.sum()
    sim_probs /= sim_probs.sum()

    real_probs = np.clip(real_probs, 1e-10, 1)
    sim_probs = np.clip(sim_probs, 1e-10, 1)

    return np.sum(rel_entr(real_probs, sim_probs))