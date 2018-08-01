import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


def collect_outliers(sample_dir: str, samples: List[str], inter_path=None, col='pc_up') -> Dict[str: List[str]]:
    """
    Collect outliers from Treehouse tertiary analysis

    Args:
        sample_dir: Directory containing patient samples
        samples: List of samples to collect
        inter_path: Intermediate path if one exists: like tertiary/treehouse-8.0.1_comp-v5
        col: Column value to use when collecting data outlier

    Returns: Dictionary of sample -> list[genes]
    """
    outliers = {}
    no_results = []
    for sample in sorted(samples):

        # Determine path to outlier_results file
        if inter_path:
            path = Path(sample_dir) / sample / inter_path / f'outlier_results_{sample}'
        else:
            path = Path(sample_dir) / sample / f'outlier_results_{sample}'

        # Check if path exists
        if not os.path.exists(str(path)):
            no_results.append(sample)
            continue

        # Read in outlier file and save up-outliers
        df = pd.read_csv(path, sep='\t')
        outliers[sample] = df[df.pc_outlier == col]

    # List results not found
    if no_results:
        print('No results found for: \n{}'.format('\n'.join(no_results)))

    return outliers
