'''Aggegated returns epochs simulation implementation.

Plots the figures of the aggregated returns epochs simulations for the paper.

This script requires the following modules:
    * gc
    * pickle
    * sys
    * typing
    * matplotlib
    * numpy
    * pandas
    * epochs_tools

The module contains the following functions:
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# ----------------------------------------------------------------------------
# Modules

import gc
import pickle
import sys
from typing import List

from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(1, '../../project/epochs_sim')
import epochs_sim_tools  # type: ignore
import epochs_sim_analysis  # type: ignore

# ----------------------------------------------------------------------------

def epochs_sim_agg_ret_pairs(normalized: bool = False) -> None:
    """Plot the simulated aggregated rotated and scaled returns without
       normalization.

    :param normalized: normalize the returns within the epochs, defaults to
     False
    :type normalized: bool, optional
    """

    figure = plt.figure(figsize=(16, 9))

    markers: List[str] = ['o', '^', 's', 'P', 'x']

    # Simulate the aggregated returns for different epochs lenghts
    for epochs_len, marker in zip([10, 25, 40, 55, 100], markers):
        agg_ret_pairs = epochs_sim_analysis \
            .epochs_sim_agg_returns_market_data(0.3, 2, 50, 100, epochs_len,
                                                normalized)

        agg_ret_pairs = agg_ret_pairs.rename(f'Epochs T={epochs_len}')

        # Log plot
        plot = agg_ret_pairs.plot(kind='density', style=marker, logy=True,
                                  legend=False, ms=10)

    x_values: np.ndarray = np.arange(-7, 7, 0.03)
    gaussian: np.ndarray = epochs_sim_tools \
        .gaussian_distribution(0, 1, x_values)

    plt.semilogy(x_values, gaussian, '-', lw=10, label='Gaussian')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3,
               fontsize=30)
    plt.xlabel(r'$\tilde{r}$', fontsize=40)
    plt.ylabel('PDF', fontsize=40)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim(-6, 6)
    plt.ylim(10 ** -5, 1)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    if normalized:
        figure.savefig(f'../plot/06_epochs_sim_agg_ret_pairs_norm.png')
    else:
        figure.savefig(f'../plot/06_epochs_sim_agg_ret_pairs_no_norm.png')

    plt.close()
    del agg_ret_pairs
    del figure
    del plot
    gc.collect()

# -----------------------------------------------------------------------------



def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

    dates: List[List[str]] = [['2021-07-19', '2021-08-14'],
                              ['1990-01-01', '2020-12-31'],
                              ['1990-01-01', '2020-12-31'],
                              ['1990-01-01', '2020-12-31']]
    time_steps: List[str] = ['1m', '1d', '1wk', '1mo']

    epochs_sim_agg_ret_pairs(normalized=False)
    epochs_sim_agg_ret_pairs(normalized=True)

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
