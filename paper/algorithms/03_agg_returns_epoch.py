'''Aggegated returns epochs implementation.

Plots the figures of the aggregated returns epochs for the paper.

This script requires the following modules:
    * gc
    * pickle
    * typing
    * matplotlib
    * numpy
    * pandas
    * exact_distributions_covariance_tools

The module contains the following functions:
    * ln_aggregated_dist_returns_market_plot - plots the aggregated
      distribution of returns for a market.
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

sys.path.insert(1, '../../project/local_normalization')
import local_normalization_tools  # type: ignore

# ----------------------------------------------------------------------------


def ln_aggregated_dist_returns_market_plot(dates: List[List[str]],
                                           time_step: str,
                                           window: str) -> None:
    """Plots the aggregated distribution of returns for a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. [['1992-01', '2012-12'], ['2012-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    try:

        figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
        ax1.get_shared_y_axes().join(ax2)

        # Load data
        agg_1: pd.Series = pickle.load(open(
            '../../project/data/local_normalization/ln_aggregated_dist_returns'
            + f'_market_data_{dates[0][0]}_{dates[0][1]}_step_{time_step}_win'
            + f'_{window}.pickle', 'rb'))
        agg_2: pd.Series = pickle.load(open(
            '../../project/data/local_normalization/ln_aggregated_dist_returns'
            + f'_market_data_{dates[1][0]}_{dates[1][1]}_step_{time_step}_win'
            + f'_{window}.pickle', 'rb'))

        agg_1 = agg_1.rename('Agg. returns')
        agg_2 = agg_2.rename('Agg. returns')

        x_gauss: np.ndarray = np.arange(-10, 10, 0.3)
        gaussian: np.ndarray = local_normalization_tools \
            .gaussian_distribution(0, 1, x_gauss)

        # Log plot
        plot_1 = agg_1.plot(kind='density', style='-', logy=True, ax=ax1,
                            legend=False, lw=5)
        plot_2 = agg_2.plot(kind='density', style='-', logy=True, ax=ax2,
                            legend=False, lw=5)

        ax1.semilogy(x_gauss, gaussian, 'o', ms=10, label='Gaussian')
        ax2.semilogy(x_gauss, gaussian, 'o', ms=10, label='Gaussian')

        ax1.set_xlabel(r'$\tilde{r}$', fontsize=30)
        ax1.set_ylabel('PDF', fontsize=25)
        ax1.tick_params(axis='both', labelsize=20)
        ax1.set_xlim(-6, 6)
        ax1.set_ylim(10 ** -5, 1)
        ax1.grid(True)

        # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2,
        #        fontsize=20)
        ax2.set_xlabel(r'$\tilde{r}$', fontsize=30)
        ax2.set_ylabel('PDF', fontsize=25)
        ax2.tick_params(axis='both', labelsize=20)
        ax2.set_xlim(-6, 6)
        ax2.set_ylim(10 ** -5, 1)
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        figure.savefig(f'../plot/03_agg_returns_epoch.png')

        plt.close()
        del agg_1
        del agg_2
        del figure
        del plot_1
        del plot_2
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# ----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

    ln_aggregated_dist_returns_market_plot([['1992-01', '2012-12'],
                                            ['2012-01', '2020-12']],
                                           '1d', '25')

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
