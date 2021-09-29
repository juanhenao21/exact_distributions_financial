'''Aggegated returns epochs implementation.

Plots the figures of the aggregated returns epochs for the paper.

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
    * epochs_aggregated_dist_returns_market_plot - plots the aggregated
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

sys.path.insert(1, '../../project/epochs')
import epochs_tools  # type: ignore

# ----------------------------------------------------------------------------


def epochs_aggregated_dist_returns_market_plot(dates: List[str],
                                               time_step: str,
                                               window: str,
                                               K_value: str) -> None:
    """Plots the aggregated distribution of returns for a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. [['1992-01', '2012-12'], ['2012-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    try:

        figure = plt.figure(figsize=(16, 9))

        # Load data
        agg: pd.Series = pickle.load(open(
            '../../project/data/epochs/epochs_aggregated_dist_returns'
            + f'_market_data_{dates[0]}_{dates[1]}_step_{time_step}_win'
            + f'_{window}_K_{K_value}.pickle', 'rb'))

        agg = agg.rename('Agg. returns')

        x_gauss: np.ndarray = np.arange(-10, 10, 0.3)
        gaussian: np.ndarray = epochs_tools \
            .gaussian_distribution(0, 1, x_gauss)

        # Log plot
        plot = agg.plot(kind='density', style='o', logy=True, legend=False,
                        ms=7)

        plt.semilogy(x_gauss, gaussian, '-', lw=5, label='Gaussian')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2,
                   fontsize=20)
        plt.xlabel(r'$\tilde{r}$', fontsize=30)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim(-6, 6)
        plt.ylim(10 ** -5, 1)
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        figure.savefig(f'../plot/03_agg_returns_epoch.png')

        plt.close()
        del agg
        del figure
        del plot
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

    epochs_aggregated_dist_returns_market_plot(['1990-01-01', '2020-12-31'],
                                              '1d', '25', '50')

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
