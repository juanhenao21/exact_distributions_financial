'''Local normalization plot module.

The functions in the module plot the data obtained in the
epochs_analysis module.

This script requires the following modules:
    * gc
    * typing
    * matplotlib
    * numpy
    * pandas
    * seaborn
    * epochs_sim_tools

The module contains the following functions:
    * epochs_sim_agg_dist_returns_market_plot - plots the aggregated
      distribution of simulated returns for a market.
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# -----------------------------------------------------------------------------
# Modules

import gc
from typing import List

from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

import epochs_sim_tools

# -----------------------------------------------------------------------------


def epochs_sim_agg_returns_market_plot(agg_ret: pd.Series,
                                            epochs_len: int) -> None:
    """Plots the aggregated distribution of simulated returns for a market.

    :param agg_ret: simulated rotated and aggregated returns from a simulated
     market.
    :type agg_ret: pd.Series
    :param epochs_len: length of the epochs.
    :type win: int
    """

    function_name: str = epochs_sim_agg_returns_market_plot.__name__
    # epochs_sim_tools \
    #     .function_header_print_plot(function_name, dates, time_step, window, K_value)

    agg_ret = agg_ret.rename('Agg. returns')

    x_values: np.ndarray = np.arange(-6, 6, 0.001)
    gaussian: np.ndarray = epochs_sim_tools \
        .gaussian_distribution(0, 1, x_values)

    plot_lin = agg_ret.plot(kind='density', style='-', logy=True,
                            figsize=(16, 9), legend=True, lw=5)

    plt.semilogy(x_values, gaussian, '-', lw=3, label='Gaussian')

    plt.legend(fontsize=20)
    plt.title(f'Simulation', fontsize=30)
    plt.xlabel(f'Aggregated simulated returns - epochs {epochs_len}',
               fontsize=25)
    plt.ylabel('PDF', fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(-6, 6)
    plt.ylim(10 ** -5, 1)
    plt.grid(True)
    plt.tight_layout()
    figure_log = plot_lin.get_figure()

    # Plotting
    figure_log.savefig(f'../plot/epochs_sim/epoch_epoch_{epochs_len}.png')
    # epochs_sim_tools \
    #     .save_plot(figure_log, function_name + '_log', dates, time_step,
    #                 window, K_value)

    plt.close()
    del agg_ret
    del figure_log
    del plot_lin
    gc.collect()

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
