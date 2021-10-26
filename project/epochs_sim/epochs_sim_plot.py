'''Local normalization plot module.

The functions in the module plot the data obtained in the
epochs_analysis module.

This script requires the following modules:
    * gc
    * pickle
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
import pickle
from typing import List

from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import epochs_sim_tools

# -----------------------------------------------------------------------------


def epochs_sim_agg_returns_market_plot(agg_ret: pd.Series,
                                       epochs_len: int,
                                       K_value: int,
                                       kind: int = 'gaussian') -> None:
    """Plots the aggregated distribution of simulated returns for a market.

    :param agg_ret: simulated rotated and aggregated returns from a simulated
     market.
    :type agg_ret: pd.Series
    :param epochs_len: length of the epochs.
    :type win: int
    """

    function_name: str = epochs_sim_agg_returns_market_plot.__name__
    epochs_sim_tools \
        .function_header_print_plot(function_name, [''], '', '', '', sim=True)

    agg_ret = agg_ret.rename('Agg. returns')

    x_values: np.ndarray = np.arange(-6, 6, 0.001)

    plot_lin = agg_ret.plot(kind='density', style='-', logy=True,
                            figsize=(16, 9), legend=True, lw=5)

    if kind == 'gaussian':
        gaussian: np.ndarray = epochs_sim_tools \
            .gaussian_distribution(0, 1, x_values)
        plt.semilogy(x_values, gaussian, '-', lw=3, label='Gaussian')
    else:
        algebraic: np.ndarray = epochs_sim_tools \
            .algebraic_distribution(K_value, (10 + K_value) / 2, x_values)
        plt.semilogy(x_values, algebraic, '-', lw=3, label='Algebraic')


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
    epochs_sim_tools \
        .save_plot(figure_log, function_name + '_' + kind, [''], '',
                   str(epochs_len), '', sim=True)

    plt.close()
    del agg_ret
    del figure_log
    del plot_lin
    gc.collect()

# -----------------------------------------------------------------------------


def epochs_sim_agg_returns_cov_market_plot(agg_ret: pd.Series,
                                           epochs_len: int) -> None:
    """Plots the aggregated distribution of simulated returns for a market.

    :param agg_ret: simulated rotated and aggregated returns from a simulated
     market.
    :type agg_ret: pd.Series
    :param epochs_len: length of the epochs.
    :type win: int
    """

    function_name: str = epochs_sim_agg_returns_cov_market_plot.__name__
    epochs_sim_tools \
        .function_header_print_plot(function_name, [''], '', '', '', sim=True)

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
    epochs_sim_tools \
        .save_plot(figure_log, function_name, [''], '', str(epochs_len), '',
                   sim=True)

    plt.close()
    del agg_ret
    del figure_log
    del plot_lin
    gc.collect()

# -----------------------------------------------------------------------------


def epochs_aggregated_dist_returns_market_plot(dates: List[str],
                                               time_step: str,
                                               window: str,
                                               K_value: str) -> None:
    """Plots the aggregated distribution of returns for a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_aggregated_dist_returns_market_plot.__name__
    epochs_sim_tools \
        .function_header_print_plot(function_name, dates, time_step, window,
                                    K_value)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            f'../data/epochs_sim/epochs_sim_no_rot_market_data_{dates[0]}'
            + f'_{dates[1]}_step_{time_step}_win_{window}_K_{K_value}.pickle',
            'rb'))[::10]

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_values: np.ndarray = np.arange(-10, 10, 0.1)
        gaussian: np.ndarray = epochs_sim_tools \
            .gaussian_distribution(0, 1, x_values)

        figure_log: plt.Figure = plt.figure(figsize=(16, 9))

        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', logy=True,
                                         figsize=(16, 9), legend=True, lw=5)

        plt.semilogy(x_values, gaussian, '-', lw=3, label='Gaussian')

        plt.legend(fontsize=20)
        plt.title(f'Epochs from {dates[0]} to {dates[1]} - {time_step}',
                  fontsize=30)
        plt.xlabel(f'Aggregated returns - window {window}', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-6, 6)
        plt.ylim(10 ** -5, 1)
        plt.grid(True)
        plt.tight_layout()
        figure_log = plot_log.get_figure()

        # Plotting
        epochs_sim_tools \
            .save_plot(figure_log, function_name + '_log', dates, time_step,
                       window, K_value)

        plt.close()
        del agg_returns_data
        del figure_log
        del plot_log
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
