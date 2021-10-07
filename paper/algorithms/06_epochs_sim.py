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

sys.path.insert(1, '../../project/epochs')
import epochs_tools  # type: ignore

# ----------------------------------------------------------------------------


def epochs_gaussian_agg_dist_returns_market_plot(dates: List[List[str]],
                                                 time_steps: List[str],
                                                 window: str,
                                                 K_value: str) -> None:
    """Plots the aggregated distribution of returns for a market.

    :param dates: list of the interval of dates to be analyzed
     (i.e. [['1992-01', '2012-12'], ['2012-01', '2020-12']).
    :param time_steps: list of the time step of the data (i.e. ['1m', '1h']).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    try:

        figure = plt.figure(figsize=(16, 9))

        markers: List[str] = ['o', '^', 's', 'P', 'x']

        for time_step, date, marker in zip(time_steps, dates, markers):
            # Load data
            agg: pd.Series = pickle.load(open(
                '../../project/data/epochs/epochs_aggregated_dist_returns'
                + f'_market_data_{date[0]}_{date[1]}_step_{time_step}_win'
                + f'_{window}_K_{K_value}.pickle', 'rb'))

            agg = agg.rename(f'Agg. returns {time_step}')

            # Log plot
            plot = agg.plot(kind='density', style=marker, logy=True,
                            legend=False, ms=10)

        x_gauss: np.ndarray = np.arange(-10, 10, 0.3)
        gaussian: np.ndarray = epochs_tools \
            .gaussian_distribution(0, 1, x_gauss)

        plt.semilogy(x_gauss, gaussian, '-', lw=10, label='Gaussian')

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
        figure.savefig(f'../plot/05_gaussian_agg_returns_epoch.png')

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


def epochs_algebraic_agg_dist_returns_market_plot(dates: List[List[str]],
                                                  time_steps: List[str],
                                                  window: str,
                                                  K_value: str,
                                                  l_values: List[int]) -> None:
    """Plots the aggregated distribution of returns for a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. [['1992-01', '2012-12'], ['2012-01', '2020-12']).
    :param time_steps: list of the time step of the data (i.e. ['1m', '1h']).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :param l_value: shape parameter for the algebraic distribution (i.e. 1, 2).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    try:

        figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 16))
        ax1.get_shared_y_axes().join(ax2)

        markers: List[str] = ['o', '^', 's', 'P', 'x']

        for time_step, date, marker in zip(time_steps, dates, markers):
            # Load data
            agg: pd.Series = pickle.load(open(
                '../../project/data/epochs/epochs_aggregated_dist_returns'
                + f'_market_data_{date[0]}_{date[1]}_step_{time_step}_win'
                + f'_{window}_K_{K_value}.pickle', 'rb'))

            agg = agg.rename(f'Agg. returns {time_step}')

            # Log plot
            plot_1 = agg.plot(kind='density', style=marker, logy=True, ax=ax1,
                              legend=False, ms=7)
            plot_2 = agg.plot(kind='density', style=marker, loglog=True,
                              ax=ax2, legend=False, ms=7)

        x_values: np.ndarray = np.arange(-10, 10, 0.3)

        for l_value in l_values:
            algebraic: np.ndarray = epochs_tools \
                .algebraic_distribution(int(K_value), l_value, x_values)
            ax1.semilogy(x_values, algebraic, '-', lw=5, alpha=0.5,
                         label=f'Algebraic l = {l_value}')
            ax2.loglog(x_values, algebraic, '-', lw=5, alpha=0.5,
                       label=f'Algebraic l = {l_value}')

        gaussian: np.ndarray = epochs_tools \
            .gaussian_distribution(0, 1, x_values)
        ax1.semilogy(x_values, gaussian, '-', lw=5, label=f'Gaussian')
        ax2.loglog(x_values, gaussian, '-', lw=5, label=f'Gaussian')

        ax1.set_xlabel(r'$\tilde{r}$', fontsize=20)
        ax1.set_ylabel('PDF', fontsize=20)
        ax1.tick_params(axis='both', labelsize=15)
        ax1.set_xlim(-6, 6)
        ax1.set_ylim(10 ** -5, 1)
        ax1.grid(True)

        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2,
                   fontsize=20)
        ax2.set_xlabel(r'$\tilde{r}$', fontsize=20)
        ax2.set_ylabel('PDF', fontsize=20)
        ax2.tick_params(axis='both', which='both', labelsize=15)
        ax2.set_xlim(3, 5)
        ax2.set_ylim(10 ** -5, 10 ** -2)
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        figure.savefig(f'../plot/05_algebraic_agg_returns_epoch.png')

        plt.close()
        del agg
        del figure
        del plot_1
        del plot_2
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# ----------------------------------------------------------------------------


def epochs_var_win_all_empirical_dist_returns_market_plot() -> None:
    """Plots all the local normalized aggregated distributions of returns for a
       market in different epochs window lengths.

    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = \
        epochs_var_win_all_empirical_dist_returns_market_plot.__name__
    epochs_tools \
        .function_header_print_plot(function_name, ['', ''], '', '', '')

    try:

        dates = ['1990-01-01', '2020-12-31']

        windows = ['10', '25', '40', '55']
        K_values = ['50']
        dates_vals = [dates]
        time_steps = ['1d']

        markers: List[str] = ['o', '^', 's', 'P', 'x']

        for K_value in K_values:
            for idx, date_val in enumerate(dates_vals):

                figure_log: plt.Figure = plt.figure(figsize=(16, 9))


                for window, marker in zip(windows, markers):

                    # Load data
                    agg_returns: pd.Series = pickle.load(open(
                        '../../project/data/epochs/epochs_aggregated_dist_returns_market'
                        + f'_data_{date_val[0]}_{date_val[1]}'
                        + f'_step_{time_steps[idx]}_win_{window}_K_{K_value}'
                        + '.pickle', 'rb'))

                    agg_returns = agg_returns.rename(f'Epochs window {window}')

                    plot_log = agg_returns.plot(kind='density', style=marker,
                                                logy=True, figsize=(16, 9),
                                                legend=True, ms=10)

                    figure_log = plot_log.get_figure()

                x_values: np.ndarray = np.arange(-10, 10, 0.1)
                gaussian: np.ndarray = epochs_tools \
                    .gaussian_distribution(0, 1, x_values)

                # Log plot
                plt.semilogy(x_values, gaussian, '-', lw=10, label='Gaussian')

                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3,
                        fontsize=25)
                plt.xlabel(r'$\tilde{r}$', fontsize=25)
                plt.ylabel('PDF', fontsize=25)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlim(-6, 6)
                plt.ylim(10 ** -5, 1)
                plt.grid(True)
                plt.tight_layout()

                # Plotting
                figure_log.savefig(f'../plot/05_window_comparison.png')

        plt.close()
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

    dates: List[List[str]] = [['2021-07-19', '2021-08-14'],
                              ['1990-01-01', '2020-12-31'],
                              ['1990-01-01', '2020-12-31'],
                              ['1990-01-01', '2020-12-31']]
    time_steps: List[str] = ['1m', '1d', '1wk', '1mo']

    # epochs_gaussian_agg_dist_returns_market_plot(dates, time_steps, '25', '50')
    # epochs_algebraic_agg_dist_returns_market_plot(dates, time_steps,
    #                                               '55', '50',
    #                                               [29, 30, 33, 36])
    epochs_var_win_all_empirical_dist_returns_market_plot()

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
