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
                + f'_market_data_short_{date[0]}_{date[1]}_step_{time_step}'
                + f'_win_{window}_K_{K_value}.pickle', 'rb'))

            agg = agg.rename(f'Agg. returns {time_step}')

            # Log plot
            plot = agg.plot(kind='density', style=marker, logy=True,
                            legend=False, ms=10)

        x_gauss: np.ndarray = np.arange(-10, 10, 0.3)
        gaussian: np.ndarray = epochs_tools \
            .gaussian_distribution(0, 1, x_gauss)

        plt.semilogy(x_gauss, gaussian, '-', lw=10, label='Gaussian')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2,
                   fontsize=30)
        plt.xlabel(r'$\tilde{r}$', fontsize=40)
        plt.ylabel('PDF', fontsize=40)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlim(-6, 6)
        plt.ylim(10 ** -4, 1)
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        figure.savefig(f'../plot/05_gaussian_agg_returns_short_epoch.png')

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

        figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

        markers: List[str] = ['o', '^', 's', 'P', 'x']

        for time_step, date, marker in zip(time_steps, dates, markers):
            # Load data
            agg: pd.Series = pickle.load(open(
                '../../project/data/epochs/epochs_aggregated_dist_returns'
                + f'_market_data_short_{date[0]}_{date[1]}_step_{time_step}'
                + f'_win_{window}_K_{K_value}.pickle', 'rb'))

            agg = agg.rename(f'Agg. returns {time_step}')

            # Log plot
            plot_1 = agg.plot(kind='density', style=marker, logy=True, ax=ax1,
                              legend=False, ms=7)
            plot_2 = agg.plot(kind='density', style=marker, loglog=True,
                              ax=ax2, legend=False, ms=7)

        x_values: np.ndarray = np.arange(-10, 10, 0.3)

        if K_value == 'all':
            K_value = 200

        for l_value in l_values:
            algebraic: np.ndarray = epochs_tools \
                .algebraic_distribution(int(K_value), l_value, x_values)
            ax1.semilogy(x_values, algebraic, '-', lw=5,
                         label=f'Algebraic l = {l_value}')
            ax2.loglog(x_values, algebraic, '-', lw=5,
                       label=f'Algebraic l = {l_value}')

        gaussian: np.ndarray = epochs_tools \
            .gaussian_distribution(0, 1, x_values)
        ax1.semilogy(x_values, gaussian, '-', lw=5, label=f'Gaussian')
        ax2.loglog(x_values, gaussian, '-', lw=5, label=f'Gaussian')

        ax1.set_xlabel(r'$\tilde{r}$', fontsize=20)
        ax1.set_ylabel('PDF', fontsize=20)
        ax1.tick_params(axis='both', labelsize=15)
        ax1.set_xlim(-6, 6)
        ax1.set_ylim(10 ** -4, 1)
        ax1.grid(True)

        ax2.legend(loc='upper center', bbox_to_anchor=(1.4, 0.6), ncol=1,
                   fontsize=20)
        ax2.set_xlabel(r'$\tilde{r}$', fontsize=20)
        ax2.set_ylabel('PDF', fontsize=20)
        ax2.tick_params(axis='both', which='both', labelsize=15)
        ax2.set_xlim(3, 5)
        ax2.set_ylim(10 ** -4, 10 ** -2)
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        figure.savefig(f'../plot/05_algebraic_agg_returns_short_epoch.png')

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


def epochs_var_win_all_empirical_dist_returns_market_plot(
                                            dates: List[str],
                                            time_steps: List[str],
                                            windows: List[str],
                                            K_values: List[str]) -> None:
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

        figure, ((ax1, ax2), (ax3, ax4)) = \
            plt.subplots(2, 2, figsize=(16, 9), sharex='col', sharey='row',
                         gridspec_kw={'hspace': 0, 'wspace': 0})

        markers: List[str] = ['o', '^', 's', 'P', 'x']

        for idx, date_val in enumerate(dates):

            for K_value, marker in zip(K_values, markers):

                # Load data
                agg_10: pd.Series = pickle.load(open(
                    '../../project/data/epochs/epochs_aggregated_dist'
                    + f'_returns_market_data_short_{date_val[0]}'
                    + f'_{date_val[1]}_step_{time_steps[idx]}_win_10'
                    + f'_K_{K_value}.pickle', 'rb'))
                agg_25: pd.Series = pickle.load(open(
                    '../../project/data/epochs/epochs_aggregated_dist'
                    + f'_returns_market_data_short_{date_val[0]}'
                    + f'_{date_val[1]}_step_{time_steps[idx]}_win_25'
                    + f'_K_{K_value}.pickle', 'rb'))
                agg_40: pd.Series = pickle.load(open(
                    '../../project/data/epochs/epochs_aggregated_dist'
                    + f'_returns_market_data_short_{date_val[0]}'
                    + f'_{date_val[1]}_step_{time_steps[idx]}_win_40'
                    + f'_K_{K_value}.pickle', 'rb'))
                agg_55: pd.Series = pickle.load(open(
                    '../../project/data/epochs/epochs_aggregated_dist'
                    + f'_returns_market_data_short_{date_val[0]}'
                    + f'_{date_val[1]}_step_{time_steps[idx]}_win_55'
                    + f'_K_{K_value}.pickle', 'rb'))

                agg_10 = agg_10.rename(f'K = {K_value}')
                agg_25 = agg_25.rename(f'K = {K_value}')
                agg_40 = agg_40.rename(f'K = {K_value}')
                agg_55 = agg_55.rename(f'K = {K_value}')

                plot_10 = agg_10.plot(kind='density', style=marker, ax=ax1,
                                      logy=True, figsize=(16, 9), ms=10)
                plot_25 = agg_25.plot(kind='density', style=marker, ax=ax2,
                                      logy=True, figsize=(16, 9), ms=10)
                plot_40 = agg_40.plot(kind='density', style=marker, ax=ax3,
                                      logy=True, figsize=(16, 9), ms=10)
                plot_55 = agg_55.plot(kind='density', style=marker, ax=ax4,
                                      logy=True, figsize=(16, 9), ms=10)

                figure_log = plot_10.get_figure()
                figure_log = plot_25.get_figure()
                figure_log = plot_40.get_figure()
                figure_log = plot_55.get_figure()

            x_values: np.ndarray = np.arange(-10, 10, 0.1)
            gaussian: np.ndarray = epochs_tools \
                .gaussian_distribution(0, 1, x_values)

            # Log plot
            ax1.semilogy(x_values, gaussian, '-', lw=10, label='Gaussian')
            ax2.semilogy(x_values, gaussian, '-', lw=10, label='Gaussian')
            ax3.semilogy(x_values, gaussian, '-', lw=10, label='Gaussian')
            ax4.semilogy(x_values, gaussian, '-', lw=10, label='Gaussian')

            for ax in [ax1, ax2, ax3, ax4]:
                ax.set
                ax.set_xlabel(r'$\tilde{r}$', fontsize=25)
                ax.set_ylabel('PDF', fontsize=25)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_xlim(-6, 6)
                ax.set_ylim(10 ** -4, 1)
                ax.grid(True)
            # for ax in [ax1, ax3]:
            #     labels_y = ax.get_yticklabels()
            #     labels_y[-1] = ""
            #     ax.set_yticklabels(labels_y)
            # for ax in [ax3, ax4]:
            #     labels_x = ax.get_xticklabels()
            #     labels_x[-1] = ""
            #     ax.set_xticklabels(labels_x)
            ax3.legend(loc='upper center', bbox_to_anchor=(1.0, -0.2), ncol=4,
                       fontsize=15)
            ax1.set_yticks(ax1.get_yticks()[2:-1])
            ax3.set_yticks(ax3.get_yticks()[1:-2])
            ax3.set_xticks(ax3.get_xticks()[:-1])
            ax4.set_xticks(ax4.get_xticks()[1:])

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

    dates: List[List[str]] = [['1990-01-01', '2020-12-31']]
    time_steps: List[str] = ['1d']

    # epochs_gaussian_agg_dist_returns_market_plot(dates, time_steps, '25',
    #                                              'all')
    epochs_algebraic_agg_dist_returns_market_plot(dates, time_steps,
                                                  '55', 'all',
                                                  [104])
    epochs_var_win_all_empirical_dist_returns_market_plot(dates, time_steps,
                                                          [''],
                                                          ['20', '100', 'all'])

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
