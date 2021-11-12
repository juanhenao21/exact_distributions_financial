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

def epochs_sim_agg_ret_pairs(K_value: int,
                             normalized: bool = False,
                             kind: str = 'gaussian') -> None:
    """Plot the simulated aggregated rotated and scaled returns without
       normalization.

    :param K_value: number of companies.
    :type K_value: int
    :param normalized: normalize the returns within the epochs, defaults to
     False
    :type normalized: bool, optional
    :param kind: kind of returns to be used, defaults to gaussian.
    :type kind: str, optional
    """

    figure = plt.figure(figsize=(16, 9))

    markers: List[str] = ['o', '^', 's', 'P', 'x']

    # Simulate the aggregated returns for different epochs lenghts
    for epochs_len, marker in zip([10, 25, 40, 55, 100], markers):
        agg_ret_pairs = epochs_sim_analysis \
            .epochs_sim_agg_returns_market_data(0.3, 2, K_value, 200,
                                                epochs_len, normalized, kind)

        agg_ret_pairs = agg_ret_pairs.rename(f'Epochs T={epochs_len}')

        # Log plot
        plot = agg_ret_pairs.plot(kind='density', style=marker, logy=True,
                                  legend=False, ms=10)

    x_values: np.ndarray = np.arange(-7, 7, 0.03)

    if kind == 'gaussian':
        gaussian: np.ndarray = epochs_sim_tools \
            .gaussian_distribution(0, 1, x_values)
        plt.semilogy(x_values, gaussian, '-', lw=10, label='Gaussian')
    if kind == 'algebraic':
        algebraic: np.ndarray = epochs_sim_tools \
            .algebraic_distribution(K_value, (10 + K_value) / 2, x_values)
        plt.semilogy(x_values, algebraic, '-', lw=10, label='Algebraic')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3,
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
    if normalized and (kind == 'gaussian'):
        figure.savefig(f'../plot/06_epochs_sim_gauss_agg_ret_pairs_norm.png')
    if normalized and (kind == 'algebraic'):
        figure.savefig(f'../plot/06_epochs_sim_alg_agg_ret_pairs_norm.png')
    if not normalized and (kind == 'gaussian'):
        figure. \
            savefig(f'../plot/06_epochs_sim_gauss_agg_ret_pairs_no_norm.png')
    if not normalized and (kind == 'algebraic'):
        figure. \
            savefig(f'../plot/06_epochs_sim_alg_agg_ret_pairs_no_norm.png')

    plt.close()
    del agg_ret_pairs
    del figure
    del plot
    gc.collect()

# ----------------------------------------------------------------------------


def epochs_sim_ts_norm_agg_ret(K_value: int,
                               kind: str) -> None:
    """Plot the simulated aggregated rotated and scaled returns with
       normalization for the full time series.

    :param K_value: number of companies.
    :type K_value: int
    :param kind: kind of returns to be used.
    :type kind: str
    """

    figure = plt.figure(figsize=(16, 9))

    markers: List[str] = ['o', '^', 's', 'P', 'x']

    try:

        # Simulate the aggregated returns for different epochs lenghts
        for epochs_len, marker in zip([10, 25, 40, 55, 100], markers):
            # Load data
            if kind == 'gaussian':
                agg_ret: pd.Series = pickle.load(open(
                    f'../../project/data/epochs_sim/epochs_sim_{kind}_agg_dist'
                    + f'_ret_market_data_long_win_{epochs_len}_K_{K_value}'
                    + f'.csv', 'rb'))
            elif kind == 'algebraic':
                agg_ret: pd.Series = pd.read_csv(
                    f'../../project/data/epochs_sim/epochs_sim_{kind}_agg_dist'
                    + f'_ret_market_data_long_win_{epochs_len}_K_{K_value}'
                    + f'.csv', names=['x'], header=0)['x']

            agg_ret= agg_ret.rename(f'Epochs T={epochs_len}')

            # Log plot
            plot = agg_ret.plot(kind='density', style=marker, logy=True,
                                legend=False, ms=10)

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

    x_values: np.ndarray = np.arange(-7, 7, 0.03)

    if kind == 'gaussian':
        gaussian: np.ndarray = epochs_sim_tools \
            .gaussian_distribution(0, 1, x_values)
        plt.semilogy(x_values, gaussian, '-', lw=10, label='Gaussian')
    if kind == 'algebraic':
        algebraic: np.ndarray = epochs_sim_tools \
            .algebraic_distribution(K_value, (10 + K_value) / 2, x_values)
        plt.semilogy(x_values, algebraic, '-', lw=10, label='Algebraic')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3,
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
    if kind == 'gaussian':
        figure.savefig(f'../plot/06_epochs_sim_gauss_ts_norm.png')
    if kind == 'algebraic':
        figure.savefig(f'../plot/06_epochs_sim_alg_ts_norm.png')

    plt.close()
    del agg_ret
    del figure
    del plot
    gc.collect()

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
    epochs_sim_tools \
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
                    + f'_returns_market_data_long_{date_val[0]}'
                    + f'_{date_val[1]}_step_{time_steps[idx]}_win_10'
                    + f'_K_{K_value}.pickle', 'rb'))
                agg_25: pd.Series = pickle.load(open(
                    '../../project/data/epochs/epochs_aggregated_dist'
                    + f'_returns_market_data_long_{date_val[0]}'
                    + f'_{date_val[1]}_step_{time_steps[idx]}_win_25'
                    + f'_K_{K_value}.pickle', 'rb'))
                agg_40: pd.Series = pickle.load(open(
                    '../../project/data/epochs/epochs_aggregated_dist'
                    + f'_returns_market_data_long_{date_val[0]}'
                    + f'_{date_val[1]}_step_{time_steps[idx]}_win_40'
                    + f'_K_{K_value}.pickle', 'rb'))
                agg_55: pd.Series = pickle.load(open(
                    '../../project/data/epochs/epochs_aggregated_dist'
                    + f'_returns_market_data_long_{date_val[0]}'
                    + f'_{date_val[1]}_step_{time_steps[idx]}_win_55'
                    + f'_K_{K_value}.pickle', 'rb'))

                agg_10 = agg_10.rename(f'K = {K_value}')
                agg_25 = agg_25.rename(f'K = {K_value}')
                agg_40 = agg_40.rename(f'K = {K_value}')
                agg_55 = agg_55.rename(f'K = {K_value}')

                plot_10 = agg_10.plot(kind='density', style=marker, ax=ax1,
                                      logy=True, figsize=(16, 9), ms=5)
                plot_25 = agg_25.plot(kind='density', style=marker, ax=ax2,
                                      logy=True, figsize=(16, 9), ms=5)
                plot_40 = agg_40.plot(kind='density', style=marker, ax=ax3,
                                      logy=True, figsize=(16, 9), ms=5)
                plot_55 = agg_55.plot(kind='density', style=marker, ax=ax4,
                                      logy=True, figsize=(16, 9), ms=5)

                figure_log = plot_10.get_figure()
                figure_log = plot_25.get_figure()
                figure_log = plot_40.get_figure()
                figure_log = plot_55.get_figure()

            x_values: np.ndarray = np.arange(-10, 10, 0.1)
            gaussian: np.ndarray = epochs_sim_tools \
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
            ax3.legend(loc='upper center', bbox_to_anchor=(1.0, -0.2), ncol=6,
                   fontsize=15)
            ax1.set_yticks(ax1.get_yticks()[2:-1])
            ax3.set_yticks(ax3.get_yticks()[1:-2])
            ax3.set_xticks(ax3.get_xticks()[:-1])
            ax4.set_xticks(ax4.get_xticks()[1:])

            plt.tight_layout()

            # Plotting
            figure_log.savefig(f'../plot/06_window_comparison.png')

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

    # epochs_sim_agg_ret_pairs(200, normalized=False, kind='gaussian')
    # epochs_sim_agg_ret_pairs(200, normalized=True, kind='gaussian')
    # epochs_sim_agg_ret_pairs(200, normalized=False, kind='algebraic')
    # epochs_sim_agg_ret_pairs(200, normalized=True, kind='algebraic')

    # epochs_sim_ts_norm_agg_ret(200, 'gaussian')
    # epochs_sim_ts_norm_agg_ret(200, 'algebraic')

    # epochs_var_win_all_empirical_dist_returns_market_plot(dates, time_steps,
    #                                                       [''],
    #                                                       ['20', '100', 'all'])


# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
