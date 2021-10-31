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
    plt.ylim(10 ** -5, 1)
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
            agg_ret: pd.Series = pickle.load(open(
                f'../data/epochs_sim/epochs_sim_{kind}_agg_dist_ret_market'
                + f'_data_long_win_{epochs_len}_K_{K_value}.pickle', 'rb'))

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
    plt.ylim(10 ** -5, 1)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    if kind == 'gaussian':
        figure.savefig(f'../plot/06_epochs_sim_gauss_ts_norm.png')
    if kind == 'algebraic':
        figure.savefig(f'../plot/06_epochs_sim_alg_ts_norm.png')

    plt.close()
    # del agg_ret_pairs
    del figure
    # del plot
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

    # epochs_sim_agg_ret_pairs(200, normalized=False, kind='gaussian')
    # epochs_sim_agg_ret_pairs(200, normalized=True, kind='gaussian')
    # epochs_sim_agg_ret_pairs(200, normalized=False, kind='algebraic')
    # epochs_sim_agg_ret_pairs(200, normalized=True, kind='algebraic')

    epochs_sim_ts_norm_agg_ret(200, 'gaussian')
    epochs_sim_ts_norm_agg_ret(200, 'algebraic')

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
