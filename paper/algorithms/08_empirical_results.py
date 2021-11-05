'''Exact distribution empirical results implementation.

Plots the figures of the empirical results for the paper.

This script requires the following modules:
    * pickle
    * sys
    * typing
    * matplotlib
    * numpy
    * pandas
    * exact_distributions_covariance_tools

The module contains the following functions:
    * pdf_all_distributions_plot - plots all the distributions and compares
      with agg. returns of a market.
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# ----------------------------------------------------------------------------
# Modules

import pickle
import sys
from typing import List

from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(1, '../../project/exact_distributions_covariance')
import exact_distributions_covariance_tools as exact_distributions_tools

# ----------------------------------------------------------------------------


def pdf_gg_distributions_plot(dates: List[str],
                               time_step: str,
                               N_values: List[int]) -> None:
    """Plots all the distributions and compares with agg. returns of a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param N_values: fit parameter (i.e. [3, 4, 5, 6])
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    markers: List[str] = ['o', '^', 's', 'P']

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../../project/data/exact_distributions_covariance/aggregated_dist'
            + f'_returns_market_data_{dates[0]}_{dates[1]}_step_{time_step}'
            + f'.pickle', 'rb'))

        agg_returns_data = agg_returns_data.rename(r'$\tilde{r}$')

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

    # Log plot
    plot_1 = agg_returns_data.plot(kind='density', style=markers[0], logy=True,
                                   ax=ax1, legend=False, ms=7)
    plot_2 = agg_returns_data.plot(kind='density', style=markers[0],
                                   loglog=True, ax=ax2, legend=False, ms=7)

    x_val_log: np.ndarray = np.arange(-10, 10, 0.05)

    for N_value in N_values:
        gg_distribution_log: np.ndarray = exact_distributions_tools\
            .pdf_gaussian_gaussian(x_val_log, N_value, 1)
        ax1.semilogy(x_val_log, gg_distribution_log, '-', lw=5,
                     label=f'N = {N_value}')
        ax2.loglog(x_val_log, gg_distribution_log, '-', lw=5,
                   label=f'N = {N_value}')

    ax1.set_xlabel(r'$\tilde{r}$', fontsize=25)
    ax1.set_ylabel('PDF', fontsize=25)
    ax1.tick_params(axis='both', which='both', labelsize=15)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(10 ** -4, 1)
    ax1.grid(True)

    ax2.legend(loc='upper right', fontsize=20)
    ax2.set_xlabel(r'$\tilde{r}$', fontsize=20)
    ax2.set_ylabel('PDF', fontsize=20)
    ax2.tick_params(axis='both', which='both', labelsize=15)
    ax2.set_xlim(3, 5)
    ax2.set_ylim(0.5 * 10 ** -3, 10 ** -2)
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    figure.savefig(f"../plot/08_gg.png")

    plt.close()
    del agg_returns_data
    del figure
    del plot_1
    del plot_2

# ----------------------------------------------------------------------------


def pdf_ga_distributions_plot(dates: List[str],
                              time_step: str,
                              N_values: List[int],
                              K_value: int,
                              L_values: List[int]) -> None:
    """Plots all the distributions and compares with agg. returns of a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param N_values: fit parameter (i.e. [3, 4, 5, 6]).
    :param K_value: number of companies.
    :param L_values: shape parameter (i.e. [3, 4, 5, 6]).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    markers: List[str] = ['o', '^', 's', 'P']

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../../project/data/exact_distributions_covariance/aggregated_dist'
            + f'_returns_market_data_{dates[0]}_{dates[1]}_step_{time_step}'
            + f'.pickle', 'rb'))

        agg_returns_data = agg_returns_data.rename(r'$\tilde{r}$')

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

    # Log plot
    plot_1 = agg_returns_data.plot(kind='density', style=markers[0], logy=True,
                                   ax=ax1, legend=False, ms=7)
    plot_2 = agg_returns_data.plot(kind='density', style=markers[0],
                                   loglog=True, ax=ax2, legend=False, ms=7)

    x_val_log: np.ndarray = np.arange(-10, 10, 0.05)

    for N_value in N_values:
        for L_value in L_values:
            ga_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_gaussian_algebraic(x_val_log, K_value, L_value,
                                        N_value, 1)
            ax1.semilogy(x_val_log, ga_distribution_log, '-', lw=5,
                        label=f'N = {N_value} - L = {L_value}')
            ax2.loglog(x_val_log, ga_distribution_log, '-', lw=5,
                    label=f'N = {N_value} - L = {L_value}')

    ax1.set_xlabel(r'$\tilde{r}$', fontsize=25)
    ax1.set_ylabel('PDF', fontsize=25)
    ax1.tick_params(axis='both', which='both', labelsize=15)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(10 ** -4, 1)
    ax1.grid(True)

    ax2.legend(loc='upper right', fontsize=20)
    ax2.set_xlabel(r'$\tilde{r}$', fontsize=20)
    ax2.set_ylabel('PDF', fontsize=20)
    ax2.tick_params(axis='both', which='both', labelsize=15)
    ax2.set_xlim(3, 5)
    ax2.set_ylim(0.5 * 10 ** -3, 10 ** -2)
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    figure.savefig(f"../plot/08_ga.png")

    plt.close()
    del agg_returns_data
    del figure
    del plot_1
    del plot_2

# ----------------------------------------------------------------------------


def pdf_ag_distributions_plot(dates: List[str],
                              time_step: str,
                              N_values: List[int],
                              K_value: int,
                              l_values: List[int]) -> None:
    """Plots all the distributions and compares with agg. returns of a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param N_values: fit parameter (i.e. [3, 4, 5, 6]).
    :param K_value: number of companies.
    :param l_values: shape parameter (i.e. [3, 4, 5, 6]).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    markers: List[str] = ['o', '^', 's', 'P']

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../../project/data/exact_distributions_covariance/aggregated_dist'
            + f'_returns_market_data_{dates[0]}_{dates[1]}_step_{time_step}'
            + f'.pickle', 'rb'))

        agg_returns_data = agg_returns_data.rename(r'$\tilde{r}$')

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

    # Log plot
    plot_1 = agg_returns_data.plot(kind='density', style=markers[0], logy=True,
                                   ax=ax1, legend=False, ms=7)
    plot_2 = agg_returns_data.plot(kind='density', style=markers[0],
                                   loglog=True, ax=ax2, legend=False, ms=7)

    x_val_log: np.ndarray = np.arange(-10, 10, 0.05)

    for N_value in N_values:
        for l_value in l_values:
            ag_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_algebraic_gaussian(x_val_log, K_value, l_value,
                                        N_value, 1)
            ax1.semilogy(x_val_log, ag_distribution_log, '-', lw=5,
                        label=f'N = {N_value} - l = {l_value}')
            ax2.loglog(x_val_log, ag_distribution_log, '-', lw=5,
                    label=f'N = {N_value} - l = {l_value}')

    ax1.set_xlabel(r'$\tilde{r}$', fontsize=25)
    ax1.set_ylabel('PDF', fontsize=25)
    ax1.tick_params(axis='both', which='both', labelsize=15)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(10 ** -4, 1)
    ax1.grid(True)

    ax2.legend(loc='upper right', fontsize=20)
    ax2.set_xlabel(r'$\tilde{r}$', fontsize=20)
    ax2.set_ylabel('PDF', fontsize=20)
    ax2.tick_params(axis='both', which='both', labelsize=15)
    ax2.set_xlim(3, 5)
    ax2.set_ylim(0.5 * 10 ** -3, 10 ** -2)
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    figure.savefig(f"../plot/08_ag.png")

    plt.close()
    del agg_returns_data
    del figure
    del plot_1
    del plot_2

# ----------------------------------------------------------------------------


def pdf_aa_distributions_plot(dates: List[str],
                              time_step: str,
                              N_values: List[int],
                              K_value: int,
                              L_values: List[int],
                              l_values: List[int]) -> None:
    """Plots all the distributions and compares with agg. returns of a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param N_values: fit parameter (i.e. [3, 4, 5, 6]).
    :param K_value: number of companies.
    :param L_values: shape parameter (i.e. [3, 4, 5, 6]).
    :param l_values: shape parameter (i.e. [3, 4, 5, 6]).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    markers: List[str] = ['o', '^', 's', 'P']

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../../project/data/exact_distributions_covariance/aggregated_dist'
            + f'_returns_market_data_{dates[0]}_{dates[1]}_step_{time_step}'
            + f'.pickle', 'rb'))

        agg_returns_data = agg_returns_data.rename(r'$\tilde{r}$')

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

    # Log plot
    plot_1 = agg_returns_data.plot(kind='density', style=markers[0], logy=True,
                                   ax=ax1, legend=False, ms=7)
    plot_2 = agg_returns_data.plot(kind='density', style=markers[0],
                                   loglog=True, ax=ax2, legend=False, ms=7)

    x_val_log: np.ndarray = np.arange(-10, 10, 0.05)

    for N_value in N_values:
        for L_value in L_values:
            for l_value in l_values:
                aa_distribution_log: np.ndarray = exact_distributions_tools\
                    .pdf_algebraic_algebraic(x_val_log, K_value, L_value,
                                             l_value, N_value, 1)
                ax1.semilogy(x_val_log, aa_distribution_log, '-', lw=5,
                             label=f'N = {N_value} - L = {L_value}'
                                   + f' - l = {l_value}')
                ax2.loglog(x_val_log, aa_distribution_log, '-', lw=5,
                           label=f'N = {N_value} - L = {L_value}'
                                 + f' - l = {l_value}')

    ax1.set_xlabel(r'$\tilde{r}$', fontsize=25)
    ax1.set_ylabel('PDF', fontsize=25)
    ax1.tick_params(axis='both', which='both', labelsize=15)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(10 ** -4, 1)
    ax1.grid(True)

    ax2.legend(loc='upper right', fontsize=18)
    ax2.set_xlabel(r'$\tilde{r}$', fontsize=20)
    ax2.set_ylabel('PDF', fontsize=20)
    ax2.tick_params(axis='both', which='both', labelsize=15)
    ax2.set_xlim(3, 5)
    ax2.set_ylim(0.5 * 10 ** -3, 10 ** -2)
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    figure.savefig(f"../plot/08_aa.png")

    plt.close()
    del agg_returns_data
    del figure
    del plot_1
    del plot_2

# ----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

    pdf_gg_distributions_plot(['1990-01-01', '2020-12-31'], '1d', [2, 3, 4, 5])
    pdf_ga_distributions_plot(['1990-01-01', '2020-12-31'], '1d',
                              [2, 3, 4], 244, [150])
    pdf_ag_distributions_plot(['1990-01-01', '2020-12-31'], '1d',
                              [2, 3, 4, 5], 244, [150])
    pdf_aa_distributions_plot(['1990-01-01', '2020-12-31'], '1d',
                              [2, 3, 4, 5], 244, [150], [150])

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
