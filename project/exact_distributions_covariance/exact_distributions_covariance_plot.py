'''Exact distributions covariance plot module.

The functions in the module plot the data obtained in the
exact_distributions_covariance_analysis module.

This script requires the following modules:
    * gc
    * pickle
    * typing
    * matplotlib
    * numpy
    * pandas
    * seaborn
    * exact_distributions_covariance_tools

The module contains the following functions:
    * returns_plot - plots the returns of five stocks.
    * aggregated_dist_returns_market_plot - plots the aggregated distribution
      of returns for a market.
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
import seaborn as sns  # type: ignore

import exact_distributions_covariance_tools

# -----------------------------------------------------------------------------


def returns_plot(dates: List[str], time_step: str) -> None:
    """Plots the returns of five stocks.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = returns_plot.__name__
    exact_distributions_covariance_tools \
        .function_header_print_plot(function_name, dates, time_step)

    try:

        # Load data
        returns_data: pd.DataFrame = pickle.load(open(
                        f'../data/exact_distributions_covariance/returns_data'
                        + f'_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
                        'rb')).iloc[:, :5]

        plot: np.ndarray = returns_data.plot(subplots=True, sharex=True,
                                             figsize=(16, 16), grid=True,
                                             sort_columns=True)

        _ = [ax.set_ylabel('Returns', fontsize=20) for ax in plot]
        _ = [plot.legend(loc=1, fontsize=20) for plot in plt.gcf().axes]
        plt.xlabel(f'Date - {time_step}', fontsize=20)
        plt.tight_layout(pad=0.5)
        figure: plt.Figure = plot[0].get_figure()

        # Plotting
        exact_distributions_covariance_tools \
            .save_plot(figure, function_name, dates, time_step)

        plt.close()
        del returns_data
        del figure
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def aggregated_dist_returns_market_plot(dates: List[str],
                                        time_step: str) -> None:
    """Plots the aggregated distribution of returns for a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = aggregated_dist_returns_market_plot.__name__
    exact_distributions_covariance_tools \
        .function_header_print_plot(function_name, dates, time_step)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../data/exact_distributions_covariance/aggregated_dist_returns'
            + f'_market_data_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
            'rb'))[::10]

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_gauss: np.ndarray = np.arange(-10, 10, 0.1)
        gaussian: np.ndarray = exact_distributions_covariance_tools \
            .gaussian_distribution(0, 1, x_gauss)

        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', logy=True,
                                         figsize=(16, 9), legend=True, lw=3)

        plt.semilogy(x_gauss, gaussian, '-', lw=3, label='Gaussian')

        plt.legend(fontsize=20)
        plt.title(f'Aggregated distribution returns from {dates[0]} to'
                  + f' {dates[1]} - {time_step}', fontsize=30)
        plt.xlabel('Aggregated returns', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-6, 6)
        plt.ylim(10 ** -4, 1)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        left, bottom, width, height = [0.36, 0.13, 0.35, 0.3]
        ax2 = figure_log.add_axes([left, bottom, width, height])
        agg_returns_data.plot(kind='density', style='-', legend=False, lw=3)
        ax2.plot(x_gauss, gaussian, '-')
        plt.xlim(-2, 2)
        plt.ylim(0, 0.7)
        plt.grid(True)

        # Plotting
        exact_distributions_covariance_tools \
            .save_plot(figure_log, function_name + '_log', dates, time_step)

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


def pdf_gg_plot(dates: List[str], time_step: str, N_values: List[int]) -> None:
    """Plots the Gaussian-Gaussian PDF and compares with agg. returns of a
       market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data
     (i.e. '1m', '1h', '1d', '1wk', '1mo')
    :param N_values: list of the values of the parameter N (i.e. [3, 4, 5, 6]).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = pdf_gg_plot.__name__
    exact_distributions_covariance_tools \
        .function_header_print_plot(function_name, dates, time_step)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../data/exact_distributions_covariance/aggregated_dist_returns'
            + f'_market_data_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
            'rb'))[::10]

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_val: np.ndarray = np.arange(-10, 10, 0.1)

        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', logy=True,
                                         figsize=(16, 9), legend=True, lw=5)

        for N_val in N_values:

            gg_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_gaussian(x_val, N_val, 1)
            plt.semilogy(x_val, gg_distribution, '-', lw=1,
                            label=f'GG - N = {N_val}')

        plt.legend(fontsize=20, loc='upper left')
        plt.title(f'Aggregated distribution returns from {dates[0]} to'
                  + f' {dates[1]} - {time_step}', fontsize=30)
        plt.xlabel('Aggregated returns', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-6, 6)
        plt.ylim(10 ** -4, 1)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        left, bottom, width, height = [0.36, 0.13, 0.35, 0.3]
        ax2 = figure_log.add_axes([left, bottom, width, height])
        agg_returns_data.plot(kind='density', style='-', legend=False, lw=5)

        for N_val in N_values:

            gg_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_gaussian(x_val, N_val, 1)
            ax2.plot(x_val, gg_distribution, '-', lw=1)

        plt.xlim(-2, 2)
        plt.ylim(0, 0.7)
        plt.grid(True)

        # Plotting
        exact_distributions_covariance_tools \
            .save_plot(figure_log, function_name, dates, time_step)

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


def pdf_loglog_gg_plot(dates: List[str], time_step: str,
                       N_values: List[int]) -> None:
    """Plots the log-log Gaussian-Gaussian PDF and compares with agg. returns
       of a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data
     (i.e. '1m', '1h', '1d', '1wk', '1mo')
    :param N_values: list of the values of the parameter N (i.e. [3, 4, 5, 6]).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = pdf_loglog_gg_plot.__name__
    exact_distributions_covariance_tools \
        .function_header_print_plot(function_name, dates, time_step)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../data/exact_distributions_covariance/aggregated_dist_returns'
            + f'_market_data_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
            'rb'))[::10]

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_val: np.ndarray = np.arange(-10, 10, 0.1)

        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', loglog=True,
                                         figsize=(16, 9), legend=True, lw=5)

        for N_val in N_values:

            gg_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_gaussian(x_val, N_val, 1)
            plt.loglog(x_val, gg_distribution, '-', lw=1,
                            label=f'GG - N = {N_val}')

        plt.legend(fontsize=20, loc='upper right')
        plt.title(f'Aggregated distribution returns from {dates[0]} to'
                  + f' {dates[1]} - {time_step}', fontsize=30)
        plt.xlabel('Aggregated returns', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(3, 6)
        plt.ylim(10 ** -4, 10 ** -2)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        # Plotting
        exact_distributions_covariance_tools \
            .save_plot(figure_log, function_name, dates, time_step)

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


def pdf_ga_plot(dates: List[str], time_step: str, L_values: List[int],
                N_values: List[int]) -> None:
    """Plots the Gaussian-Algebraic PDF and compares with agg. returns of a
       market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data
     (i.e. '1m', '1h', '1d', '1wk', '1mo')
    :param L_value: value of the shape parameter L (i.e. '2').
    :param N_values: list of the values of the parameter N (i.e. [3, 4, 5, 6]).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = pdf_ga_plot.__name__
    exact_distributions_covariance_tools \
        .function_header_print_plot(function_name, dates, time_step)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../data/exact_distributions_covariance/aggregated_dist_returns'
            + f'_market_data_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
            'rb'))

        K_value: int = len(pickle.load(open(
                        f'../data/exact_distributions_covariance/returns_data'
                        + f'_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
                        'rb')).columns)

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_val: np.ndarray = np.arange(-10, 10, 0.1)

        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', logy=True,
                                         figsize=(16, 9), legend=True, lw=3)

        for N_val in N_values:

            ga_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_algebraic(x_val, K_value, L_value, N_val, 1)
            plt.semilogy(x_val, ga_distribution, '-', lw=1,
                            label=f'GA - N = {N_val}')


        ga_distribution: np.ndarray = \
            exact_distributions_covariance_tools \
            .pdf_gaussian_algebraic(x_val, K, L, N, 1)
        plt.semilogy(x_val, ga_distribution, 'o', lw=3,
                        label=f'GA - N = {N} - K = {K} - L = {L}')

        plt.legend(fontsize=20)
        plt.title(f'Aggregated distribution returns from {dates[0]} to'
                  + f' {dates[1]} - {time_step}', fontsize=30)
        plt.xlabel('Aggregated returns', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-10, 10)
        plt.ylim(10 ** -9, 1)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        left, bottom, width, height = [0.36, 0.13, 0.35, 0.3]
        ax2 = figure_log.add_axes([left, bottom, width, height])
        agg_returns_data.plot(kind='density', style='-', legend=False, lw=3)
        ax2.plot(x_val, ga_distribution, 'o')
        plt.xlim(-2, 2)
        plt.ylim(0, 0.6)
        plt.grid(True)

        # Plotting
        exact_distributions_covariance_tools \
            .save_plot(figure_log, function_name, dates, time_step)

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


def pdf_ag_plot(dates: List[str], time_step: str) -> None:
    """Plots the Algebraic-Gaussian PDF and compares with agg. returns of a
       market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = pdf_ag_plot.__name__
    exact_distributions_covariance_tools \
        .function_header_print_plot(function_name, dates, time_step)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../data/exact_distributions_covariance/aggregated_dist_returns'
            + f'_market_data_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
            'rb'))

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_val: np.ndarray = np.arange(-10, 10, 0.1)

        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', logy=True,
                                         figsize=(16, 9), legend=True, lw=3)

        if dates[0] == '1972-01':
            N = 5.5
            K = 23
            l = 55
        elif dates[0] == '1992-01':
            N = 5
            K = 277
            l = 150
        else:
            N = 7
            K = 461
            l = 280

        ag_distribution: np.ndarray = \
            exact_distributions_covariance_tools \
            .pdf_algebraic_gaussian(x_val, K, l, N, 1)
        plt.semilogy(x_val, ag_distribution, 'o', lw=3,
                        label=f'AG - N = {N} - K = {K} - l = {l}')

        plt.legend(fontsize=20)
        plt.title(f'Aggregated distribution returns from {dates[0]} to'
                  + f' {dates[1]} - {time_step}', fontsize=30)
        plt.xlabel('Aggregated returns', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-10, 10)
        plt.ylim(10 ** -9, 1)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        left, bottom, width, height = [0.36, 0.13, 0.35, 0.3]
        ax2 = figure_log.add_axes([left, bottom, width, height])
        agg_returns_data.plot(kind='density', style='-', legend=False, lw=3)
        ax2.plot(x_val, ag_distribution, 'o')
        plt.xlim(-2, 2)
        plt.ylim(0, 0.6)
        plt.grid(True)

        # Plotting
        exact_distributions_covariance_tools \
            .save_plot(figure_log, function_name, dates, time_step)

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


def pdf_aa_plot(dates: List[str], time_step: str) -> None:
    """Plots the Algebraic-Algebraic PDF and compares with agg. returns of a
       market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = pdf_aa_plot.__name__
    exact_distributions_covariance_tools \
        .function_header_print_plot(function_name, dates, time_step)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../data/exact_distributions_covariance/aggregated_dist_returns'
            + f'_market_data_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
            'rb'))

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_val: np.ndarray = np.arange(-10, 10, 0.1)

        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', logy=True,
                                         figsize=(16, 9), legend=True, lw=3)

        if dates[0] == '1972-01':
            N = 6
            K = 23
            L = 55
            l = 55
        elif dates[0] == '1992-01':
            N = 6
            K = 277
            L = 150
            l = 150
        else:
            N = 7
            K = 461
            L = 280
            l = 280

        aa_distribution: np.ndarray = \
            exact_distributions_covariance_tools \
            .pdf_algebraic_algebraic(x_val, K, L, l, N, 1)
        plt.semilogy(x_val, aa_distribution, 'o', lw=3,
                        label=f'AA - N = {N} - K = {K} - L = {L}'
                        + f' - l = {l}')

        plt.legend(loc='upper right', fontsize=20)
        plt.title(f'Aggregated distribution returns from {dates[0]} to'
                  + f' {dates[1]} - {time_step}', fontsize=30)
        plt.xlabel('Aggregated returns', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-10, 10)
        plt.ylim(10 ** -9, 1)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        left, bottom, width, height = [0.36, 0.13, 0.35, 0.3]
        ax2 = figure_log.add_axes([left, bottom, width, height])
        agg_returns_data.plot(kind='density', style='-', legend=False, lw=3)
        ax2.plot(x_val, aa_distribution, 'o')
        plt.xlim(-2, 2)
        plt.ylim(0, 0.6)
        plt.grid(True)

        # Plotting
        exact_distributions_covariance_tools \
            .save_plot(figure_log, function_name, dates, time_step)

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


def pdf_lin_all_distributions_plot(dates: List[str], time_step: str) -> None:
    """Plots all the distributions and compares with agg. returns of a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = pdf_lin_all_distributions_plot.__name__
    exact_distributions_covariance_tools \
        .function_header_print_plot(function_name, dates, time_step)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../data/exact_distributions_covariance/aggregated_dist_returns'
            + f'_market_data_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
            'rb'))

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_val: np.ndarray = np.arange(-10, 10, 0.1)

        # Lin plot
        plot_lin = agg_returns_data.plot(kind='density', style='-',
                                         figsize=(16, 9), legend=True, lw=3)

        if dates[0] == '1972-01':
            N = 5.5
            N_aa = 6
            K = 23
            L = 55
            l = 55

            gg_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_gaussian(x_val, N, 1)
            plt.plot(x_val, gg_distribution, 'o', lw=3,
                         label=f'GG - N = {N}')

            ga_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_algebraic(x_val, K, L, N, 1)
            plt.plot(x_val, ga_distribution, 'o', lw=3,
                         label=f'GA - N = {N} - K = {K} - L = {L}')

            ag_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_gaussian(x_val, K, l, N, 1)
            plt.plot(x_val, ag_distribution, 'o', lw=3,
                         label=f'AG - N = {N} - K = {K} - l = {l}')

            aa_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_algebraic(x_val, K, L, l, N_aa, 1)
            plt.plot(x_val, aa_distribution, 'o', lw=3,
                         label=f'AA - N = {N_aa} - K = {K} - L = {L}'
                         + f' - l = {l}')

        elif dates[0] == '1992-01':
            N = 5
            N_gg = 4
            N_aa = 6
            K = 277
            L = 150
            l = 150

            gg_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_gaussian(x_val, N_gg, 1)
            plt.plot(x_val, gg_distribution, 'o', lw=3,
                         label=f'GG - N = {N_gg}')

            ga_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_algebraic(x_val, K, L, N, 1)
            plt.plot(x_val, ga_distribution, 'o', lw=3,
                         label=f'GA - N = {N} - K = {K} - L = {L}')

            ag_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_gaussian(x_val, K, l, N, 1)
            plt.plot(x_val, ag_distribution, 'o', lw=3,
                         label=f'AG - N = {N} - K = {K} - l = {l}')

            aa_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_algebraic(x_val, K, L, l, N_aa, 1)
            plt.plot(x_val, aa_distribution, 'o', lw=3,
                         label=f'AA - N = {N_aa} - K = {K} - L = {L}'
                         + f' - l = {l}')

        else:
            N = 7
            N_gg = 6
            K = 461
            L = 280
            l = 280

            gg_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_gaussian(x_val, N_gg, 1)
            plt.plot(x_val, gg_distribution, 'o', lw=3,
                         label=f'GG - N = {N_gg}')

            ga_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_algebraic(x_val, K, L, N, 1)
            plt.plot(x_val, ga_distribution, 'o', lw=3,
                         label=f'GA - N = {N} - K = {K} - L = {L}')

            ag_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_gaussian(x_val, K, l, N, 1)
            plt.plot(x_val, ag_distribution, 'o', lw=3,
                         label=f'AG - N = {N} - K = {K} - l = {l}')

            aa_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_algebraic(x_val, K, L, l, N, 1)
            plt.plot(x_val, aa_distribution, 'o', lw=3,
                         label=f'AA - N = {N} - K = {K} - L = {L} - l = {l}')

        plt.legend(fontsize=20)
        plt.title(f'Aggregated distribution returns from {dates[0]} to'
                  + f' {dates[1]} - {time_step}', fontsize=30)
        plt.xlabel('Aggregated returns', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-2, 2)
        plt.ylim(0, 0.6)
        plt.grid(True)
        plt.tight_layout()
        figure_lin: plt.Figure = plot_lin.get_figure()

        # Plotting
        exact_distributions_covariance_tools \
            .save_plot(figure_lin, function_name, dates, time_step)

        plt.close()
        del agg_returns_data
        del figure_lin
        del plot_lin
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def pdf_log_all_distributions_plot(dates: List[str], time_step: str) -> None:
    """Plots all the distributions and compares with agg. returns of a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = pdf_log_all_distributions_plot.__name__
    exact_distributions_covariance_tools \
        .function_header_print_plot(function_name, dates, time_step)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../data/exact_distributions_covariance/aggregated_dist_returns'
            + f'_market_data_{dates[0]}_{dates[1]}_step_{time_step}.pickle',
            'rb'))

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_val: np.ndarray = np.arange(-10, 10, 0.1)

        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', logy=True,
                                         figsize=(16, 9), legend=True, lw=3)

        if dates[0] == '1972-01':
            N = 5.5
            N_aa = 6
            K = 23
            L = 55
            l = 55

            gg_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_gaussian(x_val, N, 1)
            plt.semilogy(x_val, gg_distribution, 'o', lw=3,
                         label=f'GG - N = {N}')

            ga_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_algebraic(x_val, K, L, N, 1)
            plt.semilogy(x_val, ga_distribution, 'o', lw=3,
                         label=f'GA - N = {N} - K = {K} - L = {L}')

            ag_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_gaussian(x_val, K, l, N, 1)
            plt.semilogy(x_val, ag_distribution, 'o', lw=3,
                         label=f'AG - N = {N} - K = {K} - l = {l}')

            aa_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_algebraic(x_val, K, L, l, N_aa, 1)
            plt.semilogy(x_val, aa_distribution, 'o', lw=3,
                         label=f'AA - N = {N_aa} - K = {K} - L = {L}'
                         + f' - l = {l}')

        elif dates[0] == '1992-01':
            N = 5
            N_gg = 4
            N_aa = 6
            K = 277
            L = 150
            l = 150

            gg_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_gaussian(x_val, N_gg, 1)
            plt.semilogy(x_val, gg_distribution, 'o', lw=3,
                         label=f'GG - N = {N_gg}')

            ga_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_algebraic(x_val, K, L, N, 1)
            plt.semilogy(x_val, ga_distribution, 'o', lw=3,
                         label=f'GA - N = {N} - K = {K} - L = {L}')

            ag_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_gaussian(x_val, K, l, N, 1)
            plt.semilogy(x_val, ag_distribution, 'o', lw=3,
                         label=f'AG - N = {N} - K = {K} - l = {l}')

            aa_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_algebraic(x_val, K, L, l, N_aa, 1)
            plt.semilogy(x_val, aa_distribution, 'o', lw=3,
                         label=f'AA - N = {N_aa} - K = {K} - L = {L}'
                         + f' - l = {l}')

        else:
            N = 7
            N_gg = 6
            K = 461
            L = 280
            l = 280

            gg_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_gaussian(x_val, N_gg, 1)
            plt.semilogy(x_val, gg_distribution, 'o', lw=3,
                         label=f'GG - N = {N_gg}')

            ga_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_gaussian_algebraic(x_val, K, L, N, 1)
            plt.semilogy(x_val, ga_distribution, 'o', lw=3,
                         label=f'GA - N = {N} - K = {K} - L = {L}')

            ag_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_gaussian(x_val, K, l, N, 1)
            plt.semilogy(x_val, ag_distribution, 'o', lw=3,
                         label=f'AG - N = {N} - K = {K} - l = {l}')

            aa_distribution: np.ndarray = exact_distributions_covariance_tools\
                .pdf_algebraic_algebraic(x_val, K, L, l, N, 1)
            plt.semilogy(x_val, aa_distribution, 'o', lw=3,
                         label=f'AA - N = {N} - K = {K} - L = {L} - l = {l}')

        plt.legend(fontsize=20)
        plt.title(f'Aggregated distribution returns from {dates[0]} to'
                  + f' {dates[1]} - {time_step}', fontsize=30)
        plt.xlabel('Aggregated returns', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-10, 10)
        plt.ylim(10 ** -9, 1)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        # Plotting
        exact_distributions_covariance_tools \
            .save_plot(figure_log, function_name, dates, time_step)

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

    dates_1 = ['1990-01-01', '2020-12-31']
    # dates_1 = ['2021-07-19', '2021-08-14']
    N_values = [2, 3, 4]

    # pdf_gg_plot(dates_1, '1d', N_values)
    # pdf_loglog_gg_plot(dates_1, '1d', N_values)

    pdf_ga_plot(dates_1, '1d', N_values)

    # pdf_ag_plot(dates_1, '1d')

    # pdf_aa_plot(dates_1, '1d')

    # pdf_lin_all_distributions_plot(dates_1, '1d')

    # pdf_log_all_distributions_plot(dates_1, '1d')

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
