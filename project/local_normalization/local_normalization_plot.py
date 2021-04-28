'''Local normalization plot module.

The functions in the module plot the data obtained in the
local_normalization_analysis module.

This script requires the following modules:
    * gc
    * pickle
    * typing
    * matplotlib
    * numpy
    * pandas
    * seaborn
    * local_normalization_tools

The module contains the following functions:
    * ln_volatility_plot - plots the local normalized volatility of five
      stocks.
    * ln_volatility_one_stock_plot - plots the local normalized volatility of
      one stocks.
    * ln_normalized_returns_plot - plots the local normalized returns of five
      stocks.
    * ln_normalized_returns_distribution_plot - plots the normalized returns
      distribution of five stocks.
    * ln_matrix_correlation_plot - plots the local normalized correlation
      matrix.
    * ln_aggregated_dist_returns_market_plot - plots the aggregated
      distribution of returns for a market.
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

import local_normalization_tools

# -----------------------------------------------------------------------------


def ln_volatility_plot(dates: List[str], time_step: str, window: str) -> None:
    """Plots the local normalized volatility of five stocks.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = ln_volatility_plot.__name__
    local_normalization_tools \
        .function_header_print_plot(function_name, dates, time_step, window)

    try:

        # Load data
        volatility_data: pd.DataFrame = pickle.load(open(
                        f'../data/local_normalization/ln_volatility_data'
                        + f'_{dates[0]}_{dates[1]}_step_{time_step}_win'
                        + f'_{window}.pickle', 'rb')).iloc[:, :5]

        plot_vol: np.ndarray = volatility_data \
            .plot(subplots=True, sharex=True, figsize=(16, 16), grid=True,
                  sort_columns=True)

        _ = [ax.set_ylabel('Volatility', fontsize=20) for ax in plot_vol]
        _ = [plot.legend(loc=1, fontsize=20) for plot in plt.gcf().axes]
        plt.xlabel(f'Date - {time_step} - time window {window}', fontsize=20)
        plt.tight_layout(pad=0.5)
        figure_vol: plt.Figure = plot_vol[0].get_figure()

        # Plotting
        local_normalization_tools \
            .save_plot(figure_vol, function_name, dates, time_step, window)

        plt.close()
        del volatility_data
        del figure_vol
        del plot_vol
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def ln_volatility_one_stock_plot(dates: List[str], time_step: str,
                                 window: str) -> None:
    """plots the local normalized volatility of one stocks.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = ln_volatility_one_stock_plot.__name__
    local_normalization_tools \
        .function_header_print_plot(function_name, dates, time_step, window)

    try:

        # Load data
        volatility_data: pd.DataFrame = pickle.load(open(
                        f'../data/local_normalization/ln_volatility_data'
                        + f'_{dates[0]}_{dates[1]}_step_{time_step}_win'
                        + f'_{window}.pickle', 'rb'))

        figure_vol: plt.Figure = plt.figure()

        plot_vol: np.ndarray = volatility_data.plot(figsize=(16, 9), grid=True)

        plt.legend(loc=1, fontsize=20)
        plt.xlabel(f'Date - {time_step} - window {window}', fontsize=20)
        plt.ylabel(f'Volatility', fontsize=20)
        plt.grid(True)
        plt.tight_layout()

        # Plotting
        local_normalization_tools \
            .save_plot(figure_vol, function_name, dates, time_step, window)

        plt.close()
        del volatility_data
        del figure_vol
        del plot_vol
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def ln_normalized_returns_plot(dates: List[str], time_step: str,
                               window: str) -> None:
    """Plots the local normalized returns of five stocks.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = ln_normalized_returns_plot.__name__
    local_normalization_tools \
        .function_header_print_plot(function_name, dates, time_step, window)

    try:

        # Load data
        norm_returns_data: pd.DataFrame = pickle.load(open(
            f'../data/local_normalization/ln_normalized_returns_data'
            + f'_{dates[0]}_{dates[1]}_step_{time_step}_win_{window}.pickle',
            'rb')).iloc[:, :5]

        plot_ret: np.ndarray = norm_returns_data \
            .plot(subplots=True, sharex=True, figsize=(16, 16),
                  grid=True, sort_columns=True)

        _ = [ax.set_ylabel('Norm. Returns', fontsize=20) for ax in plot_ret]
        _ = [plot.legend(loc=1, fontsize=20) for plot in plt.gcf().axes]
        plt.xlabel(f'Date - {time_step} - time window {window}', fontsize=20)
        plt.tight_layout(pad=0.5)
        figure_ret: plt.Figure = plot_ret[0].get_figure()

        # Plotting
        local_normalization_tools \
            .save_plot(figure_ret, function_name, dates, time_step, window)

        plt.close()
        del norm_returns_data
        del figure_ret
        del plot_ret
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()


# -----------------------------------------------------------------------------


def ln_normalized_returns_distribution_plot(dates: List[str], time_step: str,
                                            window: str) -> None:
    """Plots the normalized returns distribution of five stocks.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = ln_normalized_returns_distribution_plot.__name__
    local_normalization_tools \
        .function_header_print_plot(function_name + 'lin', dates, time_step,
                                    window)

    try:

        # Load data
        norm_returns_data: pd.DataFrame = pickle.load(open(
            f'../data/local_normalization/ln_normalized_returns_data'
            + f'_{dates[0]}_{dates[1]}_step_{time_step}_win_{window}.pickle',
            'rb')).iloc[:, :5]

        x_gauss: np.ndarray = np.arange(-6, 6, 0.001)
        gaussian: np.ndarray = local_normalization_tools \
            .gaussian_distribution(0, 1, x_gauss)

        # Linear plot
        plot_lin = norm_returns_data.plot(kind='density', figsize=(16, 9))

        plt.plot(x_gauss, gaussian, lw=5, label='Gaussian')
        plt.title(f'Local normalized returns distribution {dates[0]} to'
                  + f' {dates[1]}',
                  fontsize=30)
        plt.legend(loc=1, fontsize=20)
        plt.xlabel(f'Returns - {time_step} - window {window}', fontsize=25)
        plt.ylabel('Counts', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-5, 5)
        plt.grid(True)
        plt.tight_layout()
        figure_lin: plt.Figure = plot_lin.get_figure()

        # Plotting
        local_normalization_tools \
            .save_plot(figure_lin, function_name + '_lin', dates, time_step,
                       window)

        plt.close()
        del figure_lin
        del plot_lin

        # Log plot
        plot_log = norm_returns_data.plot(kind='density', figsize=(16, 9),
                                          logy=True)

        plt.semilogy(x_gauss, gaussian, lw=5, label='Gaussian')
        plt.title(f'Local normalized returns distribution {dates[0]} to'
                  + f' {dates[1]}',
                  fontsize=30)
        plt.legend(loc=1, fontsize=20)
        plt.xlabel(f'Returns - {time_step} - window {window}', fontsize=25)
        plt.ylabel('Counts', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-5, 5)
        plt.ylim(10 ** -6, 10)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        # Plotting
        local_normalization_tools \
            .save_plot(figure_log, function_name + '_log', dates, time_step,
                       window)

        plt.close()
        del norm_returns_data
        del figure_log
        del plot_log
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# ----------------------------------------------------------------------------


def ln_correlation_matrix_plot(dates: List[str], time_step: str,
                               window: str) -> None:
    """Plots the local normalized correlation matrix.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = ln_correlation_matrix_plot.__name__
    local_normalization_tools \
        .function_header_print_plot(function_name, dates, time_step, window)

    try:

        figure: plt.figure = plt.figure(figsize=(16, 9))

        # Load data
        correlations: pd.DataFrame = pickle.load(open(
            f'../data/local_normalization/ln_correlation_matrix_data'
            + f'_{dates[0]}_{dates[1]}_step_{time_step}_win_{window}.pickle',
            'rb'))

        sns.heatmap(correlations, cmap='Blues')  # , vmin=-1, vmax=1)

        plt.title(f'Local norm. corr. matrix {dates[0]} to'
                  + f' {dates[1]} - {time_step} - window {window}',
                  fontsize=30)
        plt.yticks(rotation=45)
        plt.xticks(rotation=45)

        figure.tight_layout()

        # Plotting
        local_normalization_tools \
            .save_plot(figure, function_name, dates, time_step, window)

        plt.close()
        del correlations
        del figure
        gc.collect()

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def ln_aggregated_dist_returns_market_plot(dates: List[str], time_step: str,
                                           window: str) -> None:
    """Plots the aggregated distribution of returns for a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :param window: window time to compute the volatility (i.e. '60', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = ln_aggregated_dist_returns_market_plot.__name__
    local_normalization_tools \
        .function_header_print_plot(function_name, dates, time_step, window)

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../data/local_normalization/ln_aggregated_dist_returns_market'
            + f'_data_{dates[0]}_{dates[1]}_step_{time_step}_win_{window}'
            + f'.pickle', 'rb'))#[::2]

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_gauss: np.ndarray = np.arange(-10, 10, 0.1)
        gaussian: np.ndarray = local_normalization_tools \
            .gaussian_distribution(0, 1, x_gauss)

        figure_log = plt.figure(figsize=(16, 9))

        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', logy=True,
                                         figsize=(16, 9), legend=True, lw=3)

        plt.semilogy(x_gauss, gaussian, 'o', lw=3, label='Gaussian')

        plt.legend(fontsize=20)
        plt.title(f'Local norm. dist. returns from {dates[0]} to'
                  + f' {dates[1]} - {time_step}', fontsize=30)
        plt.xlabel(f'Aggregated returns - window {window}', fontsize=25)
        plt.ylabel('PDF', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-6, 6)
        plt.ylim(10 ** -5, 1)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        # Plotting
        local_normalization_tools \
            .save_plot(figure_log, function_name + '_log', dates, time_step,
                       window)

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

    ln_aggregated_dist_returns_market_plot(['1992-01', '2012-12'], '1d', '25')

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
