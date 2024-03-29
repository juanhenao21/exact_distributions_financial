"""Local normalization plot module.

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
    * epochs_tools

The module contains the following functions:
    * epochs_volatility_plot - plots the local normalized volatility of five
      stocks.
    * epochs_volatility_one_stock_plot - plots the local normalized volatility
      of one stocks.
    * epochs_normalized_returns_plot - plots the local normalized returns of
      five stocks.
    * epochs_normalized_returns_distribution_plot - plots the normalized
      returns distribution of five stocks.
    * epochs_correlation_matrix_plot - plots the local normalized correlation
      matrix.
    * epochs_aggregated_dist_returns_market_plot - plots the aggregated
      distribution of returns for a market.
    * epochs_log_log_agg_dist_returns_market_plot - plots the aggregated
      distribution of returns for a market in a log-log figure for different l.
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
"""

# -----------------------------------------------------------------------------
# Modules

import gc
import pickle
from typing import List

from matplotlib import legend, pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

import epochs_tools

# -----------------------------------------------------------------------------


def epochs_rolling_volatility_plot(
    dates: List[str], time_step: str, window: str
) -> None:
    """Plots the normalized rolling volatility.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_rolling_volatility_plot.__name__
    epochs_tools.function_header_print_plot(function_name, dates, time_step, window, "")

    try:

        # Load data
        volatility_data: pd.DataFrame = pickle.load(
            open(
                f"../data/epochs/epochs_volatility_rolling_data_{dates[0]}_{dates[1]}_step"
                + f"_{time_step}_win_{window}_K_.pickle",
                "rb",
            )
        )

        avg_volatility = volatility_data.mean(axis=1)

        plot_volatility = avg_volatility.plot(figsize=(16, 9))
        figure_volatility = plot_volatility.get_figure()

        plt.title(f"Avg Volatility Epochs - Epoch {window}", fontsize=30)
        plt.xlabel("Date", fontsize=25)
        plt.ylabel("Volatility", fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(True)
        plt.tight_layout()

        # Plotting
        epochs_tools.save_plot(
            figure_volatility, function_name, dates, time_step, window, ""
        )

        plt.close()
        del volatility_data
        del figure_volatility
        del plot_volatility
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_rolling_avg_correlation_matrix(
    dates: List[str], time_step: str, window: str
) -> None:
    """Plots the rolling average correlation matrix.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_rolling_avg_correlation_matrix.__name__
    epochs_tools.function_header_print_plot(function_name, dates, time_step, window, "")

    try:

        # Load data
        avg_corr_matrix_data: pd.Series = pickle.load(
            open(
                f"../data/epochs/epochs_rolling_avg_correlation_matrix_data_{dates[0]}_{dates[1]}_step"
                + f"_{time_step}_win_{window}_K_.pickle",
                "rb",
            )
        )
        avg_corr_matrix_data = avg_corr_matrix_data.rename(f"epochs {window}")
        print(avg_corr_matrix_data)

        plot_avg_corr_matrix = avg_corr_matrix_data.plot(figsize=(16, 9), legend=True)
        figure_avg_corr_matrix = plot_avg_corr_matrix.get_figure()

        plt.legend()
        # plt.title(f"Avg Correlation Matrix Epochs - Epoch {window}", fontsize=30)
        plt.xlabel("Date", fontsize=25)
        plt.ylabel("Avg. Correlation", fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.ylim(0, 0.5)
        plt.grid(True)
        plt.tight_layout()

        # Plotting
        epochs_tools.save_plot(
            figure_avg_corr_matrix, function_name, dates, time_step, window, ""
        )

        # plt.close()
        del avg_corr_matrix_data
        del figure_avg_corr_matrix
        del plot_avg_corr_matrix
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_volatility_one_stock_plot(
    dates: List[str], time_step: str, window: str
) -> None:
    """plots the local normalized volatility of one stocks.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_volatility_one_stock_plot.__name__
    epochs_tools.function_header_print_plot(function_name, dates, time_step, window)

    try:

        # Load data
        volatility_data: pd.DataFrame = pickle.load(
            open(
                f"../data/epochs/epochs_volatility_data_{dates[0]}"
                + f"_{dates[1]}_step_{time_step}_win_{window}.pickle",
                "rb",
            )
        )

        figure_vol: plt.Figure = plt.figure()

        plot_vol: np.ndarray = volatility_data.plot(figsize=(16, 9), grid=True)

        plt.legend(loc=1, fontsize=20)
        plt.xlabel(f"Date - {time_step} - window {window}", fontsize=20)
        plt.ylabel(f"Volatility", fontsize=20)
        plt.grid(True)
        plt.tight_layout()

        # Plotting
        epochs_tools.save_plot(figure_vol, function_name, dates, time_step, window)

        plt.close()
        del volatility_data
        del figure_vol
        del plot_vol
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_normalized_returns_plot(
    dates: List[str], time_step: str, window: str
) -> None:
    """Plots the local normalized returns of five stocks.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_normalized_returns_plot.__name__
    epochs_tools.function_header_print_plot(function_name, dates, time_step, window)

    try:

        # Load data
        norm_returns_data: pd.DataFrame = pickle.load(
            open(
                f"../data/epochs/epochs_normalized_returns_data_{dates[0]}"
                + f"_{dates[1]}_step_{time_step}_win_{window}.pickle",
                "rb",
            )
        ).iloc[:, :5]

        plot_ret: np.ndarray = norm_returns_data.plot(
            subplots=True, sharex=True, figsize=(16, 16), grid=True, sort_columns=True
        )

        _ = [ax.set_ylabel("Norm. Returns", fontsize=20) for ax in plot_ret]
        _ = [plot.legend(loc=1, fontsize=20) for plot in plt.gcf().axes]
        plt.xlabel(f"Date - {time_step} - time window {window}", fontsize=20)
        plt.tight_layout(pad=0.5)
        figure_ret: plt.Figure = plot_ret[0].get_figure()

        # Plotting
        epochs_tools.save_plot(figure_ret, function_name, dates, time_step, window)

        plt.close()
        del norm_returns_data
        del figure_ret
        del plot_ret
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_normalized_returns_distribution_plot(
    dates: List[str], time_step: str, window: str
) -> None:
    """Plots the normalized returns distribution of five stocks.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_normalized_returns_distribution_plot.__name__
    epochs_tools.function_header_print_plot(
        function_name + "lin", dates, time_step, window
    )

    try:

        # Load data
        norm_returns_data: pd.DataFrame = pickle.load(
            open(
                f"../data/epochs/epochs_normalized_returns_data_{dates[0]}"
                + f"_{dates[1]}_step_{time_step}_win_{window}.pickle",
                "rb",
            )
        ).iloc[:, :5]

        x_gauss: np.ndarray = np.arange(-6, 6, 0.001)
        gaussian: np.ndarray = epochs_tools.gaussian_distribution(0, 1, x_gauss)

        # Linear plot
        plot_lin = norm_returns_data.plot(kind="density", figsize=(16, 9))

        plt.plot(x_gauss, gaussian, lw=5, label="Gaussian")
        plt.title(
            f"Local normalized returns distribution {dates[0]} to" + f" {dates[1]}",
            fontsize=30,
        )
        plt.legend(loc=1, fontsize=20)
        plt.xlabel(f"Returns - {time_step} - window {window}", fontsize=25)
        plt.ylabel("Counts", fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-5, 5)
        plt.grid(True)
        plt.tight_layout()
        figure_lin: plt.Figure = plot_lin.get_figure()

        # Plotting
        epochs_tools.save_plot(
            figure_lin, function_name + "_lin", dates, time_step, window
        )

        plt.close()
        del figure_lin
        del plot_lin

        # Log plot
        plot_log = norm_returns_data.plot(kind="density", figsize=(16, 9), logy=True)

        plt.semilogy(x_gauss, gaussian, lw=5, label="Gaussian")
        plt.title(
            f"Local normalized returns distribution {dates[0]} to" + f" {dates[1]}",
            fontsize=30,
        )
        plt.legend(loc=1, fontsize=20)
        plt.xlabel(f"Returns - {time_step} - window {window}", fontsize=25)
        plt.ylabel("Counts", fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-5, 5)
        plt.ylim(10 ** -6, 10)
        plt.grid(True)
        plt.tight_layout()
        figure_log: plt.Figure = plot_log.get_figure()

        # Plotting
        epochs_tools.save_plot(
            figure_log, function_name + "_log", dates, time_step, window
        )

        plt.close()
        del norm_returns_data
        del figure_log
        del plot_log
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# ----------------------------------------------------------------------------


def epochs_correlation_matrix_plot(
    dates: List[str], time_step: str, window: str
) -> None:
    """Plots the local normalized correlation matrix.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_correlation_matrix_plot.__name__
    epochs_tools.function_header_print_plot(function_name, dates, time_step, window)

    try:

        figure: plt.figure = plt.figure(figsize=(16, 9))

        # Load data
        correlations: pd.DataFrame = pickle.load(
            open(
                f"../data/epochs/epochs_correlation_matrix_data_{dates[0]}"
                + f"_{dates[1]}_step_{time_step}_win_{window}.pickle",
                "rb",
            )
        )

        sns.heatmap(correlations, cmap="Blues")  # , vmin=-1, vmax=1)

        plt.title(
            f"Local norm. corr. matrix {dates[0]} to"
            + f" {dates[1]} - {time_step} - window {window}",
            fontsize=30,
        )
        plt.yticks(rotation=45)
        plt.xticks(rotation=45)

        figure.tight_layout()

        # Plotting
        epochs_tools.save_plot(figure, function_name, dates, time_step, window)

        plt.close()
        del correlations
        del figure
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_var_win_all_empirical_dist_returns_market_plot() -> None:
    """Plots all the local normalized aggregated distributions of returns for a
       market in different time steps.

    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_var_win_all_empirical_dist_returns_market_plot.__name__
    epochs_tools.function_header_print_plot(function_name, ["", ""], "", "", "")

    try:

        dates_1m = ["2021-07-19", "2021-08-14"]
        # dates_1h = ['2021-06-01', '2021-07-31']
        dates = ["1990-01-01", "2020-12-31"]

        windows = ["10", "25", "40", "55"]
        K_values = ["20", "50"]
        dates_vals = [dates_1m, dates, dates, dates]
        time_steps = ["1m", "1d", "1wk", "1mo"]

        for K_value in K_values:
            for idx, date_val in enumerate(dates_vals):

                figure_log: plt.Figure = plt.figure(figsize=(16, 9))

                x_values: np.ndarray = np.arange(-10, 10, 0.1)
                gaussian: np.ndarray = epochs_tools.gaussian_distribution(
                    0, 1, x_values
                )

                # Log plot
                plt.semilogy(x_values, gaussian, "-", lw=10, label="Gaussian")

                for window in windows:

                    # Load data
                    agg_returns: pd.Series = pickle.load(
                        open(
                            "../data/epochs/epochs_aggregated_dist_returns_market"
                            + f"_data_{date_val[0]}_{date_val[1]}"
                            + f"_step_{time_steps[idx]}_win_{window}_K_{K_value}"
                            + ".pickle",
                            "rb",
                        )
                    )[::10]

                    agg_returns = agg_returns.rename(f"Epoch {window}")

                    plot_log = agg_returns.plot(
                        kind="density",
                        style="-",
                        logy=True,
                        figsize=(16, 9),
                        legend=True,
                        lw=2,
                    )

                    figure_log = plot_log.get_figure()

                plt.legend(fontsize=20)
                plt.title(
                    f"Rotated Epochs - Time step = {time_steps[idx]}", fontsize=30
                )
                plt.xlabel(f"Aggregated returns - K = {K_value}", fontsize=25)
                plt.ylabel("PDF", fontsize=25)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlim(-6, 6)
                plt.ylim(10 ** -5, 1)
                plt.grid(True)
                plt.tight_layout()

                # Plotting
                epochs_tools.save_plot(
                    figure_log,
                    function_name + "_log_window",
                    date_val,
                    time_steps[idx],
                    "var",
                    K_value,
                )

        plt.close()
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_var_K_all_empirical_dist_returns_market_plot() -> None:
    """Plots all the local normalized aggregated distributions of returns for a
       market in different time steps.

    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_var_K_all_empirical_dist_returns_market_plot.__name__
    epochs_tools.function_header_print_plot(function_name, ["", ""], "", "", "")

    try:

        dates_1m = ["2021-07-19", "2021-08-14"]
        # dates_1h = ['2021-06-01', '2021-07-31']
        dates = ["1990-01-01", "2020-12-31"]

        windows = ["10", "25", "40", "55"]
        K_values = ["20", "50"]
        dates_vals = [dates_1m, dates, dates, dates]
        time_steps = ["1m", "1d", "1wk", "1mo"]

        for window in windows:
            for idx, date_val in enumerate(dates_vals):

                figure_log: plt.Figure = plt.figure(figsize=(16, 9))

                x_values: np.ndarray = np.arange(-10, 10, 0.1)
                gaussian: np.ndarray = epochs_tools.gaussian_distribution(
                    0, 1, x_values
                )

                # Log plot
                plt.semilogy(x_values, gaussian, "-", lw=10, label="Gaussian")

                for K_value in K_values:

                    # Load data
                    agg_returns: pd.Series = pickle.load(
                        open(
                            "../data/epochs/epochs_aggregated_dist_returns_market"
                            + f"_data_{date_val[0]}_{date_val[1]}"
                            + f"_step_{time_steps[idx]}_win_{window}_K_{K_value}"
                            + ".pickle",
                            "rb",
                        )
                    )[::10]

                    agg_returns = agg_returns.rename(f"K {K_value}")

                    plot_log = agg_returns.plot(
                        kind="density",
                        style="-",
                        logy=True,
                        figsize=(16, 9),
                        legend=True,
                        lw=2,
                    )

                    figure_log = plot_log.get_figure()

                plt.legend(fontsize=20)
                plt.title(
                    f"Rotated Epochs - Time step = {time_steps[idx]}", fontsize=30
                )
                plt.xlabel(f"Aggregated returns - Win = {window}", fontsize=25)
                plt.ylabel("PDF", fontsize=25)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlim(-6, 6)
                plt.ylim(10 ** -5, 1)
                plt.grid(True)
                plt.tight_layout()

                # Plotting
                epochs_tools.save_plot(
                    figure_log,
                    function_name + "_log_K",
                    date_val,
                    time_steps[idx],
                    window,
                    "var",
                )

        plt.close()
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_var_time_step_all_empirical_dist_returns_market_plot() -> None:
    """Plots all the local normalized aggregated distributions of returns for a
       market in different time steps.

    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = (
        epochs_var_time_step_all_empirical_dist_returns_market_plot.__name__
    )
    epochs_tools.function_header_print_plot(function_name, ["", ""], "", "", "")

    try:

        dates_1m = ["2021-07-19", "2021-08-14"]
        # dates_1h = ['2021-06-01', '2021-07-31']
        dates = ["1990-01-01", "2020-12-31"]

        windows = ["10", "25", "40", "55"]
        K_values = ["20", "50"]
        dates_vals = [dates_1m, dates, dates, dates]
        time_steps = ["1m", "1d", "1wk", "1mo"]

        for window in windows:
            for K_value in K_values:

                figure_log: plt.Figure = plt.figure(figsize=(16, 9))

                x_values: np.ndarray = np.arange(-10, 10, 0.1)
                gaussian: np.ndarray = epochs_tools.gaussian_distribution(
                    0, 1, x_values
                )

                # Log plot
                plt.semilogy(x_values, gaussian, "-", lw=10, label="Gaussian")

                for idx, date_val in enumerate(dates_vals):

                    # Load data
                    agg_returns: pd.Series = pickle.load(
                        open(
                            "../data/epochs/epochs_aggregated_dist_returns_market"
                            + f"_data_{date_val[0]}_{date_val[1]}"
                            + f"_step_{time_steps[idx]}_win_{window}_K_{K_value}"
                            + ".pickle",
                            "rb",
                        )
                    )[::10]

                    agg_returns = agg_returns.rename(f"{time_steps[idx]}")

                    plot_log = agg_returns.plot(
                        kind="density",
                        style="-",
                        logy=True,
                        figsize=(16, 9),
                        legend=True,
                        lw=2,
                    )

                    figure_log = plot_log.get_figure()

                plt.legend(fontsize=20)
                plt.title(f"Rotated Epochs - K = {K_value}", fontsize=30)
                plt.xlabel(f"Aggregated returns - Win = {window}", fontsize=25)
                plt.ylabel("PDF", fontsize=25)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlim(-6, 6)
                plt.ylim(10 ** -5, 1)
                plt.grid(True)
                plt.tight_layout()

                # Plotting
                epochs_tools.save_plot(
                    figure_log,
                    function_name + "_log_time_step",
                    date_val,
                    "var",
                    window,
                    K_value,
                )

        plt.close()
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_aggregated_dist_returns_market_plot(
    dates: List[str], time_step: str, window: str, K_value: str, l_value: str
) -> None:
    """Plots the aggregated distribution of returns for a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :param l_value: value of the shape parameter l (i.e. '2').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_aggregated_dist_returns_market_plot.__name__
    epochs_tools.function_header_print_plot(
        function_name, dates, time_step, window, K_value
    )

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(
            open(
                "../data/epochs/epochs_aggregated_dist_returns_market_data"
                + f"_{dates[0]}_{dates[1]}_step_{time_step}_win_{window}_K"
                + f"_{K_value}.pickle",
                "rb",
            )
        )[::10]

        agg_returns_data = agg_returns_data.rename("Agg. returns")

        x_values: np.ndarray = np.arange(-10, 10, 0.1)
        gaussian: np.ndarray = epochs_tools.gaussian_distribution(0, 1, x_values)
        algebraic: np.ndarray = epochs_tools.algebraic_distribution(
            int(K_value), int(l_value), x_values
        )

        figure_log: plt.Figure = plt.figure(figsize=(16, 9))

        # Log plot
        plot_log = agg_returns_data.plot(
            kind="density", style="-", logy=True, figsize=(16, 9), legend=True, lw=5
        )

        plt.semilogy(x_values, gaussian, "-", lw=3, label="Gaussian")
        plt.semilogy(
            x_values,
            algebraic,
            "-",
            lw=3,
            label=f"A - K = {K_value} - l = {l_value} -"
            + f" m = {2 * int(l_value) - int(K_value) - 2}",
        )

        plt.legend(fontsize=20)
        plt.title(f"Epochs from {dates[0]} to {dates[1]} - {time_step}", fontsize=30)
        plt.xlabel(f"Aggregated returns - window {window}", fontsize=25)
        plt.ylabel("PDF", fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-6, 6)
        plt.ylim(10 ** -5, 1)
        plt.grid(True)
        plt.tight_layout()
        figure_log = plot_log.get_figure()

        # Plotting
        epochs_tools.save_plot(
            figure_log, function_name + "_log", dates, time_step, window, K_value
        )

        plt.close()
        del agg_returns_data
        del figure_log
        del plot_log
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_log_log_agg_dist_returns_market_plot(
    dates: List[str], time_step: str, window: str, K_value: str, l_values: List[int]
) -> None:
    """Plots the aggregated distribution of returns for a market in a log-log
       figure for diferent l values.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :param l_values: List of the values of the shape parameter l
     (i.e. [2, 4, 6, 8]).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_log_log_agg_dist_returns_market_plot.__name__
    epochs_tools.function_header_print_plot(
        function_name, dates, time_step, window, K_value
    )

    try:

        # Load data
        agg_returns_data: pd.Series = pickle.load(
            open(
                "../data/epochs/epochs_aggregated_dist_returns_market_data"
                + f"_{dates[0]}_{dates[1]}_step_{time_step}_win_{window}_K"
                + f"_{K_value}.pickle",
                "rb",
            )
        )[::10]

        agg_returns_data = agg_returns_data.rename("Agg. returns")

        x_values: np.ndarray = np.arange(-10, 10, 0.1)
        gaussian: np.ndarray = epochs_tools.gaussian_distribution(0, 1, x_values)

        figure_log: plt.Figure = plt.figure(figsize=(16, 9))

        # Log plot
        for l_value in l_values:
            m_value = 2 * l_value - int(K_value) - 2
            algebraic: np.ndarray = epochs_tools.algebraic_distribution(
                int(K_value), l_value, x_values
            )
            plt.loglog(
                x_values,
                algebraic,
                "-",
                lw=1,
                label=f"A - K={K_value} - l={l_value} - m={m_value}",
            )

        plt.loglog(x_values, gaussian, "-", lw=10, label="Gaussian")
        plot_log = agg_returns_data.plot(
            kind="density", style="-", loglog=True, figsize=(16, 9), legend=True, lw=5
        )

        plt.legend(fontsize=20)
        plt.title(f"Epochs from {dates[0]} to {dates[1]} - {time_step}", fontsize=30)
        plt.xlabel(f"Aggregated returns - window {window}", fontsize=25)
        plt.ylabel("PDF", fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(3, 5.5)
        plt.ylim(10 ** -4, 10 ** -2)
        plt.grid(True)
        plt.tight_layout()
        figure_log = plot_log.get_figure()

        # Plotting
        epochs_tools.save_plot(
            figure_log, function_name + "_loglog", dates, time_step, window, K_value
        )

        plt.close()
        del agg_returns_data
        del figure_log
        del plot_log
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def epochs_log_log_all_empirical_dist_returns_market_plot() -> None:
    """Plots all the local normalized aggregated distributions of returns for a
       market in different time steps.

    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_log_log_all_empirical_dist_returns_market_plot.__name__
    epochs_tools.function_header_print_plot(function_name, ["", ""], "", "")

    try:

        dates_1m = ["2021-07-19", "2021-08-07"]
        dates_1h = ["2021-06-01", "2021-07-31"]
        dates = ["1990-01-01", "2020-12-31"]

        window = "25"

        # Load data
        agg_returns_min: pd.Series = pickle.load(
            open(
                "../data/epochs/epochs_aggregated_dist_returns_market_data"
                + f"_{dates_1m[0]}_{dates_1m[1]}_step_1m_win_{window}.pickle",
                "rb",
            )
        )[::5]
        # agg_returns_hour: pd.Series = pickle.load(open(
        #     '../data/epochs/epochs_aggregated_dist_returns_market_data'
        #     + f'_{dates_1h[0]}_{dates_1h[1]}_step_1h_win_{window}.pickle',
        #     'rb'))[::5]
        agg_returns_day: pd.Series = pickle.load(
            open(
                "../data/epochs/epochs_aggregated_dist_returns_market_data"
                + f"_{dates[0]}_{dates[1]}_step_1d_win_{window}.pickle",
                "rb",
            )
        )[::5]
        agg_returns_week: pd.Series = pickle.load(
            open(
                "../data/epochs/epochs_aggregated_dist_returns_market_data"
                + f"_{dates[0]}_{dates[1]}_step_1wk_win_{window}.pickle",
                "rb",
            )
        )[::5]
        agg_returns_month: pd.Series = pickle.load(
            open(
                "../data/epochs/epochs_aggregated_dist_returns_market_data"
                + f"_{dates[0]}_{dates[1]}_step_1mo_win_{window}.pickle",
                "rb",
            )
        )[::5]

        agg_returns_min = agg_returns_min.rename("Minute")
        agg_returns_hour = agg_returns_hour.rename("Hour")
        agg_returns_day = agg_returns_day.rename("Day")
        agg_returns_week = agg_returns_week.rename("Week")
        agg_returns_month = agg_returns_month.rename("Month")

        x_values: np.ndarray = np.arange(-10, 10, 0.1)
        gaussian: np.ndarray = epochs_tools.gaussian_distribution(0, 1, x_values)

        figure_log: plt.Figure = plt.figure(figsize=(16, 9))

        # Log plot
        plt.loglog(x_values, gaussian, "-", lw=10, label="Gaussian")

        plot_log_m = agg_returns_min.plot(
            kind="density", style="-", loglog=True, figsize=(16, 9), legend=True, lw=2
        )
        # plot_log_h = agg_returns_hour.plot(kind='density', style='-',
        #                                    loglog=True, figsize=(16, 9),
        #                                    legend=True, lw=2)
        plot_log_d = agg_returns_day.plot(
            kind="density", style="-", loglog=True, figsize=(16, 9), legend=True, lw=2
        )
        plot_log_wk = agg_returns_week.plot(
            kind="density", style="-", loglog=True, figsize=(16, 9), legend=True, lw=2
        )
        plot_log_mo = agg_returns_month.plot(
            kind="density", style="-", loglog=True, figsize=(16, 9), legend=True, lw=2
        )

        plt.legend(fontsize=20)
        plt.title(f"Rotated Epochs", fontsize=30)
        plt.xlabel(f"Aggregated returns - window {window}", fontsize=25)
        plt.ylabel("PDF", fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(2.5, 5)
        plt.ylim(10 ** -5, 10 ** -1)
        plt.grid(True)
        plt.tight_layout()
        figure_log = plot_log_m.get_figure()
        # figure_log = plot_log_h.get_figure()
        figure_log = plot_log_d.get_figure()
        figure_log = plot_log_wk.get_figure()
        figure_log = plot_log_mo.get_figure()

        # Plotting
        epochs_tools.save_plot(
            figure_log, function_name + "_loglog", ["", ""], "", window
        )

        plt.close()
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

    # dates_1m = ["2021-07-19", "2021-08-07"]
    # dates_1h = ["2021-06-01", "2021-07-31"]
    # dates = ["1990-01-01", "2020-12-31"]

    # win = "25"

    l_values = np.arange(28, 31).astype(int)
    # epochs_log_log_agg_dist_returns_market_plot(['1990-01-01', '2020-12-31'],
    #                                             '1d', '25', '1', l_values)
    # epochs_log_log_agg_dist_returns_market_plot(['1990-01-01', '2020-12-31'],
    #                                             '1d', '55', '1', l_values)

    # l_values = np.linspace(27, 45, 5).astype(int)
    # epochs_log_log_agg_dist_returns_market_plot(['1990-01-01', '2020-12-31'],
    #                                             '1d', '25', '50', l_values)
    # epochs_log_log_agg_dist_returns_market_plot(['1990-01-01', '2020-12-31'],
    #                                             '1d', '55', '50', l_values)

    # epochs_var_win_all_empirical_dist_returns_market_plot()
    # epochs_var_K_all_empirical_dist_returns_market_plot()
    # epochs_var_time_step_all_empirical_dist_returns_market_plot()

    # epochs_aggregated_dist_returns_market_plot(dates, "1d", "55", "50", "30")

    windows = ["10", "25", "40", "55"]
    dates: List[str] = ["1990-01-01", "2020-12-31"]

    for window in windows:
        epochs_rolling_avg_correlation_matrix(dates, "1d", window)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
