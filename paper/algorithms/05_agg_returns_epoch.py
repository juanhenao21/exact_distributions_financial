"""Aggegated returns epochs implementation.

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
    * epochs_gaussian_agg_dist_returns_market_plot - plots the aggregated distribution of
      returns for a market. Generates Fig. 2 of the paper.
    * epochs_algebraic_agg_dist_returns_market_plot - plots the aggregated distribution of
      returns for a market. Generates Fig. 3 of the paper.
    * epochs_var_win_all_empirical_dist_returns_market_plot - plots the aggregated
      distribution of returns for a market for different epochs window lenghts. Generates
      Fig. 4 of the paper.
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
"""

# ----------------------------------------------------------------------------
# Modules

import gc
import pickle
import sys
from typing import List

from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(1, "../../project/epochs")
import epochs_tools  # type: ignore

# ----------------------------------------------------------------------------


def epochs_gaussian_agg_dist_returns_market_plot(
    dates: List[List[str]], time_steps: List[str], window: str, K_value: str
) -> None:
    """Plots the aggregated distribution of returns for a market.

    The function loads the data from the aggregated returns and plot them along with a
    Gaussian distribution in a semilogy plot. The loaded data is obtained normalizing the
    epochs before the rotation and scaling. This plot is used as Fig 2. in the paper.

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

        markers: List[str] = ["o", "^", "s", "P", "x"]

        for time_step, date, marker in zip(time_steps, dates, markers):
            # Load data
            agg: pd.Series = pickle.load(
                open(
                    "../../project/data/epochs/epochs_aggregated_dist_returns"
                    + f"_market_data_short_{date[0]}_{date[1]}_step_{time_step}"
                    + f"_win_{window}_K_{K_value}.pickle",
                    "rb",
                )
            )

            agg = agg.rename(f"Agg. returns {time_step}")

            # Log plot
            plot = agg.plot(
                kind="density", style=marker, logy=True, legend=False, ms=10
            )

        x_gauss: np.ndarray = np.arange(-10, 10, 0.3)
        gaussian: np.ndarray = epochs_tools.gaussian_distribution(0, 1, x_gauss)

        plt.semilogy(x_gauss, gaussian, "-", lw=10, label="Gaussian")

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=30)
        plt.xlabel(r"$\tilde{r}$", fontsize=40)
        plt.ylabel("PDF", fontsize=40)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlim(-6, 6)
        plt.ylim(10 ** -4, 1)
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        figure.savefig(f"../plot/05_gaussian_agg_returns_short_epoch.png")

        plt.close()
        del agg
        del figure
        del plot
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# ----------------------------------------------------------------------------


def epochs_algebraic_agg_dist_returns_market_plot(
    dates: List[List[str]],
    time_steps: List[str],
    window: str,
    K_value: str,
    l_values: List[int],
) -> None:
    """Plots the aggregated distribution of returns for a market.

    The function loads the data from the aggregated returns and plots two columns with a
    semilogy plot and a loglog  plot. They are plotted along wiht an algebraic
    distribution and a Gaussian distribution. The loaded data is obtained normalizing the
    epochs before the rotation and scaling. This plot is used as Fig 3. in the paper.

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

        figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

        markers: List[str] = ["o", "^", "s", "P", "x"]

        for time_step, date, marker in zip(time_steps, dates, markers):
            # Load data
            agg: pd.Series = pickle.load(
                open(
                    "../../project/data/epochs/epochs_aggregated_dist_returns"
                    + f"_market_data_short_{date[0]}_{date[1]}_step_{time_step}"
                    + f"_win_{window}_K_{K_value}.pickle",
                    "rb",
                )
            )

            agg = agg.rename(f"Agg. returns {time_step}")

            # Log plot
            plot_1 = agg.plot(
                kind="density", style=marker, logy=True, ax=ax1, legend=False, ms=7
            )
            plot_2 = agg.plot(
                kind="density", style=marker, loglog=True, ax=ax2, legend=False, ms=7
            )

        x_values: np.ndarray = np.arange(-10, 10, 0.3)

        if K_value == "all":
            K_value = "200"

        for l_value in l_values:
            algebraic: np.ndarray = epochs_tools.algebraic_distribution(
                int(K_value), l_value, x_values
            )
            ax1.semilogy(
                x_values, algebraic, "-", lw=5, label=f"Algebraic l = {l_value}"
            )
            ax2.loglog(x_values, algebraic, "-", lw=5, label=f"Algebraic l = {l_value}")

        gaussian: np.ndarray = epochs_tools.gaussian_distribution(0, 1, x_values)
        ax1.semilogy(x_values, gaussian, "-", lw=5, label=f"Gaussian")
        ax2.loglog(x_values, gaussian, "-", lw=5, label=f"Gaussian")

        ax1.set_xlabel(r"$\tilde{r}$", fontsize=20)
        ax1.set_ylabel("PDF", fontsize=20)
        ax1.tick_params(axis="both", labelsize=15)
        ax1.set_xlim(-6, 6)
        ax1.set_ylim(10 ** -4, 1)
        ax1.grid(True)

        ax2.legend(loc="upper center", bbox_to_anchor=(1.4, 0.6), ncol=1, fontsize=20)
        ax2.set_xlabel(r"$\tilde{r}$", fontsize=20)
        ax2.set_ylabel("PDF", fontsize=20)
        ax2.tick_params(axis="both", which="both", labelsize=15)
        ax2.set_xlim(3, 5)
        ax2.set_ylim(10 ** -4, 10 ** -2)
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        figure.savefig(f"../plot/05_algebraic_agg_returns_short_epoch.png")

        plt.close()
        del agg
        del figure
        del plot_1
        del plot_2
        gc.collect()

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# ----------------------------------------------------------------------------


def epochs_var_win_all_empirical_dist_returns_market_plot(
    dates: List[List[str]],
    time_steps: List[str],
    windows: List[str],
    K_values: List[str],
) -> None:
    """Plots the aggregated distribution of returns for a market for different epochs
       window lenghts.Plots the aggregated distribution of returns for a market for
       different epochs window lenghts.

    The function loads the data from the aggregated returns and plots four rows, each with
    the different epochs window lenghts and number of companies. They are plotted along
    with a Gaussian distribution in a semilogy plot. The loaded data is obtained
    normalizing the epochs before the rotation and scaling. This plot is used as Fig 4.
    in the paper.

    :param dates: list of lists with the interval of dates to be analyzed
     (i.e. [['1992-01', '2012-12'], ['2012-01', '2020-12']).
    :param time_steps: list of the time step of the data (i.e. ['1m', '1h']).
    :param windows: list of window times to compute the volatility (i.e. '60', ...).
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    function_name: str = epochs_var_win_all_empirical_dist_returns_market_plot.__name__
    epochs_tools.function_header_print_plot(function_name, ["", ""], "", "", "")

    try:

        figure, axs = plt.subplots(
            4,
            3,
            figsize=(16, 9),
            sharex="col",
            sharey="row",
            gridspec_kw={"hspace": 0, "wspace": 0},
        )

        markers: List[str] = ["go", "r^", "ms", "P", "x"]

        for idx, date_val in enumerate(dates):

            for ax, win in zip(axs, windows):

                for idx_ax, (K_value, marker) in enumerate(zip(K_values, markers)):

                    # Load data
                    agg: pd.Series = pickle.load(
                        open(
                            "../../project/data/epochs/epochs_aggregated_dist"
                            + f"_returns_market_data_short_{date_val[0]}"
                            + f"_{date_val[1]}_step_{time_steps[idx]}_win_{win}"
                            + f"_K_{K_value}.pickle",
                            "rb",
                        )
                    )

                    agg = agg.rename(f"K = {K_value}")

                    plot = agg.plot(
                        kind="density",
                        style=marker,
                        marker=markers[idx_ax],
                        ax=ax[idx_ax],
                        logy=True,
                        figsize=(16, 9),
                        ms=5,
                    )

                    figure = plot.get_figure()

                    x_values: np.ndarray = np.arange(-10, 10, 0.1)
                    gaussian: np.ndarray = epochs_tools.gaussian_distribution(
                        0, 1, x_values
                    )

                    # Log plot
                    ax[idx_ax].semilogy(x_values, gaussian, "-", lw=5, label="Gaussian")
                    ax[idx_ax].semilogy(
                        x_values, gaussian, markers[idx_ax], ms=5, label="Gaussian"
                    )

        for axis in axs:
            for ax in axis:
                ax.set
                ax.set_xlabel(r"$\tilde{r}$", fontsize=25)
                ax.tick_params(axis="x", labelsize=20)
                ax.tick_params(axis="y", labelsize=20)
                ax.set_xlim(-5, 5)
                ax.set_ylim(10 ** -4, 1)
                ax.grid(True)

        axs[3][0].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=15
        )
        axs[3][1].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=15
        )
        axs[3][2].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=15
        )
        axs[0][0].set_ylabel("PDF", fontsize=25)
        axs[0][0].set_yticks(axs[0][0].get_yticks()[2:-1])
        axs[1][0].set_ylabel("PDF", fontsize=25)
        axs[1][0].set_yticks(axs[1][0].get_yticks()[2:-1])
        axs[2][0].set_ylabel("PDF", fontsize=25)
        axs[2][0].set_yticks(axs[2][0].get_yticks()[2:-1])
        axs[3][0].set_ylabel("PDF", fontsize=25)
        axs[3][0].set_yticks(axs[3][0].get_yticks()[2:-1])

        plt.tight_layout()

        # Plotting
        figure.savefig(f"../plot/05_window_comparison.png")

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

    dates: List[List[str]] = [["1990-01-01", "2020-12-31"]]
    time_steps: List[str] = ["1d"]

    # epochs_gaussian_agg_dist_returns_market_plot(dates, time_steps, '25',
    #                                              'all')
    # epochs_algebraic_agg_dist_returns_market_plot(dates, time_steps, "55", "all", [104])
    epochs_var_win_all_empirical_dist_returns_market_plot(
        dates, time_steps, ["10", "25", "40", "55"], ["20", "100", "all"]
    )


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
