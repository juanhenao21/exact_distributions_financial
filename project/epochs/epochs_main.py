'''Portfolio optimization local normalization main module.

The functions in the module compute the returns and correlation matrix of
financial time series.

This script requires the following modules:
    * typing
    * multiprocessing
    * itertools
    * local_normalization_analysis
    * local_normalization_plot
    * local_normalization_tools

The module contains the following functions:
    * data_plot_generator
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# -----------------------------------------------------------------------------
# Modules

from typing import List
import multiprocessing as mp
from itertools import product as iprod

import local_normalization_analysis
import local_normalization_plot
import local_normalization_tools

# -----------------------------------------------------------------------------


def data_plot_generator(dates: List[List[str]], time_steps: List[str],
                        windows: List[str]) -> None:
    """Generates all the analysis and plots from the data.

    :param dates: list of lists of the string of the dates to be analyzed
     (i.e. [['1980-01', '2020-12'], ['1980-01', '2020-12']).
    :param time_steps: list of the string of the time step of the data
     (i.e. ['1m', '2m', '5m']).
    :param windows: list of the string of the windows to compute the volatility
      of the data (i.e. ['60', '90']).
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    # Parallel computing
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     # Specific functions
    #     pool.starmap(local_normalization_analysis
    #                  .ln_volatility_data, iprod(dates, time_steps, windows))
    #     pool.starmap(local_normalization_analysis
    #                  .ln_normalized_returns_data, iprod(dates, time_steps,
    #                                                     windows))
    #     pool.starmap(local_normalization_analysis
    #                  .ln_correlation_matrix_data, iprod(dates, time_steps,
    #                                                     windows))

    #     # Plot
    #     pool.starmap(local_normalization_plot
    #                  .ln_volatility_plot, iprod(dates, time_steps, windows))
    #     pool.starmap(local_normalization_plot
    #                  .ln_normalized_returns_plot, iprod(dates, time_steps,
    #                                                     windows))
    #     pool.starmap(local_normalization_plot
    #                  .ln_normalized_returns_distribution_plot,
    #                  iprod(dates, time_steps, windows))
    #     pool.starmap(local_normalization_plot
    #                  .ln_correlation_matrix_plot, iprod(dates, time_steps,
    #                                                     windows))

    for date in dates:
        for time_step in time_steps:
            for window in windows:
                local_normalization_analysis \
                    .ln_aggregated_dist_returns_market_data(date, time_step,
                                                            window)
                local_normalization_plot \
                    .ln_aggregated_dist_returns_market_plot(date, time_step,
                                                            window)

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function extract, analyze and plot the data.

    :return: None.
    """

    local_normalization_tools.initial_message()

    # Initial year and time step
    dates: List[List[str]] = [['2021-07-19', '2021-07-23']]
    time_steps: List[str] = ['1m', '1h', '1d']
    windows: List[str] = ['25']

    # Basic folders
    local_normalization_tools.start_folders()

    # Run analysis
    # Analysis and plot
    data_plot_generator(dates, time_steps, windows)

    print('Ay vamos!!!')

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
