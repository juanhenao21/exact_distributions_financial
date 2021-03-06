'''Exact distributions covariance main module.

The functions in the module compute the returns and correlation matrix of
financial time series.

This script requires the following modules:
    * typing
    * multiprocessing
    * itertools
    * exact_distributions_covariance_analysis
    * exact_distributions_covariance_plot
    * exact_distributions_covariance_tools

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

import exact_distributions_covariance_analysis
import exact_distributions_covariance_plot
import exact_distributions_covariance_tools

# -----------------------------------------------------------------------------


def data_plot_generator(dates: List[List[str]], time_steps: List[str]) -> None:
    """Generates all the analysis and plots from the data.

    :param dates: list of lists of the string of the dates to be analyzed
     (i.e. [['1980-01', '2020-12'], ['1980-01', '2020-12']).
    :param time_steps: list of the string of the time step of the data
     (i.e. ['1m', '2m', '5m']).
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    # Parallel computing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Specific functions
        pool.starmap(exact_distributions_covariance_analysis
                     .returns_data, iprod(dates, time_steps))
        pool.starmap(exact_distributions_covariance_analysis
                     .aggregated_dist_returns_market_data,
                     iprod(dates, time_steps))

        # Plot
        pool.starmap(exact_distributions_covariance_plot
                     .returns_plot, iprod(dates, time_steps))
        pool.starmap(exact_distributions_covariance_plot
                     .aggregated_dist_returns_market_plot,
                     iprod(dates, time_steps))

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function extract, analyze and plot the data.

    :return: None.
    """

    exact_distributions_covariance_tools.initial_message()

    # Initial year and time step
    # dates: List[List[str]] = [['1990-01-01', '2020-12-31']]
    dates: List[List[str]] = [['2021-07-19', '2021-08-14']]
    time_steps: List[str] = ['1m']

    # Basic folders
    exact_distributions_covariance_tools.start_folders()

    # Run analysis
    # Analysis and plot
    data_plot_generator(dates, time_steps)

    print('Ay vamos!!!')

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
