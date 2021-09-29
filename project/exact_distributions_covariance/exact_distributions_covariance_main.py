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

from typing import Any, List, Tuple
import multiprocessing as mp

import exact_distributions_covariance_analysis
import exact_distributions_covariance_plot
import exact_distributions_covariance_tools

# -----------------------------------------------------------------------------


def data_plot_generator(cov_params: List[Tuple[Any]]) -> None:
    """Generates all the analysis and plots from the data.

    :param cov_params: list of tuples with the strings of the dates and time
     step to be analyzed
     (i.e. [(['1980-01', '2020-12'], '1m'), (['1980-01', '2020-12'], '1h').
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    # Parallel computing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Specific functions
        pool.starmap(exact_distributions_covariance_analysis.returns_data,
                     cov_params)
        pool.starmap(exact_distributions_covariance_analysis
                     .aggregated_dist_returns_market_data, cov_params)

        # Plot
        pool.starmap(exact_distributions_covariance_plot.returns_plot,
                     cov_params)
        pool.starmap(exact_distributions_covariance_plot
                     .aggregated_dist_returns_market_plot, cov_params)

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function extract, analyze and plot the data.

    :return: None.
    """

    exact_distributions_covariance_tools.initial_message()

    # Initial year and time step
    dates_1m = ['2021-07-19', '2021-08-14']
    dates_1h = ['2021-06-01', '2021-07-31']
    dates_other = ['1990-01-01', '2020-12-31']
    dates: List[List[str]] = [dates_1m, dates_1h, dates_other,
                              dates_other, dates_other]
    time_steps: List[str] = ['1m', '1h', '1d', '1wk', '1mo']
    cov_params: List[Tuple[Any]] = list(zip(dates, time_steps))

    # Basic folders
    exact_distributions_covariance_tools.start_folders()

    # Run analysis
    # Analysis and plot
    data_plot_generator(cov_params)

    print('Ay vamos!!!')

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
