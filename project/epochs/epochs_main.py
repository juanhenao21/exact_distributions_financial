'''Epochs main module.

The functions in the module compute the returns and correlation matrix of
financial time series.

This script requires the following modules:
    * typing
    * epochs_analysis
    * epochs_plot
    * epochs_tools

The module contains the following functions:
    * data_plot_generator
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# -----------------------------------------------------------------------------
# Modules

from typing import List

import epochs_analysis
import epochs_plot
import epochs_tools

# -----------------------------------------------------------------------------


def data_plot_generator(dates: List[List[str]], time_steps: List[str],
                        windows: List[str], K_values: List[str]) -> None:
    """Generates all the analysis and plots from the data.

    :param dates: list of lists of the string of the dates to be analyzed
     (i.e. [['1980-01', '2020-12'], ['1980-01', '2020-12']).
    :param time_steps: list of the string of the time step of the data
     (i.e. ['1m', '2m', '5m']).
    :param windows: list of the string of the windows to be analyzed
     (i.e. ['60', '90']).
    :param K_values: list of the string of the number of companies to be
     analyzed (i.e. ['40', '80']).
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    # for idx, _ in enumerate(dates):

    #     epochs_analysis.returns_data(dates[idx], time_steps[idx])

    for K_value in K_values:
        for window in windows:
            for idx, _ in enumerate(dates):

                epochs_analysis. \
                    epochs_aggregated_dist_returns_market_data(dates[idx],
                                                               time_steps[idx],
                                                               window, K_value)

    # epochs_plot.epochs_var_win_all_empirical_dist_returns_market_plot()
    # epochs_plot.epochs_var_K_all_empirical_dist_returns_market_plot()
    # epochs_plot.epochs_var_time_step_all_empirical_dist_returns_market_plot()

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function extract, analyze and plot the data.

    :return: None.
    """

    epochs_tools.initial_message()

    # Initial year and time step

    dates_1m = ['2021-07-19', '2021-08-14']
    dates_other = ['1990-01-01', '2020-12-31']

    dates: List[List[str]] = [['alg', 'alg']]#[dates_1m, dates_other, dates_other, dates_other]
    time_steps: List[str] = ['1h']#'1m', '1d', '1wk', '1mo']
    windows: List[str] = ['10', '25', '40', '55', '500']
    K_values: List[str] = ['all']#'20', '50']

    # Basic folders
    epochs_tools.start_folders()

    # Run analysis
    # Analysis and plot
    data_plot_generator(dates, time_steps, windows, K_values)

    print('Ay vamos!!!')

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
