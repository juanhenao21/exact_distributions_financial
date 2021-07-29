'''Epochs tools module.

The functions in the module do small repetitive tasks, that are used along the
whole implementation. These tools improve the way the tasks are standardized
in the modules that use them.

This script requires the following modules:
    * os
    * pickle
    * typing
    * matplotlib
    * numpy
    * scipy

The module contains the following functions:
    * save_data - saves computed data.
    * save_plot - saves figures.
    * function_header_print_data - prints info about the function running.
    * function_header_print_plot - prints info about the plot.
    * start_folders - creates folders to save data and plots.
    * initial_message - prints the initial message with basic information.
    * gaussian_distribution - compute gaussian distribution values.
    * algebraic_distribution - compute algebraic distribution values.
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# -----------------------------------------------------------------------------
# Modules

import os
import pickle
from typing import Any, List

from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
# Gamma function
from scipy.special import gamma  # type: ignore

# -----------------------------------------------------------------------------


def save_data(data: Any, function_name: str, dates: List[str], time_step: str,
              window: str) -> None:
    """Saves computed data in pickle files.

    Saves the data generated in the functions of the epochs_analysis module in
    pickle files.

    :param data: data to be saved. The data can be of different types.
    :param function_name: name of the function that generates the plot.
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to use in the computation (i.e. '25').
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    # Saving data

    pickle.dump(data, open(
        f'../data/epochs/{function_name}_{dates[0]}_{dates[1]}_step'
                + f'_{time_step}_win_{window}.pickle', 'wb'), protocol=4)

    print('Data Saved')
    print()

# -----------------------------------------------------------------------------


def save_plot(figure: plt.Figure, function_name: str, dates: List[str],
              time_step: str, window: str) -> None:
    """Saves plot in png files.

    Saves the plot generated in the functions of the epochs_analysis module in
    png files.

    :param figure: figure object that is going to be save.
    :param function_name: name of the function that generates the plot.
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function save the plot in a file and does not return
     a value.
    """

    # Saving plot data

    figure.savefig(f'../plot/epochs/{function_name}_{dates[0]}_{dates[1]}'
                   + f'_step_{time_step}_win_{window}.png')

    print('Plot Saved')
    print()

# -----------------------------------------------------------------------------


def function_header_print_data(function_name: str, dates: List[str],
                               time_step: str, window: str) -> None:
    """Prints a header of a function that generates data when it is running.

    :param function_name: name of the function that generates the data.
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function prints a message and does not return a
     value.
    """

    print('Exact Distributions')
    print(function_name)

    print(f'Computing the results of the data in the interval time from the '
          + f'years {dates[0]} to {dates[1]} in time steps of {time_step} '
          + f'with a time window of {window}')
    print()

# -----------------------------------------------------------------------------


def function_header_print_plot(function_name: str, dates: List[str],
                               time_step: str, window: str) -> None:
    """Prints a header of a function that generates a plot when it is running.

    :param function_name: name of the function that generates the data.
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function prints a message and does not return a
     value.
    """

    print('Exact Distributions')
    print(function_name)

    print(f'Computing the plots of the data in the interval time from the '
          + f'years {dates[0]} to {dates[1]} in time steps of {time_step} '
          + f'with a time window of {window}')
    print()

# -----------------------------------------------------------------------------


def start_folders() -> None:
    """Creates the initial folders to save the data and plots.

    :return: None -- The function creates folders and does not return a value.
    """

    try:
        os.mkdir(f'../data/epochs')
        os.mkdir(f'../plot/epochs')
        print('Folder to save data created')
        print()

    except FileExistsError as error:
        print('Folder exists. The folder was not created')
        print(error)
        print()

# -----------------------------------------------------------------------------


def initial_message() -> None:
    """Prints the initial message with basic information.

    :return: None -- The function prints a message and does not return a value.
    """

    print()
    print('######')
    print('Epochs')
    print('######')
    print('AG Guhr')
    print('Faculty of Physics')
    print('University of Duisburg-Essen')
    print('Author: Juan Camilo Henao Londono')
    print('More information in:')
    print('* https://juanhenao21.github.io/')
    print('* https://github.com/juanhenao21/exact_distributions')
    # print('* https://forex-response_spread-year.readthedocs.io/en/latest/')
    print()

# -----------------------------------------------------------------------------


def gaussian_distribution(mean: float, variance: float,
                          x_values: np.ndarray) -> np.ndarray:
    """Compute the Gaussian distribution values.

        :param mean: mean of the Gaussian distribution.
        :param variance: variance of the Gaussian distribution.
        :param x_values: array of the values to compute the Gaussian
         distribution
    """

    return (1 / (2 * np.pi * variance) ** 0.5) \
        * np.exp(-((x_values - mean) ** 2) / (2 * variance))

# -----------------------------------------------------------------------------


def algebraic_distribution(variance: float, K: int, l: int, m: int,
                           x_values: np.ndarray) -> np.ndarray:
    """Compute the algebraic distribution values.

        :param variance: variance of the algebraic distribution.
        :param K: number of companies analyzed.
        :param l: shape parameter.
        :param m: shape parameter.
        :param x_values: array of the values to compute the Gaussian
         distribution
    """

    return (np.sqrt(2 / m)) ** K * (gamma(l) / gamma(l - K / 2)) \
        (1 / np.sqrt(2 * np.pi * variance)) \
        * (1 / (1 + (1 / m) * x_values * x_values * (1 / variance)))

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
