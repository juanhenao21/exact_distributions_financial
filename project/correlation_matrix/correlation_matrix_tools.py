'''Portfolio optimization correlation matrix tools module.

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
# modified Bessel function of the second kind of real order v
from scipy.special import gamma, kv  # type:ignore

# -----------------------------------------------------------------------------


def save_data(data: Any, function_name: str, dates: List[str],
              time_step: str) -> None:
    """Saves computed data in pickle files.

    Saves the data generated in the functions of the
    correlation_matrix_analysis module in pickle files.

    :param data: data to be saved. The data can be of different types.
    :param function_name: name of the function that generates the plot.
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    # Saving data

    pickle.dump(data, open(
        f'../data/correlation_matrix/{function_name}_{dates[0]}_{dates[1]}'
                + f'_step_{time_step}.pickle', 'wb'), protocol=4)

    print('Data Saved')
    print()

# -----------------------------------------------------------------------------


def save_plot(figure: plt.Figure, function_name: str, dates: List[str],
              time_step: str) -> None:
    """Saves plot in png files.

    Saves the plot generated in the functions of the
    correlation_matrix_analysis module in png files.

    :param figure: figure object that is going to be save.
    :param function_name: name of the function that generates the plot.
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function save the plot in a file and does not return
     a value.
    """

    # Saving plot data

    figure.savefig(f'../plot/correlation_matrix/{function_name}_{dates[0]}'
                   + f'_{dates[1]}_step_{time_step}.png')

    print('Plot Saved')
    print()

# -----------------------------------------------------------------------------


def function_header_print_data(function_name: str, dates: List[str],
                               time_step: str) -> None:
    """Prints a header of a function that generates data when it is running.

    :param function_name: name of the function that generates the data.
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function prints a message and does not return a
     value.
    """

    print('Portfolio Optimization')
    print(function_name)

    print('Computing the results of the data in the interval time from the '
          + f'years {dates[0]} to {dates[1]} in time steps of {time_step}')
    print()

# -----------------------------------------------------------------------------


def function_header_print_plot(function_name: str, dates: List[str],
                               time_step: str) -> None:
    """Prints a header of a function that generates a plot when it is running.

    :param function_name: name of the function that generates the data.
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function prints a message and does not return a
     value.
    """

    print('Portfolio Optimization')
    print(function_name)

    print('Computing the plots of the data in the interval time from the '
          + f'years {dates[0]} to {dates[1]} in time steps of {time_step}')
    print()

# -----------------------------------------------------------------------------


def start_folders() -> None:
    """Creates the initial folders to save the data and plots.

    :return: None -- The function creates folders and does not return a value.
    """

    try:
        os.mkdir('../data/correlation_matrix')
        os.mkdir('../plot/correlation_matrix')
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
    print('##################')
    print('Correlation Matrix')
    print('##################')
    print('AG Guhr')
    print('Faculty of Physics')
    print('University of Duisburg-Essen')
    print('Author: Juan Camilo Henao Londono')
    print('More information in:')
    print('* https://juanhenao21.github.io/')
    print('* https://github.com/juanhenao21/portfolio_optimization')
    # print('* https://forex-response_spread-year.readthedocs.io/en/latest/')
    print()

# -----------------------------------------------------------------------------


def gaussian_distribution(mean: float, variance: float,
                          x_values: np.ndarray) -> np.ndarray:
    """Compute the Gaussian distribution values.

    :param mean: mean of the Gaussian distribution.
    :param variance: variance of the Gaussian distribution.
    :param x: array of the values to compute the Gaussian
     distribution
    """

    return (1 / (2 * np.pi * variance) ** 0.5) \
        * np.exp(-((x_values - mean) ** 2) / (2 * variance))

# -----------------------------------------------------------------------------


def k_distribution(returns: np.ndarray, N: float, alpha: int) -> np.ndarray:
    """Rotates the returns in the eigenbasis of the covariance matrix and
       normalizes them to standard deviation 1.

    The returns pd.DataFrame must have a TxK dimension, and K must be smaller
    than T.

    :param returns: np.ndarray with the return values.
    :param N: effective number of degrees of freedom.
    :param alpha: array of the values to compute the Gaussian
    distribution
    """

    first_part = (((np.sqrt(2)) ** (1 - N)) / (np.sqrt(np.pi) * gamma(N / 2)))\
                 * (np.sqrt(N / alpha)) ** (N / 2 + 0.5) \
                 * (np.abs(returns)) ** (N/2 - 0.5)

    second_part = kv(N / 2 - 0.5, np.abs(returns) * np.sqrt(N / alpha))

    dist = first_part * second_part

    return dist

# -----------------------------------------------------------------------------
def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
