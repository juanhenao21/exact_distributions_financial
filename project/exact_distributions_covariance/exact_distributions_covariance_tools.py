'''Exact distributions covariance tools module.

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
    * gaussian_distribution - computes gaussian distribution values.
    * pdf_gaussian_gaussian - computes the one dimensional Gaussian-Gaussian
      PDF.
    * pdf_gaussian_algebraic - computes the one dimensional Gaussian-Algebraic
      PDF.
    * pdf_algebraic_gaussian - computes the one dimensional Algebraic-Gaussian
      PDF.
    * pdf_algebraic_algebraic - computes the one dimensional
      Algebraic-Algebraic PDF.
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
# Modified Bessel function of the second kind of real order v
from scipy.special import kv  # type: ignore
# Gauss hypergeometric function 2F1(a, b; c; z)
from scipy.special import hyp2f1  # type: ignore
# Confluent hypergeometric function U
from scipy.special import hyperu  # type: ignore

# -----------------------------------------------------------------------------


def save_data(data: Any, function_name: str, dates: List[str],
              time_step: str) -> None:
    """Saves computed data in pickle files.

    Saves the data generated in the functions of the
    exact_distributions_covariance_analysis module in pickle files.

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
        f'../data/exact_distributions_covariance/{function_name}_{dates[0]}'
                + f'_{dates[1]}_step_{time_step}.pickle', 'wb'), protocol=4)

    print('Data Saved')
    print()

# -----------------------------------------------------------------------------


def save_plot(figure: plt.Figure, function_name: str, dates: List[str],
              time_step: str) -> None:
    """Saves plot in png files.

    Saves the plot generated in the functions of the
    exact_distributions_covariance_analysis module in png files.

    :param figure: figure object that is going to be save.
    :param function_name: name of the function that generates the plot.
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function save the plot in a file and does not return
     a value.
    """

    # Saving plot data

    figure.savefig(f'../plot/exact_distributions_covariance/{function_name}'
                   + f'_{dates[0]}_{dates[1]}_step_{time_step}.png')

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

    print('Exact Distributions')
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

    print('Exact Distributions')
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
        os.mkdir('../data/exact_distributions_covariance')
        os.mkdir('../plot/exact_distributions_covariance')
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
    print('###################')
    print('Exact Distributions')
    print('###################')
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
    """Computes the Gaussian distribution values.

    :param mean: mean of the Gaussian distribution.
    :param variance: variance of the Gaussian distribution.
    :param x: array of the values to compute the Gaussian
     distribution
    """

    return (1 / (2 * np.pi * variance) ** 0.5) \
        * np.exp(-((x_values - mean) ** 2) / (2 * variance))

# -----------------------------------------------------------------------------


def pdf_gaussian_gaussian(returns: np.ndarray, N: float,
                          Lambda: float) -> np.ndarray:
    ''' Computes the one dimensional Gaussian-Gaussian PDF.

    :param returns: numpy array with the returns values.
    :param N: strength of the fluctuations around the mean.
    :param Lambda: variance of the returns.
    :return: numpy array with the pdf values.
    '''

    first_part: np.float = 1 / (2 ** ((N - 1) / 2) * gamma(N / 2)
                                * np.sqrt((np.pi * Lambda) / N))
    second_part: np.ndarray = np.sqrt((N * returns ** 2) / Lambda) \
        ** ((N - 1) / 2)
    third_part: np.ndarray = kv((1 - N) / 2,
                                np.sqrt(N * returns ** 2) / Lambda)

    return first_part * second_part * third_part

# -----------------------------------------------------------------------------


def pdf_gaussian_algebraic(returns: np.ndarray, K: float, L: float, N: float,
                           Lambda: float) -> np.ndarray:
    '''Computes de one dimensional Gaussian-Algebraic PDF.

    :param returns: numpy array with the returns values.
    :param K: number of companies.
    :param L: shape parameter.
    :param N: strength of the fluctuations around the mean.
    :param Lambda: variance of the returns.
    :return: numpy array with the pdf values.
    '''

    M: np.float = 2 * L - K - N - 1

    numerator: np.float = gamma(L - (K + N) / 2 + 1) * gamma(L - (K - 1) / 2)
    denominator: np.float = gamma(L - (K + N - 1) / 2) * gamma(N / 2) \
        * np.sqrt(2 * np.pi * Lambda * M / N)

    frac: np.float = numerator / denominator

    function: np.ndarray = hyperu(L - (K + N) / 2 + 1, (1 - N) / 2 + 1,
                                  (N * returns ** 2) / (2 * M * Lambda))

    return frac * function

# -----------------------------------------------------------------------------


def pdf_algebraic_gaussian(returns: np.ndarray, K: float, l: float, N: float,
                           Lambda: float) -> np.ndarray:
    '''Computes de one dimensional Algebraic-Gaussian PDF.

    :param returns: numpy array with the returns values.
    :param K: number of companies.
    :param l: shape parameter.
    :param N: strength of the fluctuations around the mean.
    :param Lambda: variance of the returns.
    :return: numpy array with the pdf values.
    '''

    m: np.float = 2 * l - K - 2

    numerator: np.float = gamma(l - (K - 1) / 2) * gamma(l - (K - N) / 2)
    denominator: np.float = gamma(l - K / 2) * gamma(N / 2) \
        * np.sqrt(2 * np.pi * Lambda * m / N)

    frac: np.float = numerator / denominator

    function: np.ndarray = hyperu(l - (K - 1) / 2, (1 - N) / 2 + 1,
                                  (N * returns ** 2) / (2 * m * Lambda))

    return frac * function

# -----------------------------------------------------------------------------


def pdf_algebraic_algebraic(returns: np.ndarray, K: float, L: float, l: float,
                            N: float, Lambda: float) -> np.ndarray:
    '''Computes de one dimensional Algebraic-Algebraic PDF.

    :param returns: numpy array with the returns values.
    :param K: number of companies.
    :param L: shape parameter.
    :param l: shape parameter.
    :param N: strength of the fluctuations around the mean.
    :param Lambda: variance of the returns.
    :return: numpy array with the pdf values.
    '''

    M: np.float = 2 * L - K - N - 1
    m: np.float = 2 * l - K - 2

    numerator: np.float = gamma(l - (K - 1) / 2) * gamma(l - (K - N) / 2) \
        * gamma(L - (K - 1) / 2) * gamma(L - (K + N) / 2 + 1)
    denominator: np.float = np.sqrt(np.pi * Lambda * M * m / N) \
        * gamma(l - K / 2) * gamma(L + l - (K - 1)) \
        * gamma(L - (K + N - 1) / 2) * gamma(N / 2)

    frac: np.float = numerator / denominator

    function: np.ndarray = hyp2f1(l - (K - 1) / 2, L - (K + N) / 2 + 1,
                                  L + l - (K - 1),
                                  1 - (N * returns ** 2) / (M * m * Lambda))

    return frac * function

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
