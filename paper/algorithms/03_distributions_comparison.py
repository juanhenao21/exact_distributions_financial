'''Exact distributions implementation.

Plots the figures of the exact distributions implementation for the paper.

This script requires the following modules:
    * sys
    * typing
    * matplotlib
    * numpy
    * exact_distributions_covariance_tools

The module contains the following functions:
    * distributions_plot - plots the distributions in linear and logarithmic
      scale.
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# ----------------------------------------------------------------------------
# Modules

import sys
from typing import List

from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore

sys.path.insert(1, '../../project/exact_distributions_covariance')
import exact_distributions_covariance_tools as exact_distributions_tools

# ----------------------------------------------------------------------------


def distributions_plot(N: float, K: float, L: float, l: float) -> None:
    """Plots the distributions in linear and logarithmic scale.

    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    x_vals_lin: np.ndarray = np.arange(-10, 10, 0.2)
    x_vals_log: np.ndarray = np.arange(-10, 10, 0.5)

    gg_distribution_lin: np.ndarray = exact_distributions_tools \
        .pdf_gaussian_gaussian(x_vals_lin, N, 1)
    ga_distribution_lin: np.ndarray = exact_distributions_tools \
        .pdf_gaussian_algebraic(x_vals_lin, K, L, N, 1)
    ag_distribution_lin: np.ndarray = exact_distributions_tools \
        .pdf_algebraic_gaussian(x_vals_lin, K, l, N, 1)
    aa_distribution_lin: np.ndarray = exact_distributions_tools \
        .pdf_algebraic_algebraic(x_vals_lin, K, L, l, N, 1)

    gg_distribution_log: np.ndarray = exact_distributions_tools \
        .pdf_gaussian_gaussian(x_vals_log, N, 1)
    ga_distribution_log: np.ndarray = exact_distributions_tools \
        .pdf_gaussian_algebraic(x_vals_log, K, L, N, 1)
    ag_distribution_log: np.ndarray = exact_distributions_tools \
        .pdf_algebraic_gaussian(x_vals_log, K, l, N, 1)
    aa_distribution_log: np.ndarray = exact_distributions_tools \
        .pdf_algebraic_algebraic(x_vals_log, K, L, l, N, 1)

    markers: List[str] = ['-o', '-^', '-s', '-P']

    figure: plt.Figure = plt.figure(figsize=(9, 16))
    ax1: plt.subplot = plt.subplot(2, 1, 1)
    ax2: plt.subplot = plt.subplot(2, 1, 2)

    # Linear plot
    ax1.plot(x_vals_lin, gg_distribution_lin, markers[0], ms=10, label=f'GG')
    ax1.plot(x_vals_lin, ga_distribution_lin, markers[1], ms=10, label=f'GA')
    ax1.plot(x_vals_lin, ag_distribution_lin, markers[2], ms=10, label=f'AG')
    ax1.plot(x_vals_lin, aa_distribution_lin, markers[3], ms=10, label=f'AA')

    ax1.set_xlabel(r'$\tilde{r}$', fontsize=20)
    ax1.set_ylabel(r'PDF', fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(0, 0.65)
    ax1.grid(True)

    # Logarithmic plot
    ax2.semilogy(x_vals_log, gg_distribution_log, markers[0], ms=10,
                 label=f'GG')
    ax2.semilogy(x_vals_log, ga_distribution_log, markers[1], ms=10,
                 label=f'GA')
    ax2.semilogy(x_vals_log, ag_distribution_log, markers[2], ms=10,
                 label=f'AG')
    ax2.semilogy(x_vals_log, aa_distribution_log, markers[3], ms=10,
                 label=f'AA')

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=4,
               fontsize=20)
    ax2.set_xlabel(r'$\tilde{r}$', fontsize=20)
    ax2.set_ylabel(r'PDF', fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(10 ** -6.5, 1)
    ax2.grid(True)

    plt.tight_layout()

    # Save Plot
    figure.savefig(f'../plot/03_distributions_comparison.png')

# ----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

    N: float = 5
    K: float = 100
    L: float = 55
    l: float = 55

    distributions_plot(N, K, L, l)

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
