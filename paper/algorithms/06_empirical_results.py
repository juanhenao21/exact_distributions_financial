'''Exact distribution empirical results implementation.

Plots the figures of the empirical results for the paper.

This script requires the following modules:
    * pickle
    * sys
    * typing
    * matplotlib
    * numpy
    * pandas
    * exact_distributions_covariance_tools

The module contains the following functions:
    * pdf_all_distributions_plot - plots all the distributions and compares
      with agg. returns of a market.
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# ----------------------------------------------------------------------------
# Modules

import pickle
import sys
from typing import List

from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(1, '../../project/exact_distributions_covariance')
import exact_distributions_covariance_tools as exact_distributions_tools
# ----------------------------------------------------------------------------


def pdf_all_distributions_plot(dates: List[str], time_step: str) -> None:
    """Plots all the distributions and compares with agg. returns of a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    try:

        figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 16))

        # Load data
        agg_returns_data: pd.Series = pickle.load(open(
            '../../project/data/exact_distributions_covariance/aggregated_dist'
            + f'_returns_market_data_{dates[0]}_{dates[1]}_step_{time_step}'
            + f'.pickle', 'rb'))

        agg_returns_data = agg_returns_data.rename('Agg. returns')

        x_val_lin: np.ndarray = np.arange(-5, 5, 0.2)
        x_val_log: np.ndarray = np.arange(-10, 10, 0.5)

        markers: List[str] = ['o', '^', 's', 'P']

        # Lin plot
        plot_lin = agg_returns_data.plot(kind='density', style='-', ax=ax1,
                                         legend=True, lw=3)
        # Log plot
        plot_log = agg_returns_data.plot(kind='density', style='-', logy=True,
                                         ax=ax2, legend=True, lw=3)

        if dates[0] == '1992-01':
            N = 5
            N_gg = 4
            N_aa = 6
            K = 277
            L = 150
            l = 150

            gg_distribution_lin: np.ndarray = exact_distributions_tools\
                .pdf_gaussian_gaussian(x_val_lin, N_gg, 1)
            gg_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_gaussian_gaussian(x_val_log, N_gg, 1)
            ax1.plot(x_val_lin, gg_distribution_lin, markers[0], lw=3, ms=15,
                         label=f'GG - N = {N_gg}')
            ax2.semilogy(x_val_log, gg_distribution_log, markers[0], lw=3, ms=15,
                         label=f'GG - N = {N_gg}')

            ga_distribution_lin: np.ndarray = exact_distributions_tools\
                .pdf_gaussian_algebraic(x_val_lin, K, L, N, 1)
            ga_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_gaussian_algebraic(x_val_log, K, L, N, 1)
            ax1.plot(x_val_lin, ga_distribution_lin, markers[1], lw=3, ms=15,
                         label=f'GA - N = {N} - K = {K} - L = {L}')
            ax2.semilogy(x_val_log, ga_distribution_log, markers[1], lw=3, ms=15,
                         label=f'GA - N = {N} - K = {K} - L = {L}')

            ag_distribution_lin: np.ndarray = exact_distributions_tools\
                .pdf_algebraic_gaussian(x_val_lin, K, l, N, 1)
            ag_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_algebraic_gaussian(x_val_log, K, l, N, 1)
            ax1.plot(x_val_lin, ag_distribution_lin, markers[2], lw=3, ms=15,
                         label=f'AG - N = {N} - K = {K} - l = {l}')
            ax2.semilogy(x_val_log, ag_distribution_log, markers[2], lw=3, ms=15,
                         label=f'AG - N = {N} - K = {K} - l = {l}')

            aa_distribution_lin: np.ndarray = exact_distributions_tools\
                .pdf_algebraic_algebraic(x_val_lin, K, L, l, N_aa, 1)
            aa_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_algebraic_algebraic(x_val_log, K, L, l, N_aa, 1)
            ax1.plot(x_val_lin, aa_distribution_lin, markers[3], lw=3, ms=15,
                         label=f'AA - N = {N_aa} - K = {K} - L = {L}'
                         + f' - l = {l}')
            ax2.semilogy(x_val_log, aa_distribution_log, markers[3], lw=3, ms=15,
                         label=f'AA - N = {N_aa} - K = {K} - L = {L}'
                         + f' - l = {l}')

        else:
            N = 7
            N_gg = 6
            N_aa = 10
            K = 461
            L = 240
            l = 240

            gg_distribution_lin: np.ndarray = exact_distributions_tools\
                .pdf_gaussian_gaussian(x_val_lin, N_gg, 1)
            gg_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_gaussian_gaussian(x_val_log, N_gg, 1)
            ax1.plot(x_val_lin, gg_distribution_lin, markers[0], lw=3, ms=15,
                         label=f'GG - N = {N_gg}')
            ax2.semilogy(x_val_log, gg_distribution_log, markers[0], lw=3, ms=15,
                         label=f'GG - N = {N_gg}')

            ga_distribution_lin: np.ndarray = exact_distributions_tools\
                .pdf_gaussian_algebraic(x_val_lin, K, L, N, 1)
            ga_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_gaussian_algebraic(x_val_log, K, L, N, 1)
            ax1.plot(x_val_lin, ga_distribution_lin, markers[1], lw=3, ms=15,
                         label=f'GA - N = {N} - K = {K} - L = {L}')
            ax2.semilogy(x_val_log, ga_distribution_log, markers[1], lw=3, ms=15,
                         label=f'GA - N = {N} - K = {K} - L = {L}')

            ag_distribution_lin: np.ndarray = exact_distributions_tools\
                .pdf_algebraic_gaussian(x_val_lin, K, l, N, 1)
            ag_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_algebraic_gaussian(x_val_log, K, l, N, 1)
            ax1.plot(x_val_lin, ag_distribution_lin, markers[2], lw=3, ms=15,
                         label=f'AG - N = {N} - K = {K} - l = {l}')
            ax2.semilogy(x_val_log, ag_distribution_log, markers[2], lw=3, ms=15,
                         label=f'AG - N = {N} - K = {K} - l = {l}')

            aa_distribution_lin: np.ndarray = exact_distributions_tools\
                .pdf_algebraic_algebraic(x_val_lin, K, L, l, N_aa, 1)
            aa_distribution_log: np.ndarray = exact_distributions_tools\
                .pdf_algebraic_algebraic(x_val_log, K, L, l, N_aa, 1)
            ax1.plot(x_val_lin, aa_distribution_lin, markers[3], lw=3, ms=15,
                         label=f'AA - N = {N_aa} - K = {K} - L = {L} - l = {l}')
            ax2.semilogy(x_val_log, aa_distribution_log, markers[3], lw=3, ms=15,
                         label=f'AA - N = {N_aa} - K = {K} - L = {L} - l = {l}')

        # ax1.legend(fontsize=20)
        ax1.set_title(f"{dates[0].split(sep='-')[0]} - {dates[1].split(sep='-')[0]}", fontsize=30)
        ax1.set_xlabel(r'$\tilde{r}$', fontsize=25)
        ax1.set_ylabel('PDF', fontsize=25)
        ax1.tick_params(axis='x', labelsize=15)
        ax1.tick_params(axis='y', labelsize=15)
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(0, 0.6)
        ax1.grid(True)
        plt.tight_layout()

        # ax2.legend(fontsize=20)
        ax2.set_xlabel(r'$\tilde{r}$', fontsize=25)
        ax2.set_ylabel('PDF', fontsize=25)
        ax2.tick_params(axis='x', labelsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        ax2.set_xlim(-8, 8)
        ax2.set_ylim(10 ** -6, 1)
        ax2.grid(True)
        plt.tight_layout()

        # Save plot
        figure.savefig(f"../plot/06_all_{dates[0].split(sep='-')[0]}.png")

        plt.close()
        del agg_returns_data
        del figure
        del plot_lin
        del plot_log

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()


# ----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

    # pdf_all_distributions_plot(['1992-01', '2012-12'], "1d")
    pdf_all_distributions_plot(['2012-01', '2020-12'], "1d")

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
