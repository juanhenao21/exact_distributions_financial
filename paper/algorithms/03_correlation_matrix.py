'''Correlation matrix implementation.

Plots the figures of the correlation matrix comparison for the paper.

This script requires the following modules:
    * pickle
    * typing
    * matplotlib
    * pandas
    * seaborn

The module contains the following functions:
    * correlation_matrix_plot - plots the distributions matrix of the
      normalized returns.
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# ----------------------------------------------------------------------------
# Modules

import pickle
from typing import List

from matplotlib import pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

# ----------------------------------------------------------------------------


def correlation_matrix_plot(dates: List[List[str]], time_step: str) -> None:
    """Plots the correlation matrix of the normalized returns.
    :param dates: List of lists of the interval of dates to be analyzed
     (i.e. [['2005-10', '2005-12'], ['2006-01', '2006-03']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the plot in a file and does not return
     a value.
    """

    try:

        figure, (ax1, ax2, axcb) = \
            plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]},
                         figsize=(16, 9))

        ax1.get_shared_y_axes().join(ax2)

        # Load data
        corr_1: pd.DataFrame = pickle.load(open(
            f'../../project/data/correlation_matrix/correlation_matrix_data'
            + f'_{dates[0][0]}_{dates[0][1]}_step_{time_step}.pickle', 'rb'))
        corr_2: pd.DataFrame = pickle.load(open(
            f'../../project/data/correlation_matrix/correlation_matrix_data'
            + f'_{dates[1][0]}_{dates[1][1]}_step_{time_step}.pickle', 'rb'))

        hm_1 = sns.heatmap(corr_1, cmap='Blues', cbar=False, ax=ax1,
                           square=True)
        hm_1.set_ylabel('')
        hm_1.set_xlabel('')
        hm_1.set_xticks([])
        hm_1.set_yticks([])

        hm_2 = sns.heatmap(corr_2, cmap='Blues', cbar_ax=axcb, ax=ax2,
                           square=True)
        hm_2.set_ylabel('')
        hm_2.set_xlabel('')
        hm_2.set_xticks([])
        hm_2.set_yticks([])

        axcb.tick_params(labelsize=20)

        figure.tight_layout()

        # Save plot
        figure.savefig(f'../plot/03_correlation_matrix.png')

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

    correlation_matrix_plot([['2005-10', '2005-12'], ['2006-01', '2006-03']],
                            "1d")

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
