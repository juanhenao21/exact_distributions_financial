from project.epochs.epochs_tools import gaussian_distribution
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#------------------------------------------------------------------------------


def returns_simulation(out_diag_val: float,
                       size_corr_mat: int,
                       epochs_len: int) -> pd.DataFrame:

    corr_matrix: np.ndarray = out_diag_val * np.ones((size_corr_mat,
                                                      size_corr_mat))
    np.fill_diagonal(corr_matrix, 1)
    # print(corr_matrix)

    eig_val_corr, eig_vec_corr = np.linalg.eigh(corr_matrix)
    # print(eig_val_corr)
    # print(eig_vec_corr)

    eig_val_corr_mat = np.diag(np.sqrt(eig_val_corr))

    r_epochs_list = []

    for _ in range(epochs_len):

        z_val = np.random.normal(0, 1, size_corr_mat)

        r  = (eig_vec_corr * eig_val_corr_mat * z_val).diagonal()
        r_epochs_list.append(r)

    r_epochs = np.array(r_epochs_list)

    r_epochs_df = pd.DataFrame(r_epochs,
                               columns=[f'Stock_{i}'
                                        for i in range(size_corr_mat)])

    return r_epochs_df

#------------------------------------------------------------------------------


def epochs_agg_returns_pair_data(dataframe: pd.DataFrame) -> List[float]:

    agg_ret_mkt_list: List[float] = []

    # dataframe = (dataframe -dataframe.mean()) / dataframe.std()

    cov_two_col: pd.DataFrame = dataframe.cov()
    # eig_vec:  eigenvector, eig_val: eigenvalues
    eig_val, eig_vec = np.linalg.eigh(cov_two_col)

    # rot: rotation, scal: scaling
    rot, scale = eig_vec, np.diag(1 / np.sqrt(eig_val))
    # trans: transformation matrix
    # trans = rot . scale
    trans = rot.dot(scale)

    try:
        # Transform the returns
        trans_col: pd.DataFrame = dataframe.dot(trans)
        # Length DataFrame
        col_length = len(trans_col.columns)
        print(col_length)
        # Name the columns with the used stocks
        trans_col.columns = [f'Stock_{i}' for i in range(col_length)]

        one_col: List[Any] = []

        for idx in range(col_length):
            one_col.append(trans_col[f'Stock_{idx}'])

        print(one_col)
        agg_ret_mkt_list = pd.concat(one_col, ignore_index=True)

        print(agg_ret_mkt_list)
        # agg_ret_mkt_list.extend(one_col)

        del one_col
        del trans_col

    except np.linalg.LinAlgError as error:
        print(error)
        print()

        del one_col
        del trans_col

    # remove NaN and Inf
    agg_ret_mkt_list = [x for x in agg_ret_mkt_list if not math.isnan(x)
                        and not math.isinf(x)]
    # filter out values greater than 10 or smaller than -10
    agg_ret_mkt_list = [x for x in agg_ret_mkt_list if x <= 10
                        and x >= -10]

    print(type(agg_ret_mkt_list))
    return agg_ret_mkt_list

#------------------------------------------------------------------------------


def epochs_agg_returns_market_data(out_diag_val: float,
                                   size_corr_matrix: int,
                                   K_values: int,
                                   epochs_num:int,
                                   epochs_len: int) -> pd.Series:

    agg_ret_mkt_list: List[List[float]] = []

    for _ in range(K_values):
        for _ in range(epochs_num):

            returns: pd.DataFrame = \
                returns_simulation(out_diag_val, size_corr_matrix, epochs_len)
            agg_ret_list = epochs_agg_returns_pair_data(returns)
            agg_ret_mkt_list.extend(agg_ret_list)

    agg_ret_mkt_series: pd.Series = pd.Series(agg_ret_mkt_list)

    print(f'mean = {agg_ret_mkt_series.mean()}')
    print(f'std  = {agg_ret_mkt_series.std()}')

    return agg_ret_mkt_series

#------------------------------------------------------------------------------


def epochs_agg_retuns_market_plot(agg_ret: pd.Series, win: int) -> None:

    x_gauss: np.ndarray = np.arange(-6, 6, 0.001)
    gaussian: np.ndarray = gaussian_distribution(0, 1, x_gauss)

    plot_lin = agg_ret.plot(kind='density', figsize=(16, 9), logy=True)
    plt.semilogy(x_gauss, gaussian, label='Gaussian')
    plt.xlim(-6, 6)
    plt.ylim(10 ** -5, 1)
    plt.grid(True)
    plt.show()
    figure_log = plot_lin.get_figure()
    figure_log.savefig(f'epoch_win_{win}.png')

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

#------------------------------------------------------------------------------


def main() -> None:

    for epochs_len in [10, 25, 40, 55]:
        x = epochs_agg_returns_market_data(0.3, 2, 100, 40, epochs_len)
        epochs_agg_retuns_market_plot(x, epochs_len)

    # for epochs_len in [10, 25, 40, 55]:
    #     x = epochs_agg_returns_market_data(0.3, 250, 100, 40, epochs_len)
    #     epochs_agg_retuns_market_plot(x, epochs_len)

    # x = returns_simulation(0.3, 2, 25)
    # ser = epochs_agg_returns_pair_data(x)
    # epochs_agg_retuns_market_plot(ser, 25)
    # print(x)

#------------------------------------------------------------------------------


if __name__ == '__main__':
    main()