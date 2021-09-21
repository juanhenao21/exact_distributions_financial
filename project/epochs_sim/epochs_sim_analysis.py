'''Epochs simulation analysis module.

The functions in the module simulate the returns and their rotation and
aggregation. They also compute key results from other methods.

This script requires the following modules:
    * math
    * typing
    * numpy
    * pandas
    * epochs_sim_tools

The module contains the following functions:
    * returns_simulation - simulates the returns of a time series.
    * epochs_sim_agg_returns_pair_data - uses local normalization to compute
      the aggregated distribution of returns for a pair of simulated stocks.
    * epochs_agg_returns_market_data - uses local normalization to compute the
      aggregated distribution of returns for a simulated market.
    * main - the main function of the script.

..moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
'''

# -----------------------------------------------------------------------------
# Modules

import math
from typing import Any, Iterable, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import epochs_sim_tools

# -----------------------------------------------------------------------------


def returns_simulation(out_diag_val: float,
                       size_corr_mat: int,
                       epochs_len: int) -> pd.DataFrame:
    """Simulates the returns of a time series

    :param out_diag_val: numerical value of the off diagonal elements.
    :type out_diag_val: float
    :param size_corr_mat: dimensions of the correlation matrix.
    :type size_corr_mat: int
    :param epochs_len: length of the epochs.
    :type epochs_len: int
    :return: dataframe with the simulated returns.
    :rtype: pd.DataFrame
    """

    corr_matrix: np.ndarray = out_diag_val * np.ones((size_corr_mat,
                                                      size_corr_mat))
    np.fill_diagonal(corr_matrix, 1)

    eig_val_corr: np.ndarray
    eig_vec_corr: np.ndarray
    eig_val_corr, eig_vec_corr = np.linalg.eigh(corr_matrix)

    eig_val_corr_mat: np.ndarray = np.diag(np.sqrt(eig_val_corr))

    ret_epochs_list: List[np.ndarray] = []

    for _ in range(epochs_len):

        z_val: np.ndarray = np.random.normal(0, 1, size_corr_mat)

        ret_vals: np.ndarray =  \
            (eig_vec_corr * eig_val_corr_mat * z_val).diagonal()

        ret_epochs_list.append(ret_vals)

    ret_epochs: np.ndarray = np.array(ret_epochs_list)

    ret_epochs_df: pd.DataFrame = \
        pd.DataFrame(ret_epochs, columns=[f'Stock_{i}'
                                          for i in range(size_corr_mat)])

    return ret_epochs_df

# -----------------------------------------------------------------------------


def epochs_sim_agg_returns_pair_data(dataframe: pd.DataFrame) -> List[float]:
    """Uses local normalization to compute the aggregated distribution of
       returns for a pair of simulated stocks.

    :param dataframe: dataframe with the simulated returns.
    :type dataframe: pd.DataFrame
    :return: simulated rotated returns.
    :rtype: List[float]
    """

    cov_two_col: pd.DataFrame = dataframe.cov()
    # eig_vec:  eigenvector, eig_val: eigenvalues
    eig_val_corr: np.ndarray
    eig_vec_corr: np.ndarray
    eig_val, eig_vec = np.linalg.eigh(cov_two_col)

    # rot: rotation, scale: scaling
    rot: np.ndarray
    scale: np.ndarray
    rot, scale = eig_vec, np.diag(1 / np.sqrt(eig_val))
    # trans: transformation matrix
    # trans = rot . scale
    trans: np.ndarray = rot.dot(scale)

    try:
        # Transform the returns
        trans_col: pd.DataFrame = dataframe.dot(trans)
        # Length DataFrame
        col_length: int = len(trans_col.columns)
        # Name the columns with the used stocks
        trans_col.columns = [f'Stock_{i}' for i in range(col_length)]

        one_col: List[Any] = []

        for idx in range(col_length):
            one_col.append(trans_col[f'Stock_{idx}'])

        agg_ret_mkt_series: pd.Series = pd.concat(one_col, ignore_index=True)

        del one_col
        del trans_col

    except np.linalg.LinAlgError as error:
        print(error)
        print()

        del one_col
        del trans_col

    # remove NaN and Inf
    agg_ret_mkt_list: List[float] = \
        [x for x in agg_ret_mkt_series if not math.isnan(x)
         and not math.isinf(x)]
    # filter out values greater than 10 or smaller than -10
    agg_ret_mkt_list = [x for x in agg_ret_mkt_list if x <= 10
                        and x >= -10]

    return agg_ret_mkt_list

# -----------------------------------------------------------------------------


def epochs_sim_agg_returns_market_data(out_diag_val: float,
                                       size_corr_matrix: int,
                                       K_values: int,
                                       epochs_num:int,
                                       epochs_len: int) -> pd.Series:
    """Uses local normalization to compute the aggregated distribution of
       returns for a simulated market.

    :param out_diag_val: numerical value of the off diagonal elements.
    :type out_diag_val: float
    :param size_corr_matrix: dimensions of the correlation matrix
    :type size_corr_matrix: int
    :param K_values: number of companies to be simulated.
    :type K_values: int
    :param epochs_num: number of epochs to be simulated.
    :type epochs_num: int
    :param epochs_len: length of the epochs.
    :type epochs_len: int
    :return: simulated rotated and aggregated returns for a simulated market.
    :rtype: pd.Series
    """

    function_name: str = epochs_sim_agg_returns_market_data.__name__
    epochs_sim_tools \
        .function_header_print_data(function_name, '', '', '', '', sim=True)

    agg_ret_mkt_list: List[List[float]] = []

    for _ in range(K_values):
        for _ in range(epochs_num):

            returns: pd.DataFrame = \
                returns_simulation(out_diag_val, size_corr_matrix, epochs_len)
            agg_ret_list: List[float] = \
                epochs_sim_agg_returns_pair_data(returns)
            agg_ret_mkt_list.extend(agg_ret_list)

    agg_ret_mkt_series: pd.Series = pd.Series(agg_ret_mkt_list)

    print(f'mean = {agg_ret_mkt_series.mean()}')
    print(f'std  = {agg_ret_mkt_series.std()}')

    return agg_ret_mkt_series

# ----------------------------------------------------------------------------


def epochs_sim_agg_returns_cov_market_data(returns: pd.DataFrame) -> None:
    """Computes the aggregated distribution of returns for a market.

    :param returns: dataframe with the simulated returns.
    :type returns: pd.DataFrame
    """

    function_name: str = epochs_sim_agg_returns_cov_market_data.__name__
    epochs_sim_tools \
        .function_header_print_data(function_name, '', '', '', '', sim=True)

    Aqui

    try:

        # Load data
        returns_vals: pd.DataFrame = pickle.load(open(
            f'../data/exact_distributions_covariance/returns_data_{dates[0]}'
            + f'_{dates[1]}_step_{time_step}.pickle', 'rb'))

        print('Size of time series and number of companies: ',
              returns_vals.shape)

        returns_vals = (returns_vals - returns_vals.mean()) \
            / returns_vals.std()

        cov: pd.DataFrame = returns_vals.cov()
        # eig_vec:  eigenvector, eig_val: eigenvalues
        eig_val, eig_vec = np.linalg.eig(cov)

        # rot: rotation, scal: scaling
        rot, scale = eig_vec, np.diag(1 / np.sqrt(eig_val))
        # trans: transformation matrix
        # trans = rot . scal
        trans = rot.dot(scale)

        trans_returns: pd.DataFrame = returns_vals.dot(trans)
        trans_returns.columns = returns_vals.columns

        one_col: List[pd.Series] = []

        for col in trans_returns.columns:

            one_col.append(trans_returns[col])

        agg_returns = pd.concat(one_col, ignore_index=True)

        print(f'Std. Dev. {dates} = {agg_returns.std()}')

        # Saving data
        exact_distributions_covariance_tools \
            .save_data(agg_returns, function_name, dates, time_step)

        del returns_vals
        del trans_returns
        del agg_returns
        del one_col

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
