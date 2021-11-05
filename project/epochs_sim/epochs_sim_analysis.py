'''Epochs simulation analysis module.

The functions in the module simulate the returns and their rotation and
aggregation. They also compute key results from other methods.

This script requires the following modules:
    * itertools
    * math
    * multiprocessing
    * pickle
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

from itertools import product as iprod
from itertools import combinations as icomb
import math
import multiprocessing as mp
import pickle
# from scipy.stats import multivariate_t
from typing import Any, Iterable, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import epochs_sim_tools

# -----------------------------------------------------------------------------


def returns_simulation_gaussian(out_diag_val: float,
                                size_corr_mat: int,
                                epochs_len: int) -> pd.DataFrame:
    """Simulates the gaussian distributed returns of a time series

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


def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

# -----------------------------------------------------------------------------

def returns_simulation_algebraic(out_diag_val: float,
                                 size_corr_mat: int,
                                 epochs_len: int,
                                 df: int) -> pd.DataFrame:
    """Simulates the algebraic distributed returns of a time series

    :param out_diag_val: numerical value of the off diagonal elements.
    :type out_diag_val: float
    :param size_corr_mat: dimensions of the correlation matrix.
    :type size_corr_mat: int
    :param epochs_len: length of the epochs.
    :type epochs_len: int
    :param df: degress of freedom.
    :type df: int
    :return: dataframe with the simulated returns.
    :rtype: pd.DataFrame
    """

    corr_matrix: np.ndarray = out_diag_val * np.ones((size_corr_mat,
                                                      size_corr_mat))
    np.fill_diagonal(corr_matrix, 1)

    ret_epochs_list: List[np.ndarray] = []

    for _ in range(epochs_len):

        ret_vals: np.ndarray =  \
            multivariate_t_rvs(
                np.array(np.zeros((size_corr_mat, size_corr_mat))),
                corr_matrix, df=df)

        ret_epochs_list.append(ret_vals[0])

    ret_epochs: np.ndarray = np.array(ret_epochs_list)

    ret_epochs_df: pd.DataFrame = \
        pd.DataFrame(ret_epochs, columns=[f'Stock_{i}'
                                          for i in range(size_corr_mat)])

    return ret_epochs_df

# -----------------------------------------------------------------------------


def epochs_sim_agg_returns_pair_data(dataframe: pd.DataFrame,
                                     normalized: bool =False) -> List[float]:
    """Uses local normalization to compute the aggregated distribution of
       returns for a pair of simulated stocks.

    :param dataframe: dataframe with the simulated returns.
    :type dataframe: pd.DataFrame
    :param normalized: normalize the returns within the epochs, defaults to
     False
    :type normalized: bool, optional
    :return: simulated rotated returns.
    :rtype: List[float]
    """

    if normalized:
        dataframe = (dataframe - dataframe.mean()) / dataframe.std()

    cov_two_col: pd.DataFrame = dataframe.cov()
    # eig_vec:  eigenvector, eig_val: eigenvalues
    eig_val_corr: np.ndarray
    eig_vec_corr: np.ndarray
    eig_val_corr, eig_vec_corr = np.linalg.eigh(cov_two_col)

    # rot: rotation, scale: scaling
    rot: np.ndarray
    scale: np.ndarray
    rot, scale = eig_vec_corr, np.diag(1 / np.sqrt(eig_val_corr))
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
    agg_ret_mkt_list = [x for x in agg_ret_mkt_list if -10 <= x <= 10]

    return agg_ret_mkt_list

# -----------------------------------------------------------------------------


def epochs_sim_agg_returns_market_data(out_diag_val: float,
                                       size_corr_matrix: int,
                                       K_values: int,
                                       epochs_num: int,
                                       epochs_len: int,
                                       normalized: bool = False,
                                       kind: str = 'gaussian',
                                       df: int = 10) -> pd.Series:
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
    :param normalized: normalize the returns within the epochs, defaults to
     False
    :type normalized: bool, optional
    :param kind: kind of returns to be used, defaults to gaussian.
    :type kind: str, optional
    :return: simulated rotated and aggregated returns for a simulated market.
    :rtype: pd.Series
    """

    function_name: str = epochs_sim_agg_returns_market_data.__name__
    epochs_sim_tools \
        .function_header_print_data(function_name, [''], '', '', '', sim=True)

    agg_ret_mkt_list: List[float] = []

    for _ in range(K_values):
        for _ in range(epochs_num):

            if kind == 'gaussian':
                returns: pd.DataFrame = \
                    returns_simulation_gaussian(out_diag_val, size_corr_matrix,
                                                epochs_len)
            else:
                returns: pd.DataFrame = \
                    returns_simulation_algebraic(out_diag_val, size_corr_matrix,
                                                epochs_len, df)

            agg_ret_list: List[float] = \
                epochs_sim_agg_returns_pair_data(returns,normalized=normalized)
            agg_ret_mkt_list.extend(agg_ret_list)

    agg_ret_mkt_series: pd.Series = pd.Series(agg_ret_mkt_list)

    print(f'mean = {agg_ret_mkt_series.mean()}')
    print(f'std  = {agg_ret_mkt_series.std()}')

    return agg_ret_mkt_series

# ----------------------------------------------------------------------------


def epochs_sim_agg_returns_cov_market_data(returns: pd.DataFrame) -> pd.Series:
    """Computes the aggregated distribution of returns for a market.

    :param returns: dataframe with the simulated returns.
    :type returns: pd.DataFrame
    """

    function_name: str = epochs_sim_agg_returns_cov_market_data.__name__
    epochs_sim_tools \
        .function_header_print_data(function_name, [''], '', '', '', sim=True)

    print('Size of time series and number of companies: ', returns.shape)

    cov: pd.DataFrame = returns.cov()
    # eig_vec:  eigenvector, eig_val: eigenvalues
    eig_val, eig_vec = np.linalg.eigh(cov)

    # rot: rotation, scal: scaling
    rot, scale = eig_vec, np.diag(1 / np.sqrt(eig_val))
    # trans: transformation matrix
    # trans = rot . scal
    trans = rot.dot(scale)

    trans_returns: pd.DataFrame = returns.dot(trans)
    trans_returns.columns = returns.columns

    one_col: List[pd.Series] = []

    for col in trans_returns.columns:

        one_col.append(trans_returns[col])

    agg_returns: pd.Series = pd.concat(one_col, ignore_index=True)

    # remove NaN and Inf
    agg_returns_list: List[float] = [x for x in agg_returns
                                     if not math.isnan(x)
                                     and not math.isinf(x)]
    # filter out values greater than 10 or smaller than -10
    agg_returns_list = [x for x in agg_returns_list if -10 <= x <= 10]

    agg_returns_series: pd.Series = pd.Series(agg_returns_list)
    print(f'mean = {agg_returns_series.mean()}')
    print(f'std  = {agg_returns_series.std()}')

    del returns
    del trans_returns
    del one_col

    return agg_returns_series

# ----------------------------------------------------------------------------


def epochs_sim_rot_pair_data(kind: str,
                             K_value: str,
                             cols: List[str],
                             window: str) -> List[float]:
    """Uses local normalization to compute the aggregated distribution of
       returns for a pair of stocks.

    :param kind: kind of returns to be used (i.e 'gaussian').
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :param cols: pair of stocks to be analized (i. e. ['AAPL', 'MSFT']).
    :param window: window time to rotate and scale (i.e. '25').
    :return: List[float] -- The function returns a list with float numbers.
    """

    try:

        # Load data
        two_col: pd.DataFrame = pickle.load(open(
            f'../data/epochs_sim/returns_simulation_{kind}_K_{K_value}.pickle',
            'rb'))[[cols[0], cols[1]]]

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()
        return [0]

    two_col = (two_col - two_col.mean()) / two_col.std()

    # List to extend with the returns values of each pair
    agg_ret_mkt_list: List[float] = []

    # Add a column grouping the returns in the time window
    two_col['Group'] = np.arange(len(two_col)) // int(window)

    # Remove groups that are smaller than the window to avoid linalg errors
    two_col = two_col.groupby('Group') \
        .filter(lambda x: len(x) >= int(window) - 5)

    for local_data in two_col.groupby(by=['Group']):

        # Use the return columns
        local_data_df: pd.DataFrame = local_data[1][[cols[0], cols[1]]]

        cov_two_col: pd.DataFrame = local_data_df.cov()
        # eig_vec:  eigenvector, eig_val: eigenvalues
        eig_val, eig_vec = np.linalg.eig(cov_two_col)

        # rot: rotation, scale: scaling
        rot, scale = eig_vec, np.diag(1 / np.sqrt(eig_val))
        # trans: transformation matrix
        # trans = rot . scale
        trans = rot.dot(scale)

        try:
            # Transform the returns
            trans_two_col = local_data_df.dot(trans)
            # Name the columns with the used stocks
            trans_two_col.columns = [cols[0], cols[1]]

            one_col = trans_two_col[cols[0]].append(trans_two_col[cols[1]],
                                                    ignore_index=True)

            agg_ret_mkt_list.extend(one_col)

            del local_data_df
            del one_col
            del trans_two_col

        except np.linalg.LinAlgError as error:
            print(error)
            print()

            del local_data_df
            del one_col
            del trans_two_col

    del two_col

    # remove NaN and Inf
    agg_ret_mkt_list = [x for x in agg_ret_mkt_list if not math.isnan(x)
                        and not math.isinf(x)]
    # filter out values greater than 10 or smaller than -10
    agg_ret_mkt_list = [x for x in agg_ret_mkt_list if -10 <= x <= 10]

    return agg_ret_mkt_list


# ----------------------------------------------------------------------------


def epochs_sim_rot_market_data(kind: str,
                               K_value: str,
                               window: str) -> None:
    """Computes the aggregated distribution of returns for a market.

    :param kind: kind of returns to be used (i.e 'gaussian').
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    function_name: str = epochs_sim_rot_market_data.__name__
    epochs_sim_tools \
        .function_header_print_data(function_name, [''], '', window, '',
                                    sim=True)
    try:

        # Load data
        stocks_name: pd.DataFrame = pickle.load(open(
            f'../data/epochs_sim/returns_simulation_{kind}_K_{K_value}.pickle',
            'rb'))

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()
        return [0]

    agg_ret_mkt_list: List[List[float]] = []

    # Combination of stock pairs
    stocks_comb: Iterable[Tuple[Any, ...]] = icomb(stocks_name, 2)
    args_prod: Iterable[Any] = iprod([kind], [K_value], stocks_comb, [window])

    # Parallel computing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        agg_ret_mkt_list.extend(pool.starmap(epochs_sim_rot_pair_data,
                                             args_prod))

    # Flatten the list
    agg_ret_mkt_list_flat: List[float] = \
        [val for sublist in agg_ret_mkt_list for val in sublist]
    agg_ret_mkt_series: pd.Series = pd.Series(agg_ret_mkt_list_flat)

    print(f'mean = {agg_ret_mkt_series.mean()}')
    print(f'std  = {agg_ret_mkt_series.std()}')

    # Saving data
    pickle.dump(agg_ret_mkt_series, open(
        f'../data/epochs_sim/epochs_sim_{kind}_agg_dist_ret_market_data_long'
        + f'_win_{window}_K_{K_value}.pickle', 'wb'), protocol=4)

    del agg_ret_mkt_list
    del agg_ret_mkt_series

# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """

    K_value = 200
    windows = [100, 55, 40, 25, 10]

    #################
    # Code to create complete time series of simulated returns.
    # Returns
    # Gaussian
    ret_gauss: pd.DataFrame = returns_simulation_gaussian(0.3, K_value, 8000)
    print(ret_gauss.head())
    pickle.dump(ret_gauss, open(
        f'../data/epochs_sim/returns_simulation_gaussian_K_{K_value}.pickle',
        'wb'), protocol=4)
    # Algebraic
    ret_alg: pd.DataFrame = \
        returns_simulation_algebraic(0.3, K_value, 8000, 10)
    print(ret_alg.head())
    pickle.dump(ret_gauss, open(
        f'../data/epochs_sim/returns_simulation_algebraic_K_{K_value}.pickle',
        'wb'), protocol=4)
    ################

    for win in windows:
        epochs_sim_rot_market_data('gaussian', K_value, win)
        epochs_sim_rot_market_data('algebraic', K_value, win)

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
