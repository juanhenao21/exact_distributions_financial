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
                                       normalized: bool = False) -> pd.Series:
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
    :return: simulated rotated and aggregated returns for a simulated market.
    :rtype: pd.Series
    """

    function_name: str = epochs_sim_agg_returns_market_data.__name__
    epochs_sim_tools \
        .function_header_print_data(function_name, [''], '', '', '', sim=True)

    agg_ret_mkt_list: List[float] = []

    for _ in range(K_values):
        for _ in range(epochs_num):

            returns: pd.DataFrame = \
                returns_simulation(out_diag_val, size_corr_matrix, epochs_len)
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


def epochs_sim_no_rot_pair_data(dates: List[str], time_step: str,
                                cols: List[str],
                                window: str) -> List[float]:
    """Uses local normalization to compute the aggregated distribution of
       returns for a pair of stocks.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param cols: pair of stocks to be analized (i. e. ['AAPL', 'MSFT']).
    :param window: window time to compute the volatility (i.e. '25').
    :return: List[float] -- The function returns a list with float numbers.
    """

    try:

        # Load data
        two_col: pd.DataFrame = pickle.load(open(
            f'../data/epochs/returns_data_{dates[0]}_{dates[1]}_step'
            + f'_{time_step}_win__K_.pickle', 'rb'))[[cols[0], cols[1]]]

        # List to extend with the returns values of each pair
        agg_ret_mkt_list: List[float] = []

        # Add the index as a column to group the return values
        two_col['DateCol'] = two_col.index
        # Add a column grouping the returns in the time window
        if time_step == '1m':
            two_col['DateCol'] = pd.to_datetime(two_col['DateCol'])
            two_col['Group'] = two_col.groupby(
                pd.Grouper(key='DateCol', freq=window + 'T'))['DateCol'] \
                .transform('first')
        elif time_step == '1h':
            two_col['DateCol'] = pd.to_datetime(two_col['DateCol'])
            two_col['Group'] = np.arange(len(two_col)) // int(window)
        elif time_step == '1d':
            two_col['Group'] = two_col.groupby(
                pd.Grouper(key='DateCol', freq=window + 'B'))['DateCol'] \
                .transform('first')
        elif time_step == '1wk':
            two_col['Group'] = two_col.groupby(
                pd.Grouper(key='DateCol', freq=window + 'W-WED'))['DateCol'] \
                .transform('first')
            two_col = two_col.drop(pd.Timestamp('1990-01-09'))
        elif time_step == '1mo':
            two_col['Group'] = two_col.groupby(
                pd.Grouper(key='DateCol', freq=window + 'BM'))['DateCol'] \
                .transform('first')
            two_col = two_col.drop(pd.Timestamp('1990-02-28'))
        else:
            raise Exception('There is something wrong with the time_step!')

        # Remove groups that are smaller than the window to avoid linalg errors
        two_col = two_col.groupby('Group') \
            .filter(lambda x: len(x) >= int(window) - 5)

        for local_data in two_col.groupby(by=['Group']):

            # Use the return columns
            local_data_df: pd.DataFrame = local_data[1][[cols[0], cols[1]]]

            local_data_df = \
                (local_data_df - local_data_df.mean()) / local_data_df.std()

            one_col = local_data_df[cols[0]].append(local_data_df[cols[1]],
                                                    ignore_index=True)

            agg_ret_mkt_list.extend(one_col)

            del local_data_df
            del one_col

        del two_col

        # remove NaN and Inf
        agg_ret_mkt_list = [x for x in agg_ret_mkt_list if not math.isnan(x)
                            and not math.isinf(x)]
        # filter out values greater than 10 or smaller than -10
        agg_ret_mkt_list = [x for x in agg_ret_mkt_list if -10 <= x <= 10]

        return agg_ret_mkt_list

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()
        return [0]

# ----------------------------------------------------------------------------


def epochs_sim_no_rot_market_data(dates: List[str],
                                  time_step: str,
                                  window: str,
                                  K_value: str) -> None:
    """Computes the aggregated distribution of returns for a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :param K_value: number of companies to be used (i.e. '80', 'all').
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    function_name: str = epochs_sim_no_rot_market_data.__name__
    epochs_sim_tools \
        .function_header_print_data(function_name, dates, time_step, window,
                                    K_value)

    try:

        # Load name of the stocks
        if K_value == 'all':
            stocks_name: pd.DataFrame = pickle.load(open(
                f'../data/epochs/returns_data_{dates[0]}_{dates[1]}_step'
                + f'_{time_step}_win__K_.pickle', 'rb'))

        else:
            stocks_name = pickle.load(open(
                f'../data/epochs/returns_data_{dates[0]}_{dates[1]}_step'
                + f'_{time_step}_win__K_.pickle', 'rb'))\
                .sample(n=int(K_value), axis='columns')

        agg_ret_mkt_list: List[List[float]] = []

        # Combination of stock pairs
        stocks_comb: Iterable[Tuple[Any, ...]] = icomb(stocks_name, 2)
        args_prod: Iterable[Any] = iprod([dates], [time_step], stocks_comb,
                                         [window])

        # Parallel computing
        with mp.Pool(processes=mp.cpu_count()) as pool:
            agg_ret_mkt_list.extend(pool.starmap(
                epochs_sim_no_rot_pair_data, args_prod))

        # Flatten the list
        agg_ret_mkt_list_flat: List[float] = \
            [val for sublist in agg_ret_mkt_list for val in sublist]
        agg_ret_mkt_series: pd.Series = pd.Series(agg_ret_mkt_list_flat)

        print(f'mean = {agg_ret_mkt_series.mean()}')
        print(f'std  = {agg_ret_mkt_series.std()}')

        # Saving data
        epochs_sim_tools.save_data(agg_ret_mkt_series, function_name, dates,
                                   time_step, window, K_value)

        del agg_ret_mkt_list
        del agg_ret_mkt_series

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
