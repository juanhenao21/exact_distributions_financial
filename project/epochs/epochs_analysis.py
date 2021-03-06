'''Epochs analysis module.

The functions in the module use local normalization to compute the returns, the
normalized returns and the correlation matrix of financial time series.

This script requires the following modules:
    * itertools
    * math
    * multiprocessing
    * pickle
    * typing
    * numpy
    * pandas
    * epochs_tools

The module contains the following functions:
    * returns_data - computes the returns of the time series.
    * epochs_volatility_data - uses local normalization to compute the
      volatility of the time series.
    * epochs_normalized_returns_data - uses rolling normalization to normalize
      the returns of the time series.
    * epochs_correlation_matrix_data - uses local normalization to compute the
      correlation matrix of the normalized returns.
    * epochs_aggregated_dist_returns_pair_data - uses local normalization to
      compute the aggregated distribution of returns for a pair of stocks.
    * epochs_aggregated_dist_returns_market_data - uses local normalization to
      compute the aggregated distribution of returns for a market.
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

import epochs_tools

# -----------------------------------------------------------------------------


def returns_data(dates: List[str], time_step: str) -> None:
    """Computes the returns of the time series.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-01']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', 'wk',
     '1mo').
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    function_name: str = returns_data.__name__
    epochs_tools \
        .function_header_print_data(function_name, dates, time_step, '', '')

    try:

        # Load data
        data: pd.DataFrame = pickle.load(open(
            f'../data/original_data/original_data_{dates[0]}_{dates[1]}_step'
            + f'_{time_step}.pickle', 'rb'))

        returns_df: pd.DataFrame = data.pct_change().dropna()

        # Saving data
        epochs_tools \
            .save_data(returns_df, function_name, dates, time_step, '', '')

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def epochs_volatility_data(dates: List[str], time_step: str,
                           window: str) -> None:
    """Uses local normalization to compute the volatility of the time series.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    function_name: str = epochs_volatility_data.__name__
    epochs_tools \
        .function_header_print_data(function_name, dates, time_step, window)

    try:

        # Load data
        data: pd.DataFrame = pickle.load(open(
            f'../data/epochs/returns_data_{dates[0]}_{dates[1]}_step'
            + f'_{time_step}_win_.pickle', 'rb'))

        std_df: pd.DataFrame = data.rolling(window=int(window)).std().dropna()

        # Saving data
        epochs_tools.save_data(std_df, function_name, dates, time_step, window)

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def epochs_normalized_returns_data(dates: List[str], time_step: str,
                                   window: str) -> None:
    """Uses rolling normalization to normalize the returns of the time series.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', 'wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '60').
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    function_name: str = epochs_normalized_returns_data.__name__
    epochs_tools \
        .function_header_print_data(function_name, dates, time_step, window)

    try:

        # Load data
        data: pd.DataFrame = pickle.load(open(
            f'../data/epochs/returns_data_{dates[0]}_{dates[1]}_step'
            + f'_{time_step}_win_.pickle', 'rb'))

        data_win = data.iloc[int(window) - 1:]
        data_mean = data.rolling(window=int(window)).mean().dropna()
        data_std = data.rolling(window=int(window)).std().dropna()

        normalized_df: pd.DataFrame = (data_win - data_mean) / data_std

        # Saving data
        epochs_tools \
            .save_data(normalized_df, function_name, dates, time_step, window)

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# -----------------------------------------------------------------------------


def epochs_correlation_matrix_data(dates: List[str], time_step: str,
                                   window: str) -> None:
    """uses local normalization to compute the correlation matrix of the
       normalized returns.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :param window: window time to compute the volatility (i.e. '25').
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    function_name: str = epochs_correlation_matrix_data.__name__
    epochs_tools \
        .function_header_print_data(function_name, dates, time_step, window)

    try:

        # Load data
        data: pd.DataFrame = pickle.load(open(
            f'../data/epochs/epochs_normalized_returns_data_{dates[0]}'
            + f'_{dates[1]}_step_{time_step}_win_{window}.pickle', 'rb'))

        corr_matrix_df: pd.DataFrame = data.corr()

        # Saving data
        epochs_tools \
            .save_data(corr_matrix_df, function_name, dates, time_step, window)

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

    except TypeError as error:
        print('To compute the correlation is needed at least to stocks')
        print(error)
        print()

# ----------------------------------------------------------------------------


def epochs_aggregated_dist_returns_pair_data(dates: List[str], time_step: str,
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

            cov_two_col: pd.DataFrame = local_data_df.cov()
            # eig_vec:  eigenvector, eig_val: eigenvalues
            eig_val, eig_vec = np.linalg.eig(cov_two_col)

            # rot: rotation, scal: scaling
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
        agg_ret_mkt_list = [x for x in agg_ret_mkt_list if x <= 10
                            and x >= -10]

        return agg_ret_mkt_list

    except FileNotFoundError as error:
        print('No data')
        print(error)
        print()

# ----------------------------------------------------------------------------


def epochs_aggregated_dist_returns_market_data(dates: List[str],
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

    function_name: str = epochs_aggregated_dist_returns_market_data.__name__
    epochs_tools \
        .function_header_print_data(function_name, dates, time_step, window,
                                    K_value)

    try:

        # Load name of the stocks
        if K_value == 'all':
            stocks_name: pd.DataFrame = pickle.load(open(
                f'../data/epochs/returns_data_{dates[0]}_{dates[1]}_step'
                + f'_{time_step}_win__K_.pickle', 'rb'))

        else:
            stocks_name: pd.DataFrame = pickle.load(open(
                f'../data/epochs/returns_data_{dates[0]}_{dates[1]}_step'
                + f'_{time_step}_win__K_.pickle', 'rb')).sample(n=int(K_value),
                                                             axis='columns')

        agg_ret_mkt_list: List[List[float]] = []

        # Combination of stock pairs
        stocks_comb: Iterable[Tuple[Any, ...]] = icomb(stocks_name, 2)
        args_prod: Iterable[Any] = iprod([dates], [time_step], stocks_comb,
                                         [window])

        # Parallel computing
        with mp.Pool(processes=mp.cpu_count()) as pool:
            agg_ret_mkt_list.extend(pool.starmap(
                epochs_aggregated_dist_returns_pair_data, args_prod))

        # Flatten the list
        agg_ret_mkt_list_flat: List[float] = \
            [val for sublist in agg_ret_mkt_list for val in sublist]
        agg_ret_mkt_series: pd.Series = pd.Series(agg_ret_mkt_list_flat)

        print(f'mean = {agg_ret_mkt_series.mean()}')
        print(f'std  = {agg_ret_mkt_series.std()}')

        # Saving data
        epochs_tools.save_data(agg_ret_mkt_series, function_name, dates,
                               time_step, window, K_value)

        del agg_ret_mkt_list
        del agg_ret_mkt_series

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

    dates_1m = ['2021-07-19', '2021-08-07']
    dates_1h = ['2021-06-01', '2021-07-31']
    dates = ['1990-01-01', '2020-12-31']

    win = '10'

    epochs_aggregated_dist_returns_market_data(dates_1h, '1h', win, '10')

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
