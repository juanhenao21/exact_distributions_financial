"""Exact distributions covariance analysis module.

The functions in the module compute the returns, and the aggregated returns of
financial time series.

This script requires the following modules:
    * pickle
    * typing
    * numpy
    * pandas
    * exact_distributions_covariance_tools

The module contains the following functions:
    * returns_data - computes the returns of the time series.
    * aggregated_dist_returns_market_data - computes the aggregated
      distribution of returns for a market.
    * main - the main function of the script.

..moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
"""

# -----------------------------------------------------------------------------
# Modules

import pickle
from typing import List

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import exact_distributions_covariance_tools

# -----------------------------------------------------------------------------


def returns_data(dates: List[str], time_step: str) -> None:
    """Computes the returns of the time series.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    function_name: str = returns_data.__name__
    exact_distributions_covariance_tools.function_header_print_data(
        function_name, dates, time_step
    )

    try:

        # Load data
        data: pd.DataFrame = pickle.load(
            open(
                f"../data/original_data/original_data_{dates[0]}_{dates[1]}_step"
                + f"_{time_step}.pickle",
                "rb",
            )
        )

        returns_df: pd.DataFrame = data.pct_change().dropna()

        # Saving data
        exact_distributions_covariance_tools.save_data(
            returns_df, function_name, dates, time_step
        )

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()


# ----------------------------------------------------------------------------


def aggregated_dist_returns_market_data(dates: List[str], time_step: str) -> None:
    """Computes the aggregated distribution of returns for a market.

    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01', '2020-12']).
    :param time_step: time step of the data (i.e. '1m', '2m', '5m', ...).
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    function_name: str = aggregated_dist_returns_market_data.__name__
    exact_distributions_covariance_tools.function_header_print_data(
        function_name, dates, time_step
    )

    try:

        # Load data
        returns_vals: pd.DataFrame = pickle.load(
            open(
                f"../data/exact_distributions_covariance/returns_data_{dates[0]}"
                + f"_{dates[1]}_step_{time_step}.pickle",
                "rb",
            )
        )

        print("Size of time series and number of companies: ", returns_vals.shape)

    except FileNotFoundError as error:
        print("No data")
        print(error)
        print()

    # Normalization mean 0 std 1
    returns_vals = (returns_vals - returns_vals.mean()) / returns_vals.std()

    cov: pd.DataFrame = returns_vals.cov()
    # eig_vec:  eigenvector, eig_val: eigenvalues
    eig_val, eig_vec = np.linalg.eig(cov)

    # rot: rotation, scal: scaling
    rot, scale = eig_vec, np.diag(1 / np.sqrt(eig_val))
    # trans: transformation matrix
    # trans = rot . scal
    trans = rot.dot(scale)

    # Transform the returns
    trans_returns: pd.DataFrame = returns_vals.dot(trans)
    # Name of the columns with the used stocks
    trans_returns.columns = returns_vals.columns

    one_col: List[pd.Series] = []

    for col in trans_returns.columns:

        one_col.append(trans_returns[col])

    agg_returns = pd.concat(one_col, ignore_index=True)

    print(f"Mean      {dates} = {agg_returns.mean()}")
    print(f"Std. Dev. {dates} = {agg_returns.std()}")

    # Saving data
    exact_distributions_covariance_tools.save_data(
        agg_returns, function_name, dates, time_step
    )

    del returns_vals
    del trans_returns
    del agg_returns
    del one_col


# ----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function is used to test the functions in the script.

    :return: None.
    """


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
