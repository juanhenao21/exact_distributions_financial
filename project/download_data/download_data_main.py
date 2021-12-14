"""Exact distributions financial download data main module.

The functions in the module download data from Yahoo! Finance for several years
in several intervals.

This script requires the following modules:
    * typing
    * datetime
    * pandas
    * yfinance
    * download_data_tools

The module contains the following functions:
    * portfolio_download_data - download data of a ticker.
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
"""

# -----------------------------------------------------------------------------
# Modules

from typing import List

from datetime import datetime as dt
import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore

import download_data_tools

# -----------------------------------------------------------------------------


def exact_distributions_download_data(
    tickers: List[str], dates: List[str], time_step: str
) -> None:
    """Downloads the prices of a ticker for a time interval in a time step.

    :param tickers: list of the string abbreviation of the stocks to be
     analyzed (i.e. ['AAPL', 'MSFT']).
    :param dates: List of the interval of dates to be analyzed
     (i.e. ['1980-01-01', '2020-12-31']).
    :param time_step: time step of the data (i.e. '1m', '1h', '1d', '1wk',
     '1mo').
    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    try:
        function_name: str = exact_distributions_download_data.__name__
        download_data_tools.function_header_print_data(
            function_name, tickers, dates, time_step
        )

        init_year = int(dates[0].split("-")[0])
        init_month = int(dates[0].split("-")[1])
        init_day = int(dates[0].split("-")[2])
        fin_year = int(dates[1].split("-")[0])
        fin_month = int(dates[1].split("-")[1])
        fin_day = int(dates[1].split("-")[2])
        # last_day = monthrange(fin_year, fin_month)[1]

        init_date: dt = dt(year=init_year, month=init_month, day=init_day)
        fin_date: dt = dt(year=fin_year, month=fin_month, day=fin_day)

        if time_step == "1wk":
            time_step = "1d"
            time_step_ori = "1wk"
        elif time_step == "1mo":
            time_step = "1d"
            time_step_ori = "1mo"
        else:
            time_step_ori = time_step

        # Not all the periods can be combined with the time steps.
        raw_data: pd.DataFrame = yf.download(
            tickers=tickers, start=init_date, end=fin_date, interval=time_step
        )["Adj Close"]
        # Order DataFrame columns by sector
        raw_data = raw_data[tickers]

        if raw_data.isnull().values.any():
            # Remove stocks that do not have data from the initial date
            raw_data = raw_data.dropna(axis=1, thresh=len(raw_data) - 10).fillna(
                method="ffill"
            )

        if time_step == "1m":
            raw_data.index = raw_data.index.strftime("%Y-%m-%d %H:%M")
        elif time_step == "1h":
            raw_data.index = raw_data.index.strftime("%Y-%m-%d %H")
        else:
            raw_data.index = pd.to_datetime(raw_data.index.strftime("%Y-%m-%d"))
            raw_data = raw_data.resample("B").ffill()

        if time_step_ori == "1wk":
            offset = pd.tseries.frequencies.to_offset("-6d")
            raw_data = raw_data.resample("W-MON").first()
            raw_data.index = raw_data.index + offset
        elif time_step_ori == "1mo":
            raw_data = raw_data.resample("BM").last()

        download_data_tools.save_data(raw_data, dates, time_step_ori)

    except AssertionError as error:
        print("No data")
        print(error)


# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function extract, analyze and plot the data.

    :return: None.
    """

    download_data_tools.initial_message()

    # S&P 500 companies, initial year and time step
    stocks: List[str] = download_data_tools.get_stocks(["all"])
    # Stocks used in the paper "non-stationarity in financial time series"
    # stocks: List[str] = ['MMM', 'ABT', 'ADBE', 'AMD', 'AES', 'AET', 'AFL',
    #                      'APD', 'ARG', 'AA', 'AGN', 'ALTR', 'MO', 'AEP', 'AXP',
    #                      'AIG', 'AMGN', 'APH', 'APC', 'ADI', 'AON', 'APA',
    #                      'AAPL', 'AMAT', 'ADM', 'T', 'ADSK', 'ADP', 'AZO',
    #                      'AVY', 'AVP', 'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BAX',
    #                      'BBT', 'BDX', 'BMS', 'BBY', 'BIG', 'BIIB', 'HRB',
    #                      'BMC', 'BA', 'BMY', 'BF.B', 'CA', 'COG', 'CPB', 'CAH',
    #                      'CCL', 'CAT', 'CELG', 'CNP', 'CTL', 'CERN', 'SCHW',
    #                      'CVX', 'CB', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CLF',
    #                      'CLX', 'CMS', 'KO', 'CCE', 'CL', 'CMCSA', 'CMA',
    #                      'CSC', 'CAG', 'COP', 'ED', 'GLW', 'COST', 'CVH',
    #                      'CSX', 'CMI', 'CVS', 'DHR', 'DE', 'DELL', 'XRAY', 'D',
    #                      'RRD', 'DOV', 'DOW', 'DTE', 'DD', 'DUK', 'DNB', 'ETN',
    #                      'ECL', 'EIX', 'EMC', 'EMR', 'ETR', 'EOG', 'EQT',
    #                      'EFX', 'EXC', 'XOM', 'FDO', 'FAST', 'FDX', 'FITB',
    #                      'FHN', 'FISV', 'FLS', 'FMC', 'F', 'FRX', 'BEN', 'GCI',
    #                      'GPS', 'GD', 'GE', 'GIS', 'GPC', 'GT', 'GWW', 'HAL',
    #                      'HOG', 'HRS', 'HAS', 'HCP', 'HNZ', 'HP', 'HES', 'HPQ',
    #                      'HD', 'HON', 'HRL', 'HST', 'HUM', 'HBAN', 'ITW',
    #                      'TEG', 'INTC', 'IBM', 'IFF', 'IGT', 'IP', 'IPG',
    #                      'JEC', 'JNJ', 'JCI', 'JPM', 'K', 'KEY', 'KMB', 'KIM',
    #                      'KLAC', 'KR', 'LH', 'LM', 'LEG', 'LEN', 'LUK', 'LLY',
    #                      'LTD', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LSI', 'MTB',
    #                      'MRO', 'MMC', 'MAS', 'MAT', 'MKC', 'MCD', 'MHP',
    #                      'MWV', 'MDT', 'MRK', 'MU', 'MSFT', 'MOLX', 'TAP',
    #                      'MSI', 'MUR', 'MYL', 'NBR', 'NWL', 'NEM', 'NEE',
    #                      'GAS', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS',
    #                      'NOC', 'NU', 'NUE', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI',
    #                      'PCAR', 'IR', 'PLL', 'PH', 'PAYX', 'JCP', 'PBCT',
    #                      'POM', 'PEP', 'PFE', 'PCG', 'PNW', 'PBI', 'PCL',
    #                      'PNC', 'PPG', 'PPL', 'PCP', 'PG', 'PGR', 'PSA', 'PHM',
    #                      'QCOM', 'RSH', 'RTN', 'RF', 'ROK', 'ROST', 'RDC', 'R',
    #                      'SWY', 'SCG', 'SLB', 'SEE', 'SHW', 'SIAL', 'SLM',
    #                      'SNA', 'SO', 'LUV', 'SWN', 'S', 'STJ', 'SWK', 'SPLS',
    #                      'HOT', 'STT', 'SYK', 'SUN', 'STI', 'SVU', 'SYMC',
    #                      'SYY', 'TROW', 'TGT', 'TE', 'TLAB', 'THC', 'TER',
    #                      'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO', 'TIF',
    #                      'TJX', 'TMK', 'TSN', 'TYC', 'USB', 'UNP', 'UNH', 'X',
    #                      'UTX', 'VFC', 'VLO', 'VAR', 'VZ', 'VNO', 'VMC', 'WMT',
    #                      'WAG', 'DIS', 'WPO', 'WM', 'WFC', 'WDC', 'WY', 'WHR',
    #                      'WFM', 'WMB', 'WEC', 'XEL', 'XRX', 'XLNX', 'XL',
    #                      'ZION']

    dates_1m: List[str] = ["2021-08-09", "2021-08-14"]
    dates_1h: List[str] = ["2021-07-31", "2021-10-01"]
    dates_1d: List[str] = ["1990-01-01", "2020-12-31"]
    dates_1wk: List[str] = ["1990-01-01", "2020-12-31"]
    dates_1mo: List[str] = ["1990-01-01", "2020-12-31"]
    time_step_1m: str = "1m"
    time_step_1h: str = "1h"
    time_step_1d: str = "1d"
    time_step_1wk: str = "1wk"
    time_step_1mo: str = "1mo"

    # Basic folders
    download_data_tools.start_folders()

    # Run analysis
    # Download data
    # exact_distributions_download_data(stocks, dates_1m, time_step_1m)
    exact_distributions_download_data(stocks, dates_1h, time_step_1h)
    # exact_distributions_download_data(stocks[:180], dates_1d, time_step_1d)
    # exact_distributions_download_data(stocks[:180], dates_1wk, time_step_1wk)
    # exact_distributions_download_data(stocks[:180], dates_1mo, time_step_1mo)

    # import pickle

    # x = pickle.load(open('../data/original_data/original_data_2021-07-19_2021-07-24_step_1m.pickle', 'rb'))
    # y = pickle.load(open('../data/original_data/original_data_2021-07-26_2021-07-31_step_1m.pickle', 'rb'))
    # z = pickle.load(open('../data/original_data/original_data_2021-08-02_2021-08-07_step_1m.pickle', 'rb'))
    # r = pickle.load(open('../data/original_data/original_data_2021-08-09_2021-08-14_step_1m.pickle', 'rb'))

    # print(x.shape)
    # print(y.shape)
    # print(z.shape)
    # print(r.shape)

    # w = pd.concat([x, y, z, r])

    # w = w.dropna(axis=1)
    # print(w.head())
    # print(w.tail())
    # print(w.shape)

    # pickle.dump(w, open('../data/original_data/original_data_2021-07-19_2021-08-14_step_1m.pickle', 'wb'))

    print("Ay vamos!!!")


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
