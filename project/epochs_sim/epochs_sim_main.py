"""Epochs simulation main module.

The functions in the module compute and plot simulated returns and their
rotation and aggregation. They also compute and plot key results from other
methods.

This script requires the following modules:
    * typing
    * epochs_sim_analysis
    * epochs_sim_plot
    * epochs_sim_tools

The module contains the following functions:
    * data_plot_generator
    * main - the main function of the script.

.. moduleauthor:: Juan Camilo Henao Londono <www.github.com/juanhenao21>
"""

# -----------------------------------------------------------------------------
# Modules

from typing import List

import epochs_sim_analysis
import epochs_sim_plot
import epochs_sim_tools

# -----------------------------------------------------------------------------


def data_plot_generator() -> None:
    """Generates all the analysis and plots from the data.

    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    # Simulate the aggregated returns for different epochs lenghts
    for epochs_len in [10, 25, 40, 55, 500]:
        returns_pairs = epochs_sim_analysis.epochs_sim_agg_returns_market_data(
            0.3, 2, 50, 40, epochs_len, kind="gaussian", normalized=True
        )
        epochs_sim_plot.epochs_sim_agg_returns_market_plot(
            returns_pairs, epochs_len, 50, kind="gaussian"
        )
        returns_pairs = epochs_sim_analysis.epochs_sim_agg_returns_market_data(
            0.3, 2, 50, 40, epochs_len, kind="algebraic", normalized=True
        )
        epochs_sim_plot.epochs_sim_agg_returns_market_plot(
            returns_pairs, epochs_len, 50, kind="algebraic"
        )

    # for epochs_len in [10, 25, 100]:
    #     returns_market = epochs_sim_analysis \
    #         .returns_simulation(0.3, 250, epochs_len)
    #     agg_ret_mkt = epochs_sim_analysis \
    #         .epochs_sim_agg_returns_cov_market_data(returns_market)
    #     epochs_sim_plot \
    #         .epochs_sim_agg_returns_cov_market_plot(agg_ret_mkt, epochs_len)

    # Initial year and time step

    # dates = [['1990-01-01', '2020-12-31']]
    # time_step = ['1d']
    # windows: List[str] = ['10', '25', '40', '55']

    # for window in windows:
    #     for idx, val in enumerate(dates):

    #         # epochs_sim_analysis \
    #         #     .epochs_sim_no_rot_market_data(dates[idx],
    #         #                                    time_step[idx],
    #         #                                    window, 50)
    #         epochs_sim_plot \
    #             .epochs_aggregated_dist_returns_market_plot(val,
    #                                                         time_step[idx],
    #                                                         window, '50')


# -----------------------------------------------------------------------------


def main() -> None:
    """The main function of the script.

    The main function extract, analyze and plot the data.

    :return: None.
    """

    epochs_sim_tools.initial_message()

    # Basic folders
    epochs_sim_tools.start_folders()

    # Run analysis
    # Analysis and plot
    data_plot_generator()

    print("Ay vamos!!!")


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
