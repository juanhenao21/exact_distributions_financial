'''Epochs simulation main module.

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
'''

# -----------------------------------------------------------------------------
# Modules

from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import epochs_sim_analysis
import epochs_sim_plot
import epochs_sim_tools

# -----------------------------------------------------------------------------


def data_plot_generator() -> None:
    """Generates all the analysis and plots from the data.

    :return: None -- The function saves the data in a file and does not return
     a value.
    """

    for epochs_len in [10, 25, 40, 55]:
        x = epochs_sim_analysis \
            .epochs_sim_agg_returns_market_data(0.3, 2, 100, 40, epochs_len)
        epochs_sim_plot.epochs_sim_agg_returns_market_plot(x, epochs_len)

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

    print('Ay vamos!!!')

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
