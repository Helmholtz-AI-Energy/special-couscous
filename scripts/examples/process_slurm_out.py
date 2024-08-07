#!/usr/bin/env python
"""Exemplary script for processing slurm output files."""

import pathlib

import pandas as pd

from specialcouscous.utils import dataframe_from_slurm_output
from specialcouscous.utils.plot import (
    plot_single_node_capacity,
    plot_times_and_accuracy,
)

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", None)

exp_path = pathlib.Path(
    "../exps/single_node/slurm"
)  # Set path to SLURM job output files.

df = dataframe_from_slurm_output(
    exp_path
)  # Create a dataframe from SLURM output files.
plot_single_node_capacity(
    df, exp_path
)  # Plot results from single-node capacity experiments.
plot_times_and_accuracy(
    df, exp_path
)  # Plot training time and accuracy from single-node capacity experiments.
