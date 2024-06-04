import pandas as pd

from utils import (
    dataframe_from_slurm_output,
    plot_single_node_capacity,
    plot_times_and_accuracy,
)

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", None)

exp_path = "../exps/single_node/slurm"

df = dataframe_from_slurm_output(exp_path)
plot_single_node_capacity(df, exp_path)
plot_times_and_accuracy(df, exp_path)
