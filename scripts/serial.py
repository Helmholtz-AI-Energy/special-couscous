#!/usr/bin/env python
import logging
import pathlib

from mpi4py import MPI

from specialcouscous.train import train_serial
from specialcouscous.utils import parse_cli_args, set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.


def run_serial(config):
    synthetic_data_config = parse_cli_args.get_synthetic_data_kwargs(config)
    shared_config = parse_cli_args.get_general_run_kwargs(config)

    if config.dataset_name is None:  # synthetic data
        train_serial.train_serial_on_synthetic_data(**synthetic_data_config, **shared_config)
    else:  # named dataset
        train_serial.train_serial_on_dataset(dataset=config.dataset_name, **shared_config)


if __name__ == "__main__":
    # Parse command-line arguments.
    args = parse_cli_args.parse_arguments()
    set_logger_config(level=args.logging_level, log_file=f"{args.log_path}/{pathlib.Path(__file__).stem}.log")
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        log.info(
            "************************\n"
            "* Serial Random Forest *\n"
            "************************\n"
            f"Hyperparameters used are:\n{args}"
        )
    run_serial(args)
