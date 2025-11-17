#!/usr/bin/env python
import logging
import pathlib

from mpi4py import MPI

from specialcouscous.train import train_parallel
from specialcouscous.utils import parse_cli_args, set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.


def run_parallel(config, comm):
    synthetic_data_config = parse_cli_args.get_synthetic_data_kwargs(config)
    shared_config = parse_cli_args.get_general_run_kwargs(config)

    imbalanced_synthetic_data_config = {
        "globally_balanced": not config.globally_imbalanced,
        "locally_balanced": not config.locally_imbalanced,
        "shared_test_set": config.shared_test_set,
        "mu_partition": config.mu_partition,
        "mu_data": config.mu_data,
        "peak": config.peak,
        "enforce_constant_local_size": config.enforce_constant_local_size,
    }
    shared_config = {
        **shared_config,
        "mpi_comm": comm,
        "shared_global_model": config.shared_global_model,
    }

    if config.dataset_name is None:  # synthetic data
        if (
            config.globally_imbalanced or config.locally_imbalanced
        ):  # allow data imbalance
            train_parallel.train_parallel_on_synthetic_data(
                **synthetic_data_config,
                **imbalanced_synthetic_data_config,
                **shared_config,
            )
        else:  # balanced dataset
            train_parallel.train_parallel_on_balanced_synthetic_data(
                **synthetic_data_config, **shared_config
            )
    else:  # named dataset
        train_parallel.train_parallel_on_dataset(
            dataset=config.dataset_name, **shared_config
        )


if __name__ == "__main__":
    # Parse command-line arguments.
    args = parse_cli_args.parse_arguments()
    set_logger_config(
        level=args.logging_level,
        log_file=f"{args.log_path}/{pathlib.Path(__file__).stem}.log",
    )
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        log.info(
            "*****************************\n"
            "* Distributed Random Forest *\n"
            "*****************************\n"
            f"Hyperparameters used are:\n{args}"
        )
    run_parallel(args, comm)
