#!/usr/bin/env python
import logging
import pathlib

from mpi4py import MPI

from specialcouscous.train import train_serial
from specialcouscous.utils import parse_arguments, set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.


def run_serial(config):
    synthetic_data_config = {
        "n_samples": config.n_samples,
        "n_features": config.n_features,
        "n_classes": config.n_classes,
        "make_classification_kwargs": {
            "n_clusters_per_class": config.n_clusters_per_class,
            "n_informative": int(config.frac_informative * config.n_features),
            "n_redundant": int(config.frac_redundant * config.n_features),
            "flip_y": config.flip_y,
        }
    }
    shared_config = {
        "random_state": config.random_state,
        "random_state_model": config.random_state_model,
        "train_split": config.train_split,
        "stratified_train_test": config.stratified_train_test,
        "n_trees": config.n_trees,
        "detailed_evaluation": config.detailed_evaluation,
        "output_dir": config.output_dir,
        "output_label": config.output_label,
        "experiment_id": config.experiment_id,
        "save_model": config.save_model,
    }

    if config.dataset_name is None:  # synthetic data
        train_serial.train_serial_on_synthetic_data(**synthetic_data_config, **shared_config)
    else:  # named dataset
        train_serial.train_serial_on_dataset(dataset=config.dataset_name, **shared_config)


if __name__ == "__main__":
    # Parse command-line arguments.
    args = parse_arguments()
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
