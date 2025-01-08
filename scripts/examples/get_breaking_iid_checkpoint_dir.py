#!/usr/bin/env python

import pathlib

from specialcouscous.utils.slurm import find_checkpoint_dir_and_uuid

if __name__ == "__main__":
    base_path = pathlib.Path(
        "/hkfs/work/workspace/scratch/ku4408-SpecialCouscous/results/"
    )
    log_n_samples = 6
    log_n_features = 4
    n_classes = 10
    mu_global = 0.5
    mu_local = 2.0
    n_trees = 800
    data_seed = 0
    model_seed = 1

    checkpoint_dir, uuid = find_checkpoint_dir_and_uuid(
        base_path=base_path,
        log_n_samples=log_n_samples,
        log_n_features=log_n_features,
        mu_global=mu_global,
        mu_local=mu_local,
        data_seed=data_seed,
        model_seed=model_seed,
    )
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"UUID: {uuid}")
