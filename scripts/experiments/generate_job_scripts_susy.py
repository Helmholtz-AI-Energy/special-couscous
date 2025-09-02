import itertools
import math
import pathlib
from typing import Any

from generate_job_scripts_redo_runtime_measurements import (
    MAX_TIME,
    NODES,
    generate_job_script,
)

BASE_N_TREES = {
    "strong": [1000],
    "weak": [1000],
    "inference": [100],
}

SERIAL_BASELINE_TIMES = {  # in minutes TODO: update with actual values
    1000: 30,
    100: 10,
}

OVERESTIMATION_FACTOR = 2  # overestimate time limit by how much from expected time


SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}         # Job name
#SBATCH --partition={partition}       # Queue for resource allocation
#SBATCH --time={time}                 # Wall-clock time limit
#SBATCH --mem={mem}                   # Main memory
#SBATCH --cpus-per-task=76            # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --nodes={n_nodes}             # Number of nodes
#SBATCH --ntasks-per-node=1           # One MPI rank per node
#SBATCH --account={project}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

# Unload all currently loaded modules and load required modules.
ml purge
ml load compiler/llvm
ml load mpi/openmpi/4.1

# Setup paths
SCRIPT="{script_dir}/{script}"
RESDIR="{result_dir}_${{SLURM_JOB_ID}}"

# Activate venv
source {venv}/bin/activate

mkdir -p "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun python -u ${{SCRIPT}} \\
    --n_trees {n_trees} \\
    --random_state {random_state_data} \\
    --random_state_model {random_state_model} \\
    --detailed_evaluation \\
    --save_model \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --log_path ${{RESDIR}} {additional_args}
"""


def generate_strong_scaling_job_scripts(
    global_config: dict[str, Any],
    data_seeds: list[int],
    model_seeds: list[int],
    comm_sizes: list[int],
    result_base_dir: pathlib.Path,
    base_job_script_path: pathlib.Path,
) -> None:
    """
    Generate all strong scaling job scripts.

    Parameters
    ----------
    global_config : dict[str, Any]
        The global job script configuration. Should contain the following keys: project, script_dir, venv, n_classes,
        script, mem, additional_args, partition. All other keys may be overwritten.
    data_seeds : list[int]
        The list of data seeds.
    model_seeds : list[int]
        The list of model seeds.
    comm_sizes : list[int]
        The list comm sizes (node counts).
    result_base_dir : pathlib.Path
        The base directory to write the job results to (subdirectories for experiment and run config will be created).
    base_job_script_path : pathlib.Path
        The base directory to write the job scripts to (subdirectories for experiment and run config will be created).
    """
    for n_trees, data_seed, model_seed, comm_size in itertools.product(
        BASE_N_TREES["strong"], data_seeds, model_seeds, comm_sizes
    ):
        # assumes 25% un-parallelizable workload (data generation, communication, saving results,...)
        expected_time = int(
            math.ceil(SERIAL_BASELINE_TIMES[n_trees] * (0.25 + 0.75 / comm_size))
        )
        label = (
            f"strong_scaling/t_{n_trees}/n_nodes_{comm_size}/{data_seed}_{model_seed}"
        )
        run_specific_configs = {
            "job_name": label,
            "n_trees": n_trees,
            "random_state_data": data_seed,
            "random_state_model": model_seed,
            "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
            "n_nodes": comm_size,
            "result_dir": result_base_dir / label,
        }
        config = {**global_config, **run_specific_configs}
        generate_job_script(
            base_job_script_path / f"{label}.sh", config, SCRIPT_TEMPLATE
        )


def generate_weak_scaling_job_scripts(
    global_config: dict[str, Any],
    data_seeds: list[int],
    model_seeds: list[int],
    comm_sizes: list[int],
    result_base_dir: pathlib.Path,
    base_job_script_path: pathlib.Path,
) -> None:
    """
    Generate all weak scaling job scripts.

    Parameters
    ----------
    global_config : dict[str, Any]
        The global job script configuration. Should contain the following keys: project, script_dir, venv, n_classes,
        script, mem, additional_args, partition. All other keys may be overwritten.
    data_seeds : list[int]
        The list of data seeds.
    model_seeds : list[int]
        The list of model seeds.
    comm_sizes : list[int]
        The list comm sizes (node counts).
    result_base_dir : pathlib.Path
        The base directory to write the job results to (subdirectories for experiment and run config will be created).
    base_job_script_path : pathlib.Path
        The base directory to write the job scripts to (subdirectories for experiment and run config will be created).
    """
    for n_trees_local, data_seed, model_seed, comm_size in itertools.product(
        BASE_N_TREES["weak"], data_seeds, model_seeds, comm_sizes
    ):
        expected_time = SERIAL_BASELINE_TIMES[n_trees_local]
        label = f"weak_scaling/t_{n_trees_local}/n_nodes_{comm_size}/{data_seed}_{model_seed}"
        run_specific_configs = {
            "job_name": label,
            "n_trees": n_trees_local
            * comm_size,  # weak scaling = global model size scales with n_nodes
            "random_state_data": data_seed,
            "random_state_model": model_seed,
            "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
            "n_nodes": comm_size,
            "result_dir": result_base_dir / label,
        }
        config = {**global_config, **run_specific_configs}
        generate_job_script(
            base_job_script_path / f"{label}.sh", config, SCRIPT_TEMPLATE
        )


if __name__ == "__main__":
    # setup paths
    base_dir = pathlib.Path(
        "/hkfs/home/project/hk-project-test-haiga/bk6983/special-couscous"
    )
    result_base_dir = pathlib.Path(
        "/hkfs/work/workspace/scratch/bk6983-special_couscous__2025_results"
    )
    script_dir = base_dir / "scripts/examples/"
    base_job_script_path = pathlib.Path(__file__).parent / "susy"
    venv = base_dir / "venv311"

    GLOBAL_CONFIG = {
        "project": "hk-project-p0022229",
        "partition": "cpuonly",
        "script_dir": script_dir,
        "venv": venv,
        "mem": "239400mb",
        "script": "rf_training_susy.py",
        "additional_args": "",
    }

    data_seeds = [0]
    model_seeds = [1]

    args = [
        GLOBAL_CONFIG,
        data_seeds,
        model_seeds,
        NODES,
        result_base_dir,
        base_job_script_path,
    ]

    generate_strong_scaling_job_scripts(*args)
    generate_weak_scaling_job_scripts(*args)
