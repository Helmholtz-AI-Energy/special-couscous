import itertools
import math
import pathlib
from typing import Any

MAX_TIME = 60 * 24 * 3
MAX_TIME_HIGH_MEM = 2 * 24 * 60
NODES = [2, 4, 8, 16, 32, 64]

DATASETS = {
    "strong": [(6, 4, 1600), (7, 3, 448)],
    "weak": [(6, 4, 800), (7, 3, 224)],
    "inference": [(5, 3, 76), (6, 2, 76)],
}

SERIAL_BASELINE_TIMES = {  # in minutes
    (6, 4, 1600): 138,
    (7, 3, 448): 227,
    (6, 4, 800): 73,
    (7, 3, 224): 103,
    (5, 3, 76): 1,
    (6, 2, 76): 2,
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
    --n_samples {n_samples} \\
    --n_features {n_features} \\
    --n_trees {n_trees} \\
    --n_classes {n_classes} \\
    --random_state {random_state_data} \\
    --random_state_model {random_state_model} \\
    --detailed_evaluation \\
    --save_model \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --log_path ${{RESDIR}} {additional_args}
"""


def generate_job_script(path: pathlib.Path, config: dict[str, Any]) -> None:
    """
    Generate a job script by filling out the above template with the given configuration and writing it to the given path.

    Parameters
    ----------
    path : pathlib.Path
        The path to write the jobscript to.
    config : dict[str, Any]
        The job script configuration as dictionary. Needs to include the following keys: project, script_dir, venv,
        n_classes, script, job_name, additional_args, n_samples, n_features, random_state_data, random_state_model,
        time, mem, n_nodes, result_dir, n_trees, partition.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as file:
        file.write(SCRIPT_TEMPLATE.format(**config))
    print(f"Job script written to {path}")


def generate_serial_job_scripts(
    global_config: dict[str, Any],
    data_seeds: list[int],
    model_seeds: list[int],
    result_base_dir: pathlib.Path,
    base_job_script_path: pathlib.Path,
) -> None:
    """
    Generate all serial job scripts.

    Parameters
    ----------
    global_config : dict[str, Any]
        The global job script configuration. Should contain the following keys: project, script_dir, venv, n_classes,
        mem, additional_args, partition. Keys mem, partition, and all other keys may be overwritten.
    data_seeds : list[int]
        The list of data seeds.
    model_seeds : list[int]
        The list of model seeds.
    result_base_dir : pathlib.Path
        The base directory to write the job results to (subdirectories for experiment and run config will be created).
    base_job_script_path : pathlib.Path
        The base directory to write the job scripts to (subdirectories for experiment and run config will be created).
    """
    datasets = [*DATASETS["strong"], *DATASETS["weak"], *DATASETS["inference"]]
    comm_size = 1
    for dataset, data_seed, model_seed in itertools.product(
        datasets, data_seeds, model_seeds
    ):
        expected_time = SERIAL_BASELINE_TIMES[dataset]
        log_n_samples, log_n_features, n_trees = dataset
        label = f"serial_baseline/n{log_n_samples}_m{log_n_features}_t{n_trees}/{data_seed}_{model_seed}"
        run_specific_configs = {
            "job_name": label,
            "n_samples": 10**log_n_samples,
            "n_features": 10**log_n_features,
            "n_trees": n_trees,
            "random_state_data": data_seed,
            "random_state_model": model_seed,
            "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
            "n_nodes": comm_size,
            "result_dir": result_base_dir / label,
            "script": "rf_serial_synthetic.py",
        }
        # for strong scaling baselines -> use high memory node
        if n_trees in [1600, 448]:
            high_mem_config = {
                "mem": "501600mb",
                "time": min(run_specific_configs["time"], MAX_TIME_HIGH_MEM),
                "partition": "large",
            }
            run_specific_configs = {**run_specific_configs, **high_mem_config}
        config = {**global_config, **run_specific_configs}
        generate_job_script(base_job_script_path / f"{label}.sh", config)


def generate_serial_inference_job_scripts(
    global_config: dict[str, Any],
    data_seeds: list[int],
    model_seeds: list[int],
    result_base_dir: pathlib.Path,
    base_job_script_path: pathlib.Path,
) -> None:
    """
    Generate serial job scripts for all scales of the inference models (t=76, 76*2, ..., 76*64).

    Parameters
    ----------
    global_config : dict[str, Any]
        The global job script configuration. Should contain the following keys: project, script_dir, venv, n_classes,
        mem, additional_args, partition. Keys mem, partition, and all other keys may be overwritten.
    data_seeds : list[int]
        The list of data seeds.
    model_seeds : list[int]
        The list of model seeds.
    result_base_dir : pathlib.Path
        The base directory to write the job results to (subdirectories for experiment and run config will be created).
    base_job_script_path : pathlib.Path
        The base directory to write the job scripts to (subdirectories for experiment and run config will be created).
    """
    datasets = [*DATASETS["inference"]]
    scales = [1, 2, 4, 8, 16, 32, 64]
    for dataset, scale, data_seed, model_seed in itertools.product(
        datasets, scales, data_seeds, model_seeds
    ):
        expected_time = SERIAL_BASELINE_TIMES[dataset] * scale
        log_n_samples, log_n_features, n_trees = dataset
        n_trees *= scale
        label = f"serial_inference_baselines/n{log_n_samples}_m{log_n_features}_t{n_trees}/{data_seed}_{model_seed}"
        run_specific_configs = {
            "job_name": label,
            "n_samples": 10**log_n_samples,
            "n_features": 10**log_n_features,
            "n_trees": n_trees,
            "random_state_data": data_seed,
            "random_state_model": model_seed,
            "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
            "n_nodes": 1,
            "result_dir": result_base_dir / label,
            "script": "rf_serial_synthetic.py",
        }
        config = {**global_config, **run_specific_configs}
        generate_job_script(base_job_script_path / f"{label}.sh", config)


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
    datasets = DATASETS["strong"]
    for dataset, data_seed, model_seed, comm_size in itertools.product(
        datasets, data_seeds, model_seeds, comm_sizes
    ):
        # assumes 25% un-parallelizable workload (data generation, communication, saving results,...)
        expected_time = int(
            math.ceil(SERIAL_BASELINE_TIMES[dataset] * (0.25 + 0.75 / comm_size))
        )
        log_n_samples, log_n_features, n_trees = dataset
        label = f"strong_scaling/n{log_n_samples}_m{log_n_features}/n_nodes_{comm_size}/{data_seed}_{model_seed}"
        run_specific_configs = {
            "job_name": label,
            "n_samples": 10**log_n_samples,
            "n_features": 10**log_n_features,
            "n_trees": n_trees,
            "random_state_data": data_seed,
            "random_state_model": model_seed,
            "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
            "n_nodes": comm_size,
            "result_dir": result_base_dir / label,
        }
        config = {**global_config, **run_specific_configs}
        generate_job_script(base_job_script_path / f"{label}.sh", config)


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
    datasets = DATASETS["weak"]
    for dataset, data_seed, model_seed, comm_size in itertools.product(
        datasets, data_seeds, model_seeds, comm_sizes
    ):
        # assumes 1% overhead per node
        expected_time = int(
            math.ceil(SERIAL_BASELINE_TIMES[dataset] * (1 + 0.01 * comm_size))
        )
        log_n_samples, log_n_features, n_trees_local = dataset
        label = f"weak_scaling/n{log_n_samples}_m{log_n_features}/n_nodes_{comm_size}/{data_seed}_{model_seed}"
        run_specific_configs = {
            "job_name": label,
            "n_samples": 10**log_n_samples,
            "n_features": 10**log_n_features,
            "n_trees": n_trees_local
            * comm_size,  # weak scaling = global model size scales with n_nodes
            "random_state_data": data_seed,
            "random_state_model": model_seed,
            "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
            "n_nodes": comm_size,
            "result_dir": result_base_dir / label,
        }
        config = {**global_config, **run_specific_configs}
        generate_job_script(base_job_script_path / f"{label}.sh", config)


def generate_chunking_job_scripts(
    global_config: dict[str, Any],
    data_seeds: list[int],
    model_seeds: list[int],
    comm_sizes: list[int],
    result_base_dir: pathlib.Path,
    base_job_script_path: pathlib.Path,
) -> None:
    """
    Generate all chunking job scripts.

    Parameters
    ----------
    global_config : dict[str, Any]
        The global job script configuration. Should contain the following keys: project, script_dir, venv, n_classes,
        mem, partition. All other keys may be overwritten.
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
    datasets = DATASETS["strong"]
    for dataset, data_seed, model_seed, comm_size in itertools.product(
        datasets, data_seeds, model_seeds, comm_sizes
    ):
        # same as strong scaling (ignores additional speedup from chunking)
        expected_time = int(
            math.ceil(SERIAL_BASELINE_TIMES[dataset] * (0.25 + 0.75 / comm_size))
        )
        log_n_samples, log_n_features, n_trees = dataset
        label = f"chunking/n{log_n_samples}_m{log_n_features}/n_nodes_{comm_size}/{data_seed}_{model_seed}"
        run_specific_configs = {
            "job_name": label,
            "n_samples": 10**log_n_samples,
            "n_features": 10**log_n_features,
            "n_trees": n_trees,
            "random_state_data": data_seed,
            "random_state_model": model_seed,
            "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
            "n_nodes": comm_size,
            "result_dir": result_base_dir / label,
            "script": "rf_training_breaking_iid.py",
            "additional_args": "--shared_test_set",
        }
        config = {**global_config, **run_specific_configs}
        generate_job_script(base_job_script_path / f"{label}.sh", config)


def generate_inference_flavor_job_scripts(
    global_config: dict[str, Any],
    data_seeds: list[int],
    model_seeds: list[int],
    comm_sizes: list[int],
    result_base_dir: pathlib.Path,
    base_job_script_path: pathlib.Path,
) -> None:
    """
    Generate all inference flavor job scripts.

    Parameters
    ----------
    global_config : dict[str, Any]
        The global job script configuration. Should contain the following keys: project, script_dir, venv, n_classes,
        script, mem, partition. All other keys may be overwritten.
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
    datasets = DATASETS["inference"]
    for dataset, data_seed, model_seed, comm_size in itertools.product(
        datasets, data_seeds, model_seeds, comm_sizes
    ):
        log_n_samples, log_n_features, n_trees_local = dataset
        for shared_global_model in [True, False]:
            if shared_global_model:
                # assumes 50% overhead per node (for model gathering & evaluation)
                expected_time = SERIAL_BASELINE_TIMES[dataset] * (1 + 0.5 * comm_size)
                shared_model = "shared_model"
            else:
                # same as weak scaling, assumes 1% overhead per node
                expected_time = SERIAL_BASELINE_TIMES[dataset] * (1 + 0.01 * comm_size)
                shared_model = "no_shared_model"
            expected_time = int(math.ceil(expected_time))
            label = (
                f"inference_flavor/n{log_n_samples}_m{log_n_features}/{shared_model}/"
                f"n_nodes_{comm_size}/{data_seed}_{model_seed}"
            )
            run_specific_configs = {
                "job_name": label,
                "n_samples": 10**log_n_samples,
                "n_features": 10**log_n_features,
                "n_trees": n_trees_local
                * comm_size,  # weak scaling = global model size scales with n_nodes
                "random_state_data": data_seed,
                "random_state_model": model_seed,
                "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
                "n_nodes": comm_size,
                "result_dir": result_base_dir / label,
                "additional_args": (
                    "--shared_global_model" if shared_global_model else ""
                ),
            }
            config = {**global_config, **run_specific_configs}
            generate_job_script(base_job_script_path / f"{label}.sh", config)


def generate_small_scale_model_job_scripts(
    global_config: dict[str, Any],
    data_seeds: list[int],
    model_seeds: list[int],
    result_base_dir: pathlib.Path,
    base_job_script_path: pathlib.Path,
) -> None:
    """
    Generate all serial job scripts.

    Parameters
    ----------
    global_config : dict[str, Any]
        The global job script configuration. Should contain the following keys: project, script_dir, venv, n_classes,
        mem, additional_args, partition. Keys mem, partition, and all other keys may be overwritten.
    data_seeds : list[int]
        The list of data seeds.
    model_seeds : list[int]
        The list of model seeds.
    result_base_dir : pathlib.Path
        The base directory to write the job results to (subdirectories for experiment and run config will be created).
    base_job_script_path : pathlib.Path
        The base directory to write the job scripts to (subdirectories for experiment and run config will be created).
    """
    datasets = [*DATASETS["strong"], *DATASETS["weak"], *DATASETS["inference"]]
    comm_size = 1
    for dataset, data_seed, model_seed in itertools.product(
        datasets, data_seeds, model_seeds
    ):
        expected_time = SERIAL_BASELINE_TIMES[dataset]
        log_n_samples, log_n_features, n_trees = dataset
        label = f"serial_baseline/n{log_n_samples}_m{log_n_features}_t{n_trees}/{data_seed}_{model_seed}"
        run_specific_configs = {
            "job_name": label,
            "n_samples": 10**log_n_samples,
            "n_features": 10**log_n_features,
            "n_trees": n_trees,
            "random_state_data": data_seed,
            "random_state_model": model_seed,
            "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
            "n_nodes": comm_size,
            "result_dir": result_base_dir / label,
            "script": "rf_serial_synthetic.py",
        }
        # for strong scaling baselines -> use high memory node
        if n_trees in [1600, 448]:
            high_mem_config = {
                "mem": "501600mb",
                "time": min(run_specific_configs["time"], MAX_TIME_HIGH_MEM),
                "partition": "large",
            }
            run_specific_configs = {**run_specific_configs, **high_mem_config}
        config = {**global_config, **run_specific_configs}
        generate_job_script(base_job_script_path / f"{label}.sh", config)


def generate_scaling_model_and_data_job_scripts(
    global_config: dict[str, Any],
    data_seeds: list[int],
    model_seeds: list[int],
    comm_sizes: list[int],
    result_base_dir: pathlib.Path,
    base_job_script_path: pathlib.Path,
    scale_data: bool = True,
) -> None:
    """
    Generate job scripts for downscaling of model and data (i.e. scaling model and data without chunking).

    Train t0 * p trees on either n0 * p samples (scale_data == True) or n0 samples (scale_data == False) (no chunking in
    either case) with t0 = t / 64 and n0 = n / 64 for an n, t baseline.
    Note that for p=64, this is identical to strong scaling.

    Parameters
    ----------
    global_config : dict[str, Any]
        The global job script configuration. Should contain the following keys: project, script_dir, venv, n_classes,
        mem, partition. All other keys may be overwritten.
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
    scale_data : bool
        Whether to scale both model and data (True) or just the model (False).
    """
    datasets = DATASETS["strong"]
    for dataset, data_seed, model_seed, comm_size in itertools.product(
        datasets, data_seeds, model_seeds, comm_sizes
    ):
        expected_time = 15
        log_n_samples, log_n_features, n_trees = dataset
        label = (
            "scaling_model_and_data__no_chunking" if scale_data else "scaling_the_model"
        )
        label += f"/n{log_n_samples}_m{log_n_features}/n_nodes_{comm_size}/{data_seed}_{model_seed}"
        run_specific_configs = {
            "job_name": label,
            "n_samples": (10**log_n_samples) // 64 * (comm_size if scale_data else 1),
            "n_features": 10**log_n_features,
            "n_trees": n_trees // 64 * comm_size,
            "random_state_data": data_seed,
            "random_state_model": model_seed,
            "time": min(expected_time * OVERESTIMATION_FACTOR, MAX_TIME),
            "n_nodes": comm_size,
            "result_dir": result_base_dir / label,
        }
        config = {**global_config, **run_specific_configs}
        generate_job_script(base_job_script_path / f"{label}.sh", config)


if __name__ == "__main__":
    # setup paths
    base_dir = pathlib.Path(
        "/hkfs/home/project/hk-project-test-haiga/bk6983/special-couscous"
    )
    result_base_dir = pathlib.Path(
        "/hkfs/work/workspace/scratch/bk6983-special_couscous__2025_results"
    )
    script_dir = base_dir / "scripts/examples/"
    base_job_script_path = pathlib.Path(__file__).parent / "redo_runtime_measurements"
    venv = base_dir / "venv311"

    GLOBAL_CONFIG = {
        "project": "hk-project-p0022229",
        "partition": "cpuonly",
        "script_dir": script_dir,
        "venv": venv,
        "n_classes": 10,
        "mem": "243200mb",
        "script": "rf_parallel_synthetic.py",
        "additional_args": "",
    }

    data_seeds = [0]
    model_seeds = [1]

    generate_serial_job_scripts(
        GLOBAL_CONFIG, data_seeds, model_seeds, result_base_dir, base_job_script_path
    )
    generate_serial_inference_job_scripts(
        GLOBAL_CONFIG, data_seeds, model_seeds, result_base_dir, base_job_script_path
    )
    generate_strong_scaling_job_scripts(
        GLOBAL_CONFIG,
        data_seeds,
        model_seeds,
        NODES,
        result_base_dir,
        base_job_script_path,
    )
    generate_weak_scaling_job_scripts(
        GLOBAL_CONFIG,
        data_seeds,
        model_seeds,
        NODES,
        result_base_dir,
        base_job_script_path,
    )
    generate_chunking_job_scripts(
        GLOBAL_CONFIG,
        data_seeds,
        model_seeds,
        NODES,
        result_base_dir,
        base_job_script_path,
    )
    generate_inference_flavor_job_scripts(
        GLOBAL_CONFIG,
        data_seeds,
        model_seeds,
        NODES,
        result_base_dir,
        base_job_script_path,
    )
    generate_scaling_model_and_data_job_scripts(
        GLOBAL_CONFIG,
        data_seeds,
        model_seeds,
        [1, 2, 4, 8, 16, 32],
        result_base_dir,
        base_job_script_path.parent,
        scale_data=True,
    )
    generate_scaling_model_and_data_job_scripts(
        GLOBAL_CONFIG,
        data_seeds,
        model_seeds,
        [64],
        result_base_dir,
        base_job_script_path.parent,
        scale_data=False,
    )
