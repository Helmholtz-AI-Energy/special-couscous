import os
import pathlib
import subprocess


def generate_serial_strong_baseline_job_scripts(
    log_n_samples: int,
    log_n_features: int,
    wall_time: int,
    n_classes: int,
    n_trees: int,
    random_state_data: int,
    random_state_model: int,
    output_path: pathlib.Path,
    submit: bool = False,
) -> None:
    """
    Generate the job scripts for the serial strong scaling baselines.

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples in the synthetic dataset.
    log_n_features : int
        The common logarithm of the number of features in the synthetic dataset.
    wall_time : int
        The wall-time in minutes.
    n_classes : int
        The number of classes in the synthetic dataset.
    n_trees : int
        The number of trees to use.
    random_state_data : int
        The random state.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    job_name = f"n{log_n_samples}_m{log_n_features}_nodes_1_{random_state_data}_{random_state_model}"
    job_script_name = f"{job_name}.sh"
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}  # Job name
#SBATCH --nodes=1              # Number of nodes
#SBATCH --mem=501600mb         # Use large-memory nodes.
#SBATCH --partition=cpuonly    # Queue for resource allocation
#SBATCH --time={wall_time}          # Wall-clock time limit
#SBATCH --cpus-per-task=76     # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL        # Notify user by email when certain event types occur
#SBATCH --account=hk-project-p0022229

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-SpecialCouscous}}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi/4.1
source "${{BASE_DIR}}"/special-couscous-venv-openmpi4/bin/activate  # Activate venv.

SCRIPT="special-couscous/scripts/examples/rf_serial_synthetic.py"

RESDIR="${{BASE_DIR}}"/results/strong_scaling/n{log_n_samples}_m{log_n_features}/nodes_1/${{SLURM_JOB_ID}}_{random_state_data}_{random_state_model}/
mkdir -p "${{RESDIR}}"
cd "${{RESDIR}}" || exit

python -u ${{BASE_DIR}}/${{SCRIPT}} \\
    --n_samples {10**log_n_samples} \\
    --n_features {10**log_n_features} \\
    --n_trees {n_trees} \\
    --n_classes {n_classes} \\
    --random_state {random_state_data} \\
    --random_state_model {random_state_model} \\
    --detailed_evaluation \\
    --save_model \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --log_path ${{RESDIR}}
                                """
    # Write script content to file.
    with open(output_path / job_script_name, "wt") as f:
        f.write(script_content)
    # Possibly submit script to cluster.
    if submit:
        subprocess.run(f"sbatch {output_path}/{job_script_name}", shell=True)


if __name__ == "__main__":
    wall_time_limit = 60 * 24 * 3
    # Considered datasets given as pairs of (log10 <number of samples>, log10 <number of features>, <number of trees>):
    datasets = [(6, 4, 1600), (7, 3, 448)]
    n_classes = 10
    # Considered seeds:
    random_state_data = 0
    seeds_model = [1, 2, 3]
    output_path = pathlib.Path("./strong_scaling/serial")
    os.makedirs(output_path, exist_ok=True)

    for (
        random_state_model
    ) in seeds_model:  # Test five model seeds for each dataset seed.
        for dataset in datasets:  # Loop over considered datasets.
            log_n_samples = dataset[0]  # Extract number of samples.
            log_n_features = dataset[1]  # Extract number of features.
            n_trees = dataset[2]
            generate_serial_strong_baseline_job_scripts(
                log_n_samples=log_n_samples,
                log_n_features=log_n_features,
                wall_time=wall_time_limit,
                n_classes=n_classes,
                n_trees=n_trees,
                random_state_data=random_state_data,
                random_state_model=random_state_model,
                output_path=output_path,
                submit=True,
            )
