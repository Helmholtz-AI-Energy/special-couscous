import pathlib
import subprocess


def generate_serial_acc_drop_job_scripts(
    log_n_samples: int,
    log_n_features: int,
    wall_time: int,
    n_classes: int,
    n_trees: int,
    random_state: int,
    output_path: pathlib.Path,
    submit: bool = False,
) -> None:
    """
    Generate the job scripts for the parallel dropping accuracy experiments.

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
    random_state : int
        The random state.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    job_name = f"n{log_n_samples}_m{log_n_features}_ntasks_1_seed_{random_state}"
    job_script_name = f"{job_name}.sh"
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}  # Job name
#SBATCH --partition=cpuonly    # Queue for resource allocation
#SBATCH --mem=501600mb         # Memory required per node
#SBATCH --time={wall_time}          # Wall-clock time limit
#SBATCH --cpus-per-task=76     # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL        # Notify user by email when certain event types occur

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-special-couscous/}}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export PYDIR=${{BASE_DIR}}/special-couscous/specialcouscous

ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi
source "${{BASE_DIR}}"/special-couscous-venv/bin/activate  # Activate venv.

# Set hyperparameters of synthetic dataset and random forest model.
N_SAMPLES={10**log_n_samples}
N_FEATURES={10**log_n_features}
N_TREES={n_trees}

SCRIPT="scripts/examples/rf_serial_synthetic.py"

RESDIR="${{BASE_DIR}}"/results/acc_drop/n${{N_SAMPLES}}_m${{N_FEATURES}}/ntasks_1_${{SLURM_JOB_ID}}/
mkdir "${{RESDIR}}"
cd "${{RESDIR}}" || exit

python -u ${{PYDIR}}/${{SCRIPT}} \\
    --n_samples ${{N_SAMPLES}} \\
    --n_features ${{N_FEATURES}} \\
    --n_trees ${{N_TREES}} \\
    --n_classes {n_classes} \\
    --random_state {random_state} \\
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
        subprocess.run(f"sbatch {job_script_name}", shell=True)


if __name__ == "__main__":
    n_trees = 76  # Number of trees in global forest
    # Considered datasets given as pairs of (log10 <number of samples>, log10 <number of features>, wall-time):
    datasets = [(5, 3, 10), (6, 2, 30)]
    n_classes = 10
    # Considered seeds given as pairs of (<data-related seed>, <model-related seed>):
    seeds = [0, 1, 2, 3, 4, 5]
    output_path = pathlib.Path("dropping_acc/")

    for random_state in seeds:  # Loop over five different seed combinations
        for dataset in datasets:  # Loop over considered datasets.
            log_n_samples = dataset[0]  # Extract number of samples.
            log_n_features = dataset[1]  # Extract number of features.
            wall_time = dataset[2]
            generate_serial_acc_drop_job_scripts(
                log_n_samples=log_n_samples,
                log_n_features=log_n_features,
                wall_time=wall_time,
                n_classes=n_classes,
                n_trees=n_trees,
                random_state=random_state,
                output_path=output_path,
                submit=False,
            )
