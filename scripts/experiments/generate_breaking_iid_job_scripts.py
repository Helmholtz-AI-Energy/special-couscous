import pathlib
import subprocess


def generate_breaking_iid_job_scripts(
    log_n_samples: int,
    log_n_features: int,
    n_classes: int,
    mu_global: float | str,
    mu_local: float | str,
    n_trees: int,
    data_seed: int,
    model_seed: int,
    enforce_constant_size: bool = False,
    submit: bool = False,
) -> None:
    """
    Generate the job scripts for the breaking-IID experiments.

    To study the effect of breaking IID for distributed RFs, we perform a series of 16-node experiments combining
    different types and degrees of class imbalances. The experiments are based on the corresponding weak scaling setup,
    i.e., n6m4 baseline (n, m, t) = (10^6, 10^4, 800) and n7m3 baseline: (n, m, t) = (10^7, 10^3, 224) with number of
    samples n, number of features m, and number of trees t.
    We compare three different imbalance factors µ = {0.5, 2, ∞} for both global and local imbalance, resulting in nine
    combinations. Each experiment is run with different seed combinations. The overall number of experiments is
    determined by the number of data seeds x the number of model seeds provided in the main script.
    Note that both model and data are distributed (similar to the chunking experiments).

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples to use.
    log_n_features : int
        The common logarithm of the number of features to use.
    n_classes : int
        The number of classes to use.
    mu_global : float | str
        The global imbalance factor.
    mu_local : float | str
        The local imbalance factor.
    n_trees : int
        The number of trees to use. For weak scaling, this number serves as a baseline and is scaled up with the number
        of nodes.
    data_seed : int
        The random state used for synthetic dataset generation, splitting, and distribution.
    model_seed : int
        The (base) random state used for initializing the (distributed) model.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    enforce_constant_size : bool
        If true, the local class distribution is relaxed to instead force all local subsets to have the same size.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    # All weak-scaling style experiments should take approx. the same time (in min).
    # Note that the time is reduced compared to normal weak scaling as both model and data are distributed.
    mem = 239400  # Use standard nodes.
    n_nodes = 16
    # time = 4 * 3600 // n_nodes
    time = 10 if log_n_samples <= 5 else 60
    project = "hk-project-p0022229"

    print(
        f"Current config uses {n_nodes} nodes and {n_nodes * n_trees} trees. Wall-clock time is {time / 60}h."
    )

    dataset_name = f"n{log_n_samples}_m{log_n_features}"
    enforce_constant_local_size = ""

    if enforce_constant_size:
        dataset_name += "_const_size"
        enforce_constant_local_size = "--enforce_constant_local_size"

    def format_mu(mu: float | str) -> str:
        return str(mu).replace(".", "")

    seeds_and_mus = (
        f"{data_seed}_{model_seed}_{format_mu(mu_global)}_{format_mu(mu_local)}"
    )
    label = f"breaking_iid/{dataset_name}/n_nodes_{n_nodes}/{seeds_and_mus}"
    result_dir = RESULT_BASE_DIR / label

    script_content = f"""#!/bin/bash
#SBATCH --job-name={label}         # Job name
#SBATCH --partition=cpuonly           # Queue for resource allocation
#SBATCH --time={time}                 # Wall-clock time limit
#SBATCH --mem={mem}mb                 # Main memory
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
SCRIPT="{SCRIPT_DIR}/{SCRIPT}"
RESDIR="{result_dir}_${{SLURM_JOB_ID}}"

# Activate venv
source {VENV}/bin/activate

mkdir -p "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun python -u ${{SCRIPT}} \\
    --n_samples {10**log_n_samples} \\
    --n_features {10**log_n_features} \\
    --n_classes {n_classes} \\
    --shared_test_set \\
    --globally_imbalanced \\
    --mu_data {mu_global} \\
    --locally_imbalanced \\
    --mu_partition {mu_local} \\
    --random_state {data_seed} \\
    --n_trees {n_nodes * n_trees} \\
    --random_state_model {model_seed} \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --log_path ${{RESDIR}} \\
    --detailed_evaluation \\
    --save_model \\
    {enforce_constant_local_size}
"""
    output_path = BASE_JOB_SCRIPT_PATH / f"{label}.sh"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wt") as f:
        f.write(script_content)
    if submit:
        subprocess.run(f"sbatch {output_path}", shell=True)


if __name__ == "__main__":
    data_sets = [
        (6, 4, 800),
        (7, 3, 224),
        (5, 3, 224),
    ]  # Baseline problem as (`log_n_samples`, `log_n_features`, `n_trees`)
    data_seeds = [0]  # , 1, 2]  # Data seed to use
    model_seeds = [0, 1, 2]  # Model seeds to use
    n_classes = 10  # Number of classes to use
    mu_global = [0.5, 1.0, 2.0, 5.0, 10.0, "inf"]  # Global imbalance factors considered
    mu_local = [0.5, 1.0, 2.0, 5.0, 10.0, "inf"]  # Local imbalance factors considered

    # setup paths
    BASE_DIR = pathlib.Path(
        "/hkfs/home/project/hk-project-test-haiga/bk6983/special-couscous"
    )
    RESULT_BASE_DIR = pathlib.Path(
        "/hkfs/work/workspace/scratch/bk6983-special_couscous__2025_results"
    )
    BASE_JOB_SCRIPT_PATH = pathlib.Path(__file__).parent
    BASE_JOB_SCRIPT_PATH.mkdir(exist_ok=True, parents=True)
    SCRIPT_DIR = BASE_DIR / "scripts/examples/"
    SCRIPT = "rf_training_breaking_iid.py"
    VENV = BASE_DIR / "venv311"

    # Loop over all considered configurations.
    for random_state_data in data_seeds:
        for random_state_model in model_seeds:
            for data_set in data_sets:
                for m_global in mu_global:
                    for m_local in mu_local:
                        for enforce_constant_size in [True, False]:
                            log_n_samples = data_set[0]
                            log_n_features = data_set[1]
                            n_trees = data_set[2]
                            subdir = (
                                "const_local_size"
                                if enforce_constant_size
                                else "original"
                            )
                            # Generate job scripts and possibly submit them to the cluster.
                            assert (
                                isinstance(m_global, str) or isinstance(m_global, float)
                            ) and (
                                isinstance(m_local, str) or isinstance(m_local, float)
                            )
                            generate_breaking_iid_job_scripts(
                                log_n_samples=log_n_samples,
                                log_n_features=log_n_features,
                                n_classes=n_classes,
                                mu_global=m_global,
                                mu_local=m_local,
                                n_trees=n_trees,
                                data_seed=random_state_data,
                                model_seed=random_state_model,
                                enforce_constant_size=enforce_constant_size,
                                submit=False,
                            )
