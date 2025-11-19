import itertools
import logging
import math
import pathlib
import re
from typing import Any, Tuple

from specialcouscous.utils import set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.


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
RESULT_DIR="{result_dir}_${{SLURM_JOB_ID}}"

# Activate venv
source {venv}/bin/activate

mkdir -p "${{RESULT_DIR}}"
cd "${{RESULT_DIR}}" || exit

srun python -u ${{SCRIPT}} \\
    {dataset_config} \\
    --n_trees {n_trees} \\
    --random_state {random_state_data} \\
    --random_state_model {random_state_model} \\
    --detailed_evaluation \\
    --save_model \\
    --output_dir ${{RESULT_DIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --log_path ${{RESULT_DIR}} {additional_args}
"""


class BatchScriptConfigInference:
    """
    Horeka specific batch script configuration.

    Determines system specific time limits, partition, and memory settings.
    """

    SERIAL_BASELINE_TIMES: dict[Tuple[str, int], int] = {  # in minutes
        ("n6_m4", 1600): 138,
        ("n7_m3", 448): 227,
        ("n6_m4", 800): 73,
        ("n7_m3", 224): 103,
        ("n5_m3", 76): 1,
        ("n6_m2", 76): 2,
        ("susy", 1000): 30,
        ("susy", 100): 10,
        ("cover_type", 1000): 10,
        ("cover_type", 500): 5,
        ("cover_type", 100): 2,
        ("higgs", 320): 15,
        ("higgs", 640): 30,
        ("higgs", 1280): 60,
        ("higgs", 2560): 120,
        ("higgs", 5120): 240,
        ("higgs", 10): 5,
    }

    MEM_CONFIGS: dict[str, dict[str, str]] = {
        "standard": {"partition": "cpuonly", "mem": "239400mb"},  # standard nodes
        "high-memory": {
            "partition": "cpuonly",
            "mem": "497500mb",
        },  # high-memory nodes (32 nodes available)
        "large": {
            "partition": "large",
            "mem": "4120112mb",
        },  # extra-large nodes (8 nodes available)
    }

    # time limit by partition in minutes: 3 days for cpuonly, 2 days for large
    MAX_TIME_LIMITS = {"cpuonly": 60 * 24 * 3, "large": 60 * 24 * 3}

    def __init__(
        self,
        overestimation_factor: int | float = 2,
        strong_scaling_overhead: float = 0.25,
        weak_scaling_overhead: float = 0.01,
    ):
        """
        Initialize BatchScriptConfigInference, setting the defaults for the time estimation.

        Parameters
        ----------
        overestimation_factor : int | float
            How much to overestimate the time limit, i.e. time limit = estimated time * overestimation factor.
        strong_scaling_overhead : float
            Expected strong scaling overhead, i.e. expected T(p) = T(1) * ((1-overhead) / p + overhead)
        weak_scaling_overhead : float
            Expected weak scaling overhead, i.e. expected T(p) = T(1) * (1 + overhead * p)
        """
        self.overestimation_factor = overestimation_factor
        self.strong_scaling_overhead = strong_scaling_overhead
        self.weak_scaling_overhead = weak_scaling_overhead

    def get_serial_runtime(self, dataset: str, n_trees: int) -> int:
        """
        Get the serial runtime for a given dataset and forest size.

        Raises a ValueError if no baseline runtimes are recorded for the requested dataset.
        Approximates the runtime from the closest match if no exact match of forest size is available.

        Parameters
        ----------
        dataset : str
            The dataset to train and evaluate on.
        n_trees : str
            The size of the (serial) forest in number of trees.

        Returns
        -------
        The recorded or approximated serial runtime for this combination.

        Raises
        ------
        ValueError
            If no baseline runtime is available for this dataset.
        """
        if (dataset, n_trees) not in self.SERIAL_BASELINE_TIMES:
            log.debug(
                f"No baseline runtime available for {dataset=}, {n_trees=}. Trying to find alternative."
            )
            available_n_trees = [
                key[1] for key in self.SERIAL_BASELINE_TIMES if dataset == key[0]
            ]
            if not available_n_trees:
                raise ValueError(f"No baseline runtime available for {dataset=}.")
            closest_n_trees = min(
                available_n_trees, key=lambda x: abs(n_trees / x - 1.0)
            )
            log.info(
                f"No baseline runtime available for {dataset=}, {n_trees=}. "
                f"Interpolating from n_trees={closest_n_trees}"
            )
            return int(
                self.SERIAL_BASELINE_TIMES[(dataset, closest_n_trees)]
                * n_trees
                / closest_n_trees
            )
        return self.SERIAL_BASELINE_TIMES[(dataset, n_trees)]

    def estimate_runtime(
        self,
        dataset: str,
        n_trees_serial: int,
        n_nodes: int,
        scaling: str | None,
        parallel_overhead: float | None = None,
    ) -> int:
        """
        Roughly estimate the runtime T(p) with p nodes by scaling the serial runtime T(1).

        The scaling depends on the scaling type, the number of nodes, and the overestimation factor.
        - For strong scaling: T(p) = T(1) * ((1-overhead) / p + overhead)
        - For weak scaling:   T(p) = T(1) * (1 + overhead * p)

        Parameters
        ----------
        dataset : str
            The dataset to train and evaluate on.
        n_trees_serial : str
            The size of the serial forest in number of trees.
        n_nodes : int
            The number of parallel nodes.
        scaling : str | None
            The type of parallel scaling: "strong" or "weak" for strong and weak scaling or None (with n_nodes = 1) for
            serial runs.
        parallel_overhead : float | None
            Used to overwrite the default parallel overhead. Note that strong and weak scaling typically have different overhead.

        Returns
        -------
        int
            A rough estimation of the runtime with p nodes.

        Raises
        ------
        ValueError
            If no baseline runtime is available for this dataset.
        """
        serial_runtime = self.get_serial_runtime(dataset, n_trees_serial)
        if scaling == "strong":
            parallel_overhead = (
                self.strong_scaling_overhead
                if parallel_overhead is None
                else parallel_overhead
            )
            parallel_multiplier = (
                parallel_overhead + max(1 - parallel_overhead, 0) / n_nodes
            )
        elif scaling == "weak":
            parallel_overhead = (
                self.weak_scaling_overhead
                if parallel_overhead is None
                else parallel_overhead
            )
            parallel_multiplier = 1 + parallel_overhead * n_nodes
        elif scaling is None:
            if n_nodes > 1:
                raise UserWarning(
                    f"{scaling=} interpreted as serial run but {n_nodes=} > 1."
                )
            parallel_multiplier = 1
        else:
            raise ValueError(f"Unexpected {scaling=}.")
        return int(math.ceil(serial_runtime * parallel_multiplier))

    def infer_time_limit(
        self,
        dataset: str,
        n_trees_serial: int,
        n_nodes: int,
        scaling: str | None,
        partition: str | None = None,
        parallel_overhead: float | None = None,
    ) -> int:
        """
        Determine the time limit for the given run by estimating the runtime.

        When given a partition, additionally ensures that the time limit is within the maximum of this partition.

        Parameters
        ----------
        dataset : str
            The dataset to train and evaluate on.
        n_trees_serial : str
            The size of the serial forest in number of trees.
        n_nodes : int
            The number of parallel nodes.
        scaling : str | None
            The type of parallel scaling: "strong" or "weak" for strong and weak scaling or None (with n_nodes = 1) for
            serial runs.
        partition : str | None
            If given, the time limit is limited to the maximum on the given partition.
        parallel_overhead : float | None
            Used to overwrite the default parallel overhead. Note that strong and weak scaling typically have different overhead.

        Returns
        -------
        int
            The time limit to use for the given run.

        Raises
        ------
        ValueError
            If no baseline runtime is available for this dataset.
        """
        estimated_runtime = self.estimate_runtime(
            dataset, n_trees_serial, n_nodes, scaling, parallel_overhead
        )
        time_limit = int(estimated_runtime * self.overestimation_factor)
        if partition is not None and partition in self.MAX_TIME_LIMITS:
            time_limit = min(time_limit, self.MAX_TIME_LIMITS[partition])
        return time_limit

    def infer_memory_limit_and_partition(
        self, dataset: str, n_trees_local: int
    ) -> dict[str, str]:
        """
        Infer the required memory and partition based on the dataset and tree size.

        Parameters
        ----------
        dataset : str
            The dataset to train and evaluate on.
        n_trees_local : int
            The forest size in number of trees.

        Returns
        -------
        dict[str, str]
            The memory and partition configuration.
        """
        node_type = "standard"
        if dataset == "n6_m4" and n_trees_local > 1000:
            node_type = "high-memory"
        if dataset == "n7_m3" and n_trees_local > 400:
            node_type = "high-memory"
        if dataset == "higgs" and n_trees_local > 1000:
            node_type = "high-memory"
        if dataset == "higgs" and n_trees_local > 3000:
            node_type = "large"
        return self.MEM_CONFIGS[node_type]


class BatchScriptGenerator:
    """Class to generate slurm batch scripts."""

    SYNTHETIC_DATASET_PATTERN = r"n(\d+)_m(\d+)"

    def __init__(
        self,
        script_template: str,
        default_values: dict[str, Any],
        job_script_root_dir: str | pathlib.Path,
        config_inference: BatchScriptConfigInference,
        data_seeds: list[int],
        model_seeds: list[int],
        nodes: list[int],
        result_base_dir: str | pathlib.Path,
    ):
        """
        Initialize a BatchScriptGenerator, setting the default parameters.

        Parameters
        ----------
        script_template : str
            The template string for the batch script.
        default_values : dict[str, Any]
            The default values shared by all/most batch scripts.
        job_script_root_dir : str | pathlib.Path
            The root directory to write the batch scripts to.
        config_inference : BatchScriptConfigInference
            A BatchScriptConfigInference object to infer system specific runtime, memory, and partition information.
        data_seeds : list[int]
            List of default data seeds.
        model_seeds : list[int]
            List of default model seeds.
        nodes : list[int]
            List of default parallel node counts.
        result_base_dir : str | pathlib.Path
            On-system path to the root result directory.
        """
        self.script_template = script_template
        self.default_values = default_values
        self.config_inference = config_inference
        self.job_script_root_dir = pathlib.Path(job_script_root_dir)

        self.data_seeds = data_seeds
        self.model_seeds = model_seeds
        self.nodes = nodes

        self.result_base_dir = pathlib.Path(result_base_dir)
        self.scripts = {"serial": "serial.py", "parallel": "parallel.py"}

    @classmethod
    def dataset_config_from_name(
        cls, dataset_name: str, n_classes_synth: int = 10
    ) -> str:
        """
        Get a dataset configuration from a dataset name.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        n_classes_synth : int
            The number of classes for synthetic datasets.

        Returns
        -------
        str
            The CLI parameters for the dataset configuration.
        """
        if match := re.match(cls.SYNTHETIC_DATASET_PATTERN, dataset_name):
            log_n_samples, log_n_features = [int(x) for x in match.groups()]
            return f"--n_samples {10**log_n_samples} --n_features {10**log_n_features} --n_classes {n_classes_synth}"
        return f"--dataset_name {dataset_name}"

    @staticmethod
    def get_local_global_n_trees(
        n_trees_serial: int, n_nodes: int, scaling_type: str | None
    ) -> Tuple[int, int]:
        """
        Get the local and global tree counts for different parallel scaling types.

        Parameters
        ----------
        n_trees_serial : int
            The serial tree count.
        n_nodes : int
            The number of nodes.
        scaling_type : str | None
            The scaling type: strong, weak, or None.

        Returns
        -------
        int
            The global tree count (total #trees over all nodes).
        int
            The local tree count (local #trees on each node nodes).
        """
        if scaling_type == "strong":
            n_trees_local = n_trees_serial // n_nodes
            n_trees_global = n_trees_local * n_nodes
        elif scaling_type == "weak":
            n_trees_global, n_trees_local = n_trees_serial * n_nodes, n_trees_serial
        elif scaling_type is None:
            n_trees_global, n_trees_local = n_trees_serial, n_trees_serial
        else:
            raise ValueError(f"Unexpected {scaling_type=}.")
        return n_trees_global, n_trees_local

    def generate_job_scripts(
        self,
        dataset_n_trees: list[Tuple[str, int]],
        scaling_type: str | None,
        label: str,
        script_type: str = "parallel",
        data_seeds: list[int] | None = None,
        model_seeds: list[int] | None = None,
        nodes: list[int] | None = None,
        shared_global_model: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Generate all job scripts for one experiment.

        Generates a batch script for each combination of seeds (model and data), nodes, and dataset + forest size.

        Parameters
        ----------
        dataset_n_trees : list[Tuple[str, int]]
            TODO
        scaling_type : str | None
            TODO
        label : str
            TODO
        script_type : str = "parallel"
            TODO
        data_seeds : list[int] | None = None
            Optional list of data seeds to overwrite the default in self.data_seeds
        model_seeds : list[int] | None = None
            Optional list of model seeds to overwrite the default in self.model_seeds
        nodes : list[int] | None = None
            Optional list of nodes to overwrite the default in self.nodes
        shared_global_model : bool | None = None
            Whether to use a shared model or not. When not None, "/(no_)shared_model" is added to the label.
            When None, no shared model is used and no such tag is added to the label.
        kwargs : Any
            Optional additional arguments to be passed to the script template.
        """
        data_seeds = self.data_seeds if data_seeds is None else data_seeds
        model_seeds = self.model_seeds if model_seeds is None else model_seeds
        nodes = self.nodes if nodes is None else nodes

        if shared_global_model:
            kwargs["additional_args"] = (
                kwargs.get("additional_args", "") + " --shared_global_model"
            )

        for data_seed, model_seed, (
            dataset,
            n_trees_serial,
        ), n_nodes in itertools.product(
            data_seeds, model_seeds, dataset_n_trees, nodes
        ):
            n_trees_global, n_trees_local = self.get_local_global_n_trees(
                n_trees_serial, n_nodes, scaling_type
            )
            mem_and_partition_config = (
                self.config_inference.infer_memory_limit_and_partition(
                    dataset, n_trees_local
                )
            )
            partition = mem_and_partition_config["partition"]
            parallel_overhead = 0.5 if shared_global_model else None
            time_limit = self.config_inference.infer_time_limit(
                dataset,
                n_trees_serial,
                n_nodes,
                scaling_type,
                partition,
                parallel_overhead,
            )
            dataset_config = self.dataset_config_from_name(dataset)

            nodes_label = "" if scaling_type is None else f"/n_nodes_{n_nodes}"
            inference_flavor_label = {
                True: "/shared_model",
                False: "/no_shared_model",
            }.get(shared_global_model, "")  # type: ignore
            experiment_label = (
                f"{label}/{dataset}_t{n_trees_serial}{inference_flavor_label}"
            )
            run_label = f"{experiment_label}{nodes_label}/{data_seed}_{model_seed}"

            run_specific_configs = {
                "job_name": run_label,
                "n_trees": n_trees_global,
                "random_state_data": data_seed,
                "random_state_model": model_seed,
                "time": time_limit,
                "n_nodes": n_nodes,
                "result_dir": self.result_base_dir / run_label,
                "script": self.scripts[script_type],
                "dataset_config": dataset_config,
                **mem_and_partition_config,
                **kwargs,
            }
            script_content = self.create_script_content(**run_specific_configs)
            self.write_script_to_file(f"{run_label}.sh", script_content)

    def generate_serial_job_scripts(
        self, dataset_n_trees: list[Tuple[str, int]], label: str = "serial_baseline"
    ) -> None:
        """
        Generate all serial scripts for a given list of dataset + #trees combinations.

        Parameters
        ----------
        dataset_n_trees : list[Tuple[str, int]]
            The list of dataset + #trees combinations to generate jobs scripts for.
        label : str
            Base label, default is "serial_baseline".
        """
        self.generate_job_scripts(dataset_n_trees, None, label, "serial", nodes=[1])

    def generate_scaling_job_scripts(
        self,
        dataset_n_trees: list[Tuple[str, int]],
        scaling_type: str,
        label: str | None = None,
    ) -> None:
        """
        Generate all strong or weak scripts for a given list of dataset + #trees combinations.

        Parameters
        ----------
        dataset_n_trees : list[Tuple[str, int]]
            The list of dataset + #trees combinations to generate jobs scripts for.
        scaling_type : str
            Switch between strong and weak scaling by passing either "strong" or "weak".
        label : str
            Base label, default is "<scaling_type>_scaling".
        """
        label = f"{scaling_type}_scaling" if label is None else label
        self.generate_job_scripts(dataset_n_trees, scaling_type, label)

    def generate_chunking_job_scripts(
        self, dataset_n_trees: list[Tuple[str, int]], label: str = "chunking"
    ) -> None:
        """
        Generate all chunking scripts for a given list of dataset + #trees combinations.

        Parameters
        ----------
        dataset_n_trees : list[Tuple[str, int]]
            The list of dataset + #trees combinations to generate jobs scripts for.
        label : str
            Base label, default is "chunking".
        """
        self.generate_job_scripts(
            dataset_n_trees,
            "strong",
            label,
            additional_args="--distribute_data --shared_test_set",
        )

    def generate_inference_job_scripts(
        self, dataset_n_trees: list[Tuple[str, int]], label: str = "inference_flavor"
    ) -> None:
        """
        Generate all inference scripts for a given list of dataset + #trees combinations.

        Generates scripts for both inference flavors: with and without a shared model.

        Parameters
        ----------
        dataset_n_trees : list[Tuple[str, int]]
            The list of dataset + #trees combinations to generate jobs scripts for.
        label : str
            Base label, default is "inference_flavor".
        """
        self.generate_job_scripts(
            dataset_n_trees, "weak", label, shared_global_model=True
        )
        self.generate_job_scripts(
            dataset_n_trees, "weak", label, shared_global_model=False
        )

    def create_script_content(self, **values: Any) -> str:
        """
        Prepare a batch script by filling out the template with the given values.

        Parameters
        ----------
        values : Any
            kwargs to format the script template.

        Returns
        -------
        The script template filled with the given values.
        """
        values = {**self.default_values, **values}
        log.debug(f"Creating batch script using the following values: {values}.")
        return self.script_template.format(**values)

    def write_script_to_file(self, file_path: str, script_content: str) -> None:
        """
        Write a given string to a batch script at job_script_root_dir / file_path.

        Parameters
        ----------
        file_path : str
            (Sub-directory and) file name of the batch script to create.
        script_content : str
            Content to write to the batch script, e.g. generated with create_script_content.

        """
        path = self.job_script_root_dir / file_path
        path.parent.mkdir(exist_ok=True, parents=True)
        log.info(f"Writing batch script to {path}")
        with open(path, "w") as file:
            file.write(script_content)


if __name__ == "__main__":
    set_logger_config()  # setup default logging options

    # setup paths
    base_dir = pathlib.Path(
        "/hkfs/home/project/hk-project-test-haiga/bk6983/special-couscous"
    )
    script_dir = base_dir / "scripts/"
    base_job_script_path = pathlib.Path(__file__).parent / "batch_scripts"
    venv = base_dir / "venv311"
    result_base_dir = (
        "/hkfs/work/workspace/scratch/bk6983-special_couscous__2025_results"
    )

    GLOBAL_CONFIG = {
        "project": "hk-project-p0022229",
        "script_dir": script_dir,
        "venv": venv,
        "additional_args": "",
    }

    data_seeds = [0]
    model_seeds = [1]
    nodes = [2, 4, 8, 16, 32, 64]

    synthetic_datasets = {
        "strong": [("n6_m4", 1600), ("n7_m3", 448), ("higgs", 640)],
        "weak": [("n6_m4", 800), ("n7_m3", 224), ("higgs", 10)],
        "inference": [("n5_m3", 76), ("n6_m2", 76)],
    }
    serial_baselines = [
        *synthetic_datasets["strong"],
        *synthetic_datasets["weak"],
        *synthetic_datasets["inference"],
    ]
    serial_inference_baselines = [
        (dataset, n_trees * p)
        for p in [1] + nodes
        for (dataset, n_trees) in synthetic_datasets["inference"]
    ]

    config_inference = BatchScriptConfigInference()
    script_generator = BatchScriptGenerator(
        SCRIPT_TEMPLATE,
        GLOBAL_CONFIG,
        base_job_script_path,
        config_inference,
        data_seeds,
        model_seeds,
        nodes,
        result_base_dir,
    )

    log.info("-" * 20 + " Generating Serial Baseline Scripts  " + "-" * 20)
    script_generator.generate_serial_job_scripts(serial_baselines)
    script_generator.generate_serial_job_scripts(
        serial_inference_baselines, "serial_inference_baselines"
    )
    log.info("-" * 20 + " Generating Strong Scaling Scripts   " + "-" * 20)
    script_generator.generate_scaling_job_scripts(
        synthetic_datasets["strong"], "strong"
    )
    log.info("-" * 20 + " Generating Weak Scaling Scripts     " + "-" * 20)
    script_generator.generate_scaling_job_scripts(synthetic_datasets["weak"], "weak")
    log.info("-" * 20 + " Generating Chunking Scripts         " + "-" * 20)
    script_generator.generate_chunking_job_scripts(synthetic_datasets["strong"][:-1])
    log.info("-" * 20 + " Generating Inference Flavor Scripts " + "-" * 20)
    script_generator.generate_inference_job_scripts(synthetic_datasets["inference"])
