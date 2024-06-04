import argparse
import time
from RF_dataloaders import *
from RF_parallel import DistributedRandomForest


# SETTINGS

# Communication
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

# Parse command-line arguments.
parser = argparse.ArgumentParser(
    prog="Distributed Random Forests",
    description="Set up and distributed random forest classification.",
    # epilog="Add help text here.",
    )
parser.add_argument(
    "-dl",
    "--dataloader",
    type=str,
    choices=[
        "root_with_replace",
        "parallel_with_replace",
        "proof_of_concept",
        "root_wo_replace",
        "parallel_wo_replace"
    ],
    default='root_wo_replace'
)
parser.add_argument("-rsd", "--random_state_data", type=int, default=0)
parser.add_argument("-t", "--n_trees", type=int, default=1000)
parser.add_argument("-rsf", "--random_state_forest", type=int, default=0)
parser.add_argument("-g", "--global_model", action="store_true")
parser.add_argument("-c", "--n_classes", type=int, default=2)
args = parser.parse_args()

# CONFIG
config = {
    "header_lines": 0,
    "sep": ",",
    "encoding": "utf-8",
    "path_to_data": "/pfs/work7/workspace/scratch/ku4408-RandomForest/data/SUSY.csv",
    "verbose": False,
    "train_split": 0.9,
    "job_id": int(os.getenv("SLURM_JOB_ID"))
}

# DATA LOADING
print(f"[{rank}/{size}]: Loading data...")
start_load = time.perf_counter()
if args.dataloader == "root_with_replace":  # Root-based dataloader
    if rank == 0: 
        print("Using root-based dataloader (sample train sets with replacement).")
    (
        train_samples,
        train_targets,
        test_samples,
        test_targets,
    ) = load_data_root(
        path_to_data=config["path_to_data"],
        header_lines=config["header_lines"],
        comm=comm,
        random_state=args.random_state_data,
        sample_with_replacement=True,
        train_split=config["train_split"],
        sep=config["sep"],
        verbose=config["verbose"]
    )
elif args.dataloader == "parallel_with_replace":  # Truly parallel dataloader
    if rank == 0:
        print("Using truly parallel dataloader (sample train sets with replacement).")
    (
        train_samples,
        train_targets,
        test_samples,
        test_targets,
    ) = load_data_parallel_bytes(
        path_to_data=config["path_to_data"],
        header_lines=config["header_lines"],
        comm=comm,
        random_state=args.random_state_data,
        train_split=config["train_split"],
        sample_with_replacement=True,
        sep=config["sep"],
        verbose=config["verbose"],
    )
elif args.dataloader == "proof_of_concept":  # All-parallel dataloader
    if rank == 0:
        print("Using proof-of-concept dataloader (everyone gets all the data).")
    (
        train_samples,
        train_targets,
        test_samples,
        test_targets,
    ) = load_data_parallel_poc(
        path_to_data=config["path_to_data"],
        header_lines=config["header_lines"],
        comm=comm,
        random_state=args.random_state_data,
        train_split=config["train_split"],
        sep=config["sep"]
    )
elif args.dataloader == "root_wo_replace":
    if rank == 0:
        print("Using root-based dataloader with real chunking (sample train sets without replacement).")
    (
        train_samples,
        train_targets,
        test_samples,
        test_targets,
    ) = load_data_root(
        path_to_data=config["path_to_data"],
        header_lines=config["header_lines"],
        comm=comm,
        random_state=args.random_state_data,
        sample_with_replacement=False,
        train_split=config["train_split"],
        sep=config["sep"],
        verbose=config["verbose"]
    )
elif args.dataloader == "parallel_wo_replace":
    if rank == 0:
        print("Using truly parallel dataloader with real chunking (sample train sets without replacement).")
    (
        train_samples,
        train_targets,
        test_samples,
        test_targets,
    ) = load_data_parallel_bytes(
        path_to_data=config["path_to_data"],
        header_lines=config["header_lines"],
        comm=comm,
        random_state=args.random_state_data,
        train_split=config["train_split"],
        sample_with_replacement=False,
        sep=config["sep"],
        verbose=config["verbose"]
    )
else:
    raise ValueError(f"Invalid dataloader {args.dataloader}")

elapsed_load_local = time.perf_counter() - start_load
print(f"[{rank}/{size}]: Done...\nLocal train samples and targets have shapes {train_samples.shape} "
      f"and {train_targets.shape}.\nGlobal test samples and targets have shapes {test_samples.shape} "
      f"and {test_targets.shape}.\n Labels are {test_targets}"
      )
elapsed_load_average = comm.allreduce(elapsed_load_local, op=MPI.SUM) / size

if rank == 0:
    print(f"Average time elapsed for data loading: {elapsed_load_average} s")

# RANDOM FOREST
dist_forest = DistributedRandomForest(
    n_trees_global=args.n_trees,  # global number of trees
    comm=comm,  # communicator
    random_state=args.random_state_forest,  # random state for rank-local sklearn sub-forest
    global_model=args.global_model,
)
# Train
start_train = time.perf_counter()
dist_forest.train(
    train_samples,
    train_targets,
    args.global_model,
)
elapsed_train_local = time.perf_counter() - start_train
elapsed_train_average = comm.allreduce(elapsed_train_local, op=MPI.SUM) / size
if rank == 0:
    print(f"Average time elapsed for training: {elapsed_train_average} s")

# Test
start_test = time.perf_counter()
dist_forest.test(
    test_samples,  # test samples
    test_targets,  # test targets
    args.n_classes,  # number of classes
    args.global_model,
)
elapsed_test_local = time.perf_counter() - start_test
elapsed_test_average = comm.allreduce(elapsed_test_local, op=MPI.SUM) / size
if rank == 0:
    print(f"Average time elapsed for testing: {elapsed_test_average} s")
