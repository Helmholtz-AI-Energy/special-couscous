# Sample data samples from CSV file with replacement
# using multiple processors in truly parallel fashion.
# Each processor needs to know
#   - overall number samples
#   - absolute byte positions of data samples' start / end as
#     array byte_pos = [[start0, end0], [start1, end1],...]
# Once I got this array:
# - Perform train-test split by randomly sampling `test_frac`
#   indices from overall number of samples xn = len(byte_pos).
#   Those samples are held out to form test dataset.
#   The other samples form global train dataset.
# - Assume p processors each of which holds sub random forest.
#   Each sub forest should train on `train_frac` * xn/p data samples,
#   where `train_frac` = 1 - `test_frac`.
#   For this, each processor should sample those `train_frac` * xn/p
#   data samples from global train dataset with replacement, i.e.,
#   globally, the random forest is trained on `train_frac` * xn samples,
#   however, some of them might occur multiple times while others might
#   not occur at all.
# To do so, each processor needs to have byte_pos array!
from typing import BinaryIO, Any
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype
import os.path
import numpy as np
from sklearn.model_selection import train_test_split


# TRULY PARALLEL DATALOADERS
# Helper functions
def _determine_line_starts(
        f: BinaryIO,
        comm: MPI.Comm,
        displs: np.ndarray,
        counts: np.ndarray
) -> (list, Any, int):
    """
    Determine line starts in bytes from CSV file.

    Parameters
    ----------
    f : BinaryIO
        handle of CSV file to read from
    comm : MPI.Comm
           communicator to use
    displs : np.ndarray
             displacements of byte chunk starts on each rank
    counts : np.ndarray
             counts of byte chunk on each rank

    Returns
    -------
    line_starts : list[int]
                  list of line starts in each byte chunk
    r : Any
        chunk read in on each rank
    lineter_len : int
                  number of chars used to indicate line end in file
    """
    # Set up communicator.
    rank, size = comm.rank, comm.size
    # Read bytes chunk and count linebreaks.
    # \r (Carriage Return) : Move cursor to line start w/o advancing to next line.
    # \n (Line Feed) : Move cursor down to next line w/o returning to line start.
    # \r\n (End Of Line) : Combination of \r and \n.

    lineter_len = 1  # Set number of line termination chars.
    line_starts = []  # Set up list to save line starts.
    f.seek(displs[rank])  # Jump to pre-assigned position in file.
    r = f.read(counts[rank])  # Read number of bytes from starting position.

    for pos, l in enumerate(r):  # Determine line breaks in bytes chunk.
        if chr(l) == "\n":  # Line terminated by '\n' only.
            if not chr(r[pos - 1]) == "\r":  # No \r\n.
                line_starts.append(pos + 1)
        elif chr(l) == "\r":
            if pos + 1 < len(r) and chr(r[pos + 1]) == "\n":  # Line terminated by '\r\n'.
                line_starts.append(pos + 2)
                lineter_len = 2
            else:  # Line terminated by '\r' only.
                line_starts.append(pos + 1)
    return line_starts, r, lineter_len


def _get_byte_pos_from_line_starts(
        line_starts: np.ndarray,
        file_size: int,
        lineter_len: int
) -> np.ndarray:
    """
    Get line starts and counts in byte from line starts to read lines via seek and read.

    Parameters
    ----------
    line_starts : np.ndarray
                  absolute positions of line starts in byte
    file_size : int
                absolute file size in byte
    lineter_len : int
                  number of line termination characters

    Returns
    -------
    lines_byte : np.ndarray
                 array containing vectors of (start, count) for lines in byte
    """
    lines_byte = []  # list for line starts and counts
    for idx in range(len(line_starts)):  # Loop through all line starts.
        if idx == len(line_starts) - 1:  # Special case for last line.
            temp = [
                line_starts[idx],  # line start in bytes
                file_size
                - line_starts[idx]
                - lineter_len,  # bytes count of line length via difference
            ]
        else:  # all other lines
            temp = [
                line_starts[idx],  # line start in bytes
                line_starts[idx + 1]
                - line_starts[idx]
                - lineter_len,  # bytes count of line length via difference
            ]
        lines_byte.append(temp)
    return np.array(lines_byte)


def _decode_bytes_array(
        byte_pos: np.ndarray,
        f: BinaryIO,
        sep: str = ",",
        encoding: str = "utf-8"
) -> list:
    """
    Decode lines from byte positions and counts.

    Parameters
    ----------
    byte_pos : np.ndarray
               vectors of line starts and lengths in bytes
    f : BinaryIO
        handle of CSV file to read from
    sep : str
          character used in file to separate entries
    encoding : str
               encoding used to decode entries from bytes

    Returns
    -------
    lines : list[np.ndarray]
            values read from CSV file as numpy array entries
            in float format
    """
    lines = []  # list for saving decoded lines
    byte_pos = byte_pos[byte_pos[:, 0].argsort()]  # Sort line starts of data items read from file in ascending order.
    for item in byte_pos:
        f.seek(item[0])  # Go to line start.
        line = f.read(item[1]).decode(encoding)  # Read specified number of bytes and decode.
        if len(line) > 0:
            sep_values = [float(val) for val in line.split(sep)]  # Separate values in each line.
            line = np.array(sep_values)  # Convert list of separated values to numpy array.
        lines.append(line)  # Append numpy data entry to output list.
    return lines


def _get_all_lines(
        comm: MPI.Comm,
        f: BinaryIO,
        file_size: int,
        header_lines: int,
        verbose: bool = False,
) -> np.ndarray:
    """
    Construct array with line starts and lengths in bytes from csv file handle.
    Parameters
    ----------
    comm: MPI.Comm
          communicator to use
    f: BinaryIO
       file handle
    file_size: int
               file size in bytes
    header_lines: int
                  header lines
    verbose: bool
             verbosity level

    Returns
    -------
    np.ndarray : array with line starts and lengths in bytes
    """
    # Set up communicator.
    rank, size = comm.rank, comm.size

    # Determine displs + counts of bytes chunk to read on each rank.
    base = file_size // size  # Determine base chunk size for each process.
    remainder = file_size % size  # Determine remainder bytes.
    counts = base * np.ones((size,), dtype=int)  # Construct array with each rank's chunk counts.
    if remainder > 0:  # Equally distribute remainder over respective ranks to balance load.
        counts[:remainder] += 1
    displs = np.concatenate(  # Determine displs via cumulative sum from counts.
        (np.zeros((1,), dtype=int), np.cumsum(counts, dtype=int)[:-1])
    )
    if rank == 0:
        print(f"File size is {file_size} bytes.")

    if rank == 0 and verbose:
        print(f"Displs {displs}, counts {counts} for reading bytes chunks from file.")

    # Determine line starts in bytes chunks on each rank.
    line_starts, r, lineter_len = _determine_line_starts(f, comm, displs, counts)
    if rank == 0:  # On rank 0, add very first line.
        line_starts = [0] + line_starts

    if verbose:
        print(f"[{rank}/{size}]: {len(line_starts)} line starts in chunk.")

    # Find correct starting point, considering header lines.
    # All-gather numbers of line starts in each chunk in `total_lines` array.
    total_lines = np.empty(size, dtype=int)
    comm.Allgather(
        [np.array(len(line_starts), dtype=int), MPI.INT],
        [total_lines, MPI.INT]
    )
    cum_sum = list(np.cumsum(total_lines))
    # Determine rank where actual data lines start,
    # i.e. remove ranks only containing header lines.
    start = next(i for i in range(size) if cum_sum[i] > header_lines)
    if verbose:
        print(f"[{rank}/{size}]: total_lines is {total_lines}.\ncumsum is {cum_sum}.\nstart is {start}.")

    if rank < start:  # Ranks containing only header lines.
        line_starts = []
    elif rank == start:  # Rank containing header + data lines.
        rem = header_lines - (0 if start == 0 else cum_sum[start - 1])
        line_starts = line_starts[rem:]

    # Share line starts of data samples across all ranks via Allgatherv.
    line_starts += displs[rank]  # Shift line starts on each rank according to displs.
    if verbose:
        print(f"[{rank}/{size}]: {len(line_starts)} line starts of shape {line_starts.shape} "
              f"and type {line_starts.dtype} in local chunk: {line_starts}")
    count_linestarts = np.array(  # Determine local number of line starts.
        len(line_starts), dtype=int
    )
    counts_linestarts = np.empty(  # Initialize array to all-gather local numbers of line starts.
        size, dtype=int
    )
    comm.Allgather([count_linestarts, MPI.INT], [counts_linestarts, MPI.INT])
    n_linestarts = np.sum(counts_linestarts)  # Determine overall number of line starts.
    displs_linestarts = (
        np.concatenate(  # Determine displacements of line starts from counts.
            (
                np.zeros((1,), dtype=int),
                np.cumsum(counts_linestarts, dtype=int)[:-1],
            ),
            dtype=int,
        )
    )
    if verbose and rank == 0:
        print(f"Overall {n_linestarts} linestarts.\n"
              f"Number of linestarts in each chunk is {counts_linestarts}.\n"
              f"Displs of linestarts in each chunk is {displs_linestarts}.")

    all_line_starts = np.empty(  # Initialize array to allgatherv line starts from all ranks.
        (n_linestarts,), dtype=line_starts.dtype
    )
    if verbose and rank == 0:
        print(
            f"Recvbuf {all_line_starts}, {all_line_starts.shape}, {all_line_starts.dtype}."
        )
    comm.Allgatherv(
        line_starts,
        [
            all_line_starts,
            counts_linestarts,
            displs_linestarts,
            from_numpy_dtype(line_starts.dtype),
        ],
    )
    # Line starts were determined as those positions following a line end.
    # But: There is no line after last line end in file.
    # Thus, remove last entry from all_line_starts.
    all_line_starts = all_line_starts[:-1]
    if rank == 0:
        print(f"After Allgatherv: All line starts: {all_line_starts}")
    # Construct array with line starts and lengths in bytes.
    print(f"[{rank}/{size}]: Construct array with line starts and lengths in bytes.")
    lines_byte = _get_byte_pos_from_line_starts(
        all_line_starts,
        file_size,
        lineter_len
    )
    return lines_byte


def _split_indices_train_test(
        n_samples: int,
        train_split: float = 0.9,
        seed: int = 0,
) -> (int, int, np.ndarray, np.ndarray):
    """
    Make index-based train test split with train shuffle.
    
    Parameters
    ----------
    n_samples : int
                overall number of samples in dataset
    train_split : float
                  fraction of dataset used for training
                  0 < train_split < 1
    seed : int
           seed used for random number generator

    Returns
    -------
    n_train_samples : int
                      number of train samples
    n_test_samples : int
                     number of test samples
    train_indices : np.ndarray
                    indices of samples in dataset used for training
    test_indices : np.ndarray
                   indices of samples in dataset used for testing
    """
    n_train = int(train_split * n_samples)  # Determine number of train samples from split.
    n_test = n_samples - n_train  # Determine number of test samples.
    indices = np.arange(0, n_samples)  # Construct array of all indices.
    # In the original paper, the last 500,000 SUSY samples form the test set.
    train_indices = indices[:n_train]  # First `n_train_samples` indices form train set.
    test_indices = indices[n_train:]  # Remaining ones form test set.
    rng = np.random.default_rng(seed=seed)  # Set same seed over all ranks for consistent shuffle.
    rng.shuffle(train_indices)  # Shuffle them.
    return n_train, n_test, train_indices, test_indices


def load_data_parallel_bytes(
        path_to_data: str,
        header_lines: int,
        comm: MPI.Comm = MPI.COMM_WORLD,
        random_state: int = 0,
        train_split: float = 0.9,
        sample_with_replacement: bool = True,
        sep: str = ",",
        verbose: bool = True,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Load data from CSV file in truly parallel fashion. Sample with replacement.

    Parameters
    ----------
    path_to_data : str
                   path to .csv file
    header_lines : int
                   number of header lines
    comm : MPI.Comm
           communicator to use
    random_state: int
                  random state
    train_split : float
                  train-test split fraction
    sample_with_replacement: bool
                             Whether to sample trainset with or without replacement.
    sep: str
         character used in file to separate entries
    verbose : bool
              verbosity level
              
    Returns
    -------
    train_samples : np.ndarray
                    rank-local train samples
    train_targets : np.ndarray
                    rank-local train targets
    test_samples : np.ndarray
                   (global) test samples
    test_targets : np.ndarray
                   (global) test targets
    """
    # Set up communicator.
    rank, size = comm.rank, comm.size
    with open(path_to_data, "rb") as f:  # Open csv file to read from.
        # Construct array with line starts and lengths in bytes.
        print(f"[{rank}/{size}]: Construct array with line starts and lengths in bytes.")
        lines_byte = _get_all_lines(
            comm,
            f,
            os.stat(path_to_data).st_size,  # Get file size in bytes.
            header_lines,
            verbose,
        )
        # Make global train-test split.
        print(f"[{rank}/{size}]: Make global train-test split.")
        n_train, n_test, train_indices, test_indices = _split_indices_train_test(
            n_samples=len(lines_byte),
            train_split=train_split,
            seed=random_state
        )
        # Construct held-out test dataset (same on each rank).
        print(f"[{rank}/{size}]: Decode {n_test} test samples from file.")
        test_lines = _decode_bytes_array(
            lines_byte[test_indices],
            f,
            sep=sep,
            encoding="utf-8"
        )
        test_samples = np.array(test_lines)[:, 1:]
        test_targets = np.array(test_lines)[:, 0]

        if verbose:
            print(f"[{rank}/{size}]: Test samples: {test_samples[0]}\n"
                  f"Test targets: {test_targets[0]}"
                  )
        # Construct train dataset (different on each rank).
        n_train_local = n_train // size  # Determine local train dataset size.
        remainder_train = n_train % size  # Balance load.

        if sample_with_replacement:
            if rank < remainder_train:
                n_train_local += 1
            rng = np.random.default_rng(seed=rank)
            print(f"[{rank}/{size}]: Draw local {n_train_local} train indices.")
            train_indices_local = rng.choice(train_indices, size=n_train_local)
        else:
            train_counts = n_train_local * np.ones(size, dtype=int)
            for idx in range(remainder_train):
                train_counts[idx] += 1
            train_displs = np.concatenate(
                (np.zeros(1, dtype=int), np.cumsum(train_counts, dtype=int)[:-1]), dtype=int
            )

            print(f"[{rank}/{size}]: Extract exclusive rank-local train subset of {train_counts[rank]} "
                  f"samples from train indices at positions {train_displs[rank]} to "
                  f"{train_displs[rank]+train_counts[rank]}."
                  )
            train_indices_local = train_indices[
                                  train_displs[rank]: train_displs[rank] + train_counts[rank]
                                  ]

        print(f"[{rank}/{size}]: Decode train lines from file.")
        train_lines_local = _decode_bytes_array(
            lines_byte[train_indices_local], f, sep=sep, encoding="utf-8"
        )
        train_samples = np.array(train_lines_local)[:, 1:]
        train_targets = np.array(train_lines_local)[:, 0]

    # Now, each rank holds a local train set (samples + targets)
    # and the (global) held-out test set (samples + targets).
    return train_samples, train_targets, test_samples, test_targets


def load_data_parallel_all(
        path_to_data: str,
        header_lines: int,
        comm: MPI.Comm,
        random_state: int,
        train_split: float = 0.9,
        sep: str = ",",
        verbose: bool = False
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Load data from CSV file. All ranks load all the data.

    Each rank samples its local train samples.

    Parameters
    ----------
    path_to_data : str
                   path to .csv file
    header_lines : int
                   number of header lines
    comm : MPI.Comm
           communicator to use
    random_state : int
                   seed used for sklearn train-test split
    train_split : float
                  train-test split fraction
    sep : str
          character used in file to separate entries
    verbose: bool
             verbosity level

    Returns
    -------
    train_samples_local : np.ndarray
                          rank-local train samples
    train_targets_local : np.ndarray
                          rank-local train targets
    test_samples : np.ndarray
                   global test samples
    test_targets : np.ndarray
                   global test targets
    """
    # Set up communicator.
    rank, size = comm.rank, comm.size

    # Load data into numpy array.
    data = np.loadtxt(
        path_to_data,
        dtype=float,
        delimiter=sep,
        skiprows=header_lines
    )
    if verbose:
        print(f"[{rank}/{size}]: Data loading done.")
    # Divide data into samples and targets.
    samples, targets = data[:, 1:], data[:, 0]
    # Perform train-test split.
    train_samples, test_samples, train_targets, test_targets = train_test_split(
        samples,
        targets,
        test_size=1-train_split,
        random_state=random_state
    )
    if verbose:
        print(f"[{rank}/{size}]: Data splitting done.")

    n_train = len(train_samples)   # Determine number of train samples.
    n_train_local = n_train // size  # Determine rank-local number of train samples.
    remainder_train = n_train % size
    # Determine load-balanced counts.
    if rank < remainder_train:
        n_train_local += 1
    if verbose:
        print(f"[{rank}/{size}]: There are {n_train} train and {len(test_samples)} test samples."
              f"\nLocal train samples: {n_train_local}"
              )
    # Sample `n_train_local` indices from train set with replacement.
    rng = np.random.default_rng(seed=random_state+rank)
    train_indices_local = rng.choice(range(n_train), size=n_train_local)
    # Return train dataset from drawn indices (repetitions possible!) and test dataset.
    return train_samples[train_indices_local], train_targets[train_indices_local], test_samples, test_targets


def load_data_parallel_poc(
        path_to_data: str,
        header_lines: int,
        comm: MPI.Comm,
        random_state: int,
        train_split: float = 0.9,
        sep: str = ",",
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Load data from CSV file. All processing elements load and keep all the data.

    Useful for comparing sequential baseline with distributed case,
    where each PE has all the data (should be the same!).

    Parameters
    ----------
    path_to_data : str
                   path to .csv file
    header_lines : int
                   number of header lines
    comm : MPI.Comm
           communicator to use
    random_state : int
                   seed used for sklearn train-test split
    train_split : float
                  train-test split fraction
    sep : str
          character used in file to separate entries
    Returns
    -------
    train_samples : np.ndarray
                    train samples
    train_targets : np.ndarray
                    train targets
    test_samples : np.ndarray
                   global test samples
    test_targets : np.ndarray
                   global test targets
    """
    rank, size = comm.rank, comm.size  # Set up communicator.
    data = np.loadtxt(  # Load data into array.
        path_to_data,
        dtype=float,
        delimiter=sep,
        skiprows=header_lines
    )
    print(f"[{rank}/{size}]: Data loading done.")
    samples, targets = data[:, 1:], data[:, 0]  # Divide data into samples and targets.
    train_samples, test_samples, train_targets, test_targets = train_test_split(  # Perform train-test split.
        samples,
        targets,
        test_size=1-train_split,
        random_state=random_state
    )
    print(f"[{rank}/{size}]: Data splitting done. \nReturning train and test data arrays...")
    return train_samples, train_targets, test_samples, test_targets


# ROOT-BASED DATALOADERS
def load_data_root(
        path_to_data: str,
        header_lines: int,
        comm: MPI.Comm,
        random_state: int,
        sample_with_replacement: bool = True,
        train_split: float = 0.9,
        sep: str = ",",
        verbose: bool = True
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Load data via root. Draw rank-local data subsets with replacement.

    Parameters
    ----------
    path_to_data : str
                   path to .csv file
    header_lines : int
                   number of header lines
    comm : MPI.Comm
           communicator to use
    random_state : int
                   seed used for sklearn train-test split
    sample_with_replacement: bool
                             Whether to sample train data with or without replacement.
    train_split : float
                  train-test split fraction
    sep : str
          character used in file to separate entries
    verbose : bool
              verbosity level
    Returns
    -------
    train_samples_local : np.ndarray
                          rank-local train samples
    train_targets_local : np.ndarray
                          rank-local train targets
    test_samples : np.ndarray
                   (global) test samples
    test_targets : np.ndarray
                   (global) test targets
    """
    # Set up communicator.
    rank, size = comm.rank, comm.size

    if rank == 0:
        # Load data into numpy array.
        data = np.loadtxt(
            path_to_data,
            dtype=float,
            delimiter=sep,
            skiprows=header_lines
        )
        # Divide data into samples and targets.
        samples, targets = data[:, 1:], data[:, 0]
        # Perform train-test split.
        train_samples, test_samples, train_targets, test_targets = train_test_split(
            samples,
            targets,
            test_size=1-train_split,
            random_state=random_state
        )

        n_train, n_test = len(train_samples), len(test_samples)  # Determine number of samples in train and test set.
        n_features = train_samples.shape[1]  # Determine number of features.
        n_train_local = n_train // size  # Determine rank-local number of train samples.
        remainder_train = n_train % size
        # Determine load-balanced counts and displacements.
        train_counts = n_train_local * np.ones(size, dtype=int)
        for idx in range(remainder_train):
            train_counts[idx] += 1
        train_displs = np.concatenate(
            (np.zeros(1, dtype=int), np.cumsum(train_counts, dtype=int)[:-1]), dtype=int
        )
        print(f"There are {n_train} train and {n_test} test samples.\n"
              f"Local train samples: {train_counts}"
              )
        # Sample `n_train` indices from train set with or without replacement.
        rng = np.random.default_rng(seed=random_state)
        if sample_with_replacement:  # Repetitions possible!
            train_indices = rng.choice(range(n_train), size=n_train)
        else:
            train_indices = rng.permutation(n_train)  # Repetitions not possible!
        # Construct train dataset from drawn indices.
        train_samples_shuffled = train_samples[train_indices]
        train_targets_shuffled = train_targets[train_indices]
        send_buf_train_samples = [
            train_samples_shuffled,
            train_counts * n_features,
            train_displs * n_features,
            from_numpy_dtype(train_samples_shuffled.dtype),
        ]
        send_buf_train_targets = [
            train_targets_shuffled,
            train_counts,
            train_displs,
            from_numpy_dtype(train_targets_shuffled.dtype),
        ]
    else:
        train_counts = None
        n_features = None
        n_test = None
        send_buf_train_samples = None
        send_buf_train_targets = None

    n_features = comm.bcast(n_features, root=0)
    n_test = comm.bcast(n_test, root=0)
    train_counts = comm.bcast(train_counts, root=0)
    train_samples_local = np.empty((train_counts[rank], n_features), dtype=float)
    train_targets_local = np.empty((train_counts[rank],), dtype=float)
    recv_buf_train_samples = [
        train_samples_local,
        from_numpy_dtype(train_samples_local.dtype),
    ]
    recv_buf_train_targets = [
        train_targets_local,
        from_numpy_dtype(train_targets_local.dtype),
    ]
    if rank != 0:
        test_samples = np.empty((n_test, n_features), dtype=float)
        test_targets = np.empty((n_test,), dtype=float)
    comm.Scatterv(send_buf_train_samples, recv_buf_train_samples, root=0)
    comm.Scatterv(send_buf_train_targets, recv_buf_train_targets, root=0)
    comm.Bcast(test_samples, root=0)
    comm.Bcast(test_targets, root=0)
    if verbose:
        print(f"[{rank}/{size}]: Train samples have shape {train_samples_local.shape}.\n"
              f"Train targets have shape {train_targets_local.shape}.\n"
              f"Test samples have shape {test_samples.shape}.\n"
              f"Test targets have shape {test_targets.shape}."
              )
    return train_samples_local, train_targets_local, test_samples, test_targets


def load_data_root_seq(
        path_to_data: str,
        header_lines: int,
        comm: MPI.Comm,
        random_state: int,
        train_split: float = 0.9,
        sep: str = ",",
        verbose: bool = True
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Load data from CSV file via root process with sequential Send and Recv.

    Parameters
    ----------
    path_to_data : str
                   path to .csv file
    header_lines : int
                   number of header lines
    comm : MPI.Comm
           communicator to use
    random_state : int
                   seed used for sklearn train-test split
    train_split : float
                  train-test split fraction
    sep : str
          character used in file to separate entries
    verbose : bool
              verbosity level

    Returns
    -------
    train_samples_local : np.ndarray
                          rank-local train samples
    train_targets_local : np.ndarray
                          rank-local train targets
    test_samples : np.ndarray
                   global test samples
    test_targets : np.ndarray
                   global test targets
    """
    # Set up communicator stuff.
    rank, size = comm.rank, comm.size

    if rank == 0:
        # Load data into numpy array.
        data = np.loadtxt(
                path_to_data, 
                dtype=float, 
                delimiter=sep,
                skiprows=header_lines
            )
        # Divide data into samples and targets.
        samples, targets = data[:, 1:], data[:, 0]
        # Perform train-test split.
        train_samples, test_samples, train_targets, test_targets = train_test_split(
            samples,
            targets,
            test_size=1-train_split,
            random_state=random_state
        )
        
        n_train, n_test = len(train_samples), len(test_samples)  # Determine number of train and test samples.
        n_features = train_samples.shape[1]  # Determine number of features.
        n_train_local = n_train // size  # Determine rank-local number of train samples.
        remainder_train = n_train % size
        # Determine load-balanced counts and displacements.
        train_counts = n_train_local * np.ones(size, dtype=int)
        for idx in range(remainder_train):
            train_counts[idx] += 1
        print(f"There are {n_train} train and {n_test} test samples."
              f"\nLocal train samples: {train_counts}"
              )
        # For each rank, sample `n_train_local` indices from train set with replacement.
        # Use different seed for each rank to make sure subsets are different.

    else:
        n_features = None
        n_test = None
        train_counts = None

    n_features = comm.bcast(n_features, root=0)
    n_test = comm.bcast(n_test, root=0)
    train_counts = comm.bcast(train_counts, root=0)

    if rank == 0:
        for idx_rank, n_train_samples_rank in enumerate(train_counts):
            if idx_rank != rank:
                rng = np.random.default_rng(seed=idx_rank)
                train_indices_local = rng.choice(range(n_train), size=n_train_samples_rank)
                train_samples_local = train_samples[train_indices_local]
                train_targets_local = train_targets[train_indices_local]
                print(f"Root {rank} about to send train data to rank {idx_rank}.")
                comm.Send(train_samples_local, dest=idx_rank, tag=9)
                comm.Send(train_targets_local, dest=idx_rank, tag=17)
                print("DONE.")
    else:
        train_samples_local = np.empty((train_counts[rank], n_features), dtype=float)
        train_targets_local = np.empty((train_counts[rank],), dtype=float)
        comm.Recv(train_samples_local, source=0, tag=9)
        comm.Recv(train_targets_local, source=0, tag=17)

    if rank == 0: 
        rng = np.random.default_rng(seed=rank)
        train_indices_local = rng.choice(range(n_train), size=n_train_samples_rank)
        train_samples_local = train_samples[train_indices_local]
        train_targets_local = train_targets[train_indices_local]

    else:
        test_samples = np.empty((n_test, n_features), dtype=float)
        test_targets = np.empty((n_test,), dtype=float)

    comm.Bcast(test_samples, root=0)
    comm.Bcast(test_targets, root=0)

    if verbose:
        print(f"[{rank}/{size}]: Train samples after Send/Recv have shape {train_samples_local.shape}.\n"
              f"Train targets after Send/Recv have shape {train_targets_local.shape}.\n"
              f"Test samples after Bcast have shape {test_samples.shape}.\n"
              f"Test targets after Bcast have shape {test_targets.shape}."
              )
    return train_samples_local, train_targets_local, test_samples, test_targets
