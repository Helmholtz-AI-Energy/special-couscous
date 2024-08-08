import logging
import string
import time
from typing import Any

from mpi4py import MPI

log = logging.getLogger(__name__)  # Get logger instance.


class MPITimer:
    """
    MPI timer for measuring execution times in distributed environments over all ranks.

    Attributes
    ----------
    comm : MPI.Comm
        The MPI communicator to use.
    elapsed_time_average : float
        The average elapsed time in seconds.
    end_time : float
        The rank-local end time in seconds.
    name : str
        Label describing what this timer measured, can be used for printing the results.
    output_format : str
        Format string template used for printing the output. May reference all attributes of the timer.
    print_on_exit : bool
        Whether to print the measured time in ``__exit__``.
    start_time : float
        The rank-local start time in seconds.

    Methods
    -------
    start()
        Start the timer.
    stop()
        Stop the timer.
    allreduce_for_average()
        Compute the global average using allreduce.
    print()
        Print the elapsed time using the given template.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        print_on_exit: bool = True,
        name: str = "",
        output_format: str = "Elapsed time {name}: global average {elapsed_time_average:.2g}s, "
        "local {elapsed_time_local:.2g}s",
    ) -> None:
        """
        Create a new distributed context-manager enabled timer.

        Parameters
        ----------
        comm : MPI.Comm
            The MPI communicator.
        print_on_exit : bool
            Whether to print the measured time in ``__exit__``.
        name : str
            Label describing what this timer measured, can be used for printing the results.
        output_format : str
            Format string template used for printing the output. May reference all attributes of the timer.
        """
        self.comm = comm
        self.output_format = output_format
        self.print_on_exit = print_on_exit
        self.name = name

        # NOTE: In the constructor, the following variables are only partially initialized in terms of their types as
        # initializing their values with None causes problems with mypy.
        self.start_time: float
        self.end_time: float
        self.elapsed_time_local: float
        self.elapsed_time_average: float

    def start(self) -> None:
        """Start the timer by setting the start time."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer by setting the end time and updating ``elapsed_time_local``."""
        self.end_time = time.perf_counter()
        self.elapsed_time_local = self.end_time - self.start_time

    def allreduce_for_average_time(self) -> None:
        """Compute the global average using allreduce and update ``elapsed_time_average``."""
        self.elapsed_time_average = (
            self.comm.allreduce(self.elapsed_time_local, op=MPI.SUM) / self.comm.size
        )

    def print(self) -> None:
        """Print the elapsed time using the given template."""
        template_keywords = {
            key for (_, key, _, _) in string.Formatter().parse(self.output_format)
        }
        template_kwargs = {
            key: value for key, value in vars(self).items() if key in template_keywords
        }
        log.info(self.output_format.format(**template_kwargs))

    def __enter__(self) -> "MPITimer":
        """
        Start the timer.

        Called on entering the respective context (i.e., with a 'with' statement).

        Returns
        -------
        MPITimer
            This timer object.
        """
        self.start()
        return self

    def __exit__(self, *args: tuple[Any, ...]) -> None:
        """
        Stop the timer, compute the global average, and optionally print the result on rank 0.

        Called on exiting the respective context (i.e., after a 'with' statement).

        Parameters
        ----------
        args : Any
            Unused, only to fulfill ``__exit__`` interface.
        """
        self.stop()
        self.allreduce_for_average_time()
        if self.print_on_exit and self.comm.rank == 0:
            self.print()
