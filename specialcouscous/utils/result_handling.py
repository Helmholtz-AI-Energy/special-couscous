import datetime
import logging
import pathlib
import uuid

import pandas

log = logging.getLogger(__name__)  # Get logger instance.


def construct_output_path(
    output_path: pathlib.Path | str = ".",
    output_name: str = "",
    experiment_id: str = "",
    mkdir: bool = True,
) -> tuple[pathlib.Path, str]:
    """
    Construct the path and filename to save results to based on the current time and date.

    Returns an output directory: 'output_path / year / year-month / date / YYYY-mm-dd' or the subdirectory
    'output_path / year / year-month / date / YYYY-mm-dd / experiment_id' if an ``experiment_id`` is given and a
    base name for output files: 'YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>'.
    All directories on this path are created automatically unless ``mkdir`` is set to False.

    Parameters
    ----------
    output_path : pathlib.Path | str
        The path to the base output directory to create the date-based output directory tree in.
    output_name : str
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : str
        If this is given, the file is placed in a further subdirectory of that name, i.e.,
        'output_path / year / year-month / date / experiment_id / <filename>.csv'.
        This can be used to group multiple runs of an experiment. Default is an empty string.
    mkdir : bool
        Whether to create all directories on the output path. Default is True.

    Returns
    -------
    pathlib.Path
        The path to the output directory.
    str
        The base file name.
    """
    today = datetime.datetime.today()
    path = (
        pathlib.Path(output_path)
        / str(today.year)
        / f"{today.year}-{today.month}"
        / str(today.date())
    )
    if experiment_id != "":
        path /= experiment_id
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    base_filename = (
        f"{today.strftime('%Y-%m-%d--%H-%M-%S')}-{output_name}-{str(uuid.uuid4())[:8]}"
    )
    return path, base_filename


def save_dataframe(
    dataframe: pandas.DataFrame, output_path: pathlib.Path | str
) -> None:
    """
    Safe the given dataframe as csv to ``output_path``.

    If the path does not end with the '.csv' suffix, the suffix is appended to the filename.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe to save as csv.
    output_path : pathlib.Path | str
        The path to save to dataframe to.
    """
    output_path = pathlib.Path(output_path)
    if output_path.suffix != ".csv":
        output_path = output_path.parent / (output_path.name + ".csv")
    log.info(f"Saving results to {output_path.absolute()}")
    dataframe["result_filename"] = output_path
    dataframe.to_csv(output_path, index=False)
