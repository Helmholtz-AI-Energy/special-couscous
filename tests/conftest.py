# DO NOT CHANGE THE NAME OF THIS FILE!
# In pytest, the conftest.py file serves as a means of providing fixtures for an entire directory. Fixtures defined in
# a contest.py can be used by any test in that package without needing to import them (pytest will automatically
# discover them. Also see https://docs.pytest.org/en/stable/reference/fixtures.html.
import pathlib
import shutil
from typing import Generator

import pytest


@pytest.fixture
def clean_mpi_tmp_path(
    mpi_tmp_path: pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    """
    Fixture to automatically clean up the temporary path after the test runs.

    Parameters
    ----------
    mpi_tmp_path : pathlib.Path
        The temporary path considered.
    """
    yield mpi_tmp_path
    # Automatically clean up the temporary directory after the test.
    shutil.rmtree(str(mpi_tmp_path), ignore_errors=True)
