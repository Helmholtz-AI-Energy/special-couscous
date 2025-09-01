import gzip
import logging
import pathlib
import shutil
import urllib.request

import numpy as np
import pandas
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)  # Get logger instance.


class SUSYDataset:
    """
    The SUSY binary classification dataset from the UCI machine learning repository.

    This class handles downloading the data, reading the data, splitting it into both features and targets and train and
    test set, and optional feature normalization.
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"

    def __init__(
        self,
        data_dir: pathlib.Path | str,
        split_seed: int | np.random.RandomState = 0,
        train_split: float = 0.75,
        stratified_train_test: bool = False,
        normalize: bool = False,
    ):
        """
        Initialize the SUSY dataset.

        If necessary, download the raw data from the UCI machine learning repository. Load the raw data from csv and
        split into features and target variables and train and test samples. Optionally, normalize the features using
        the train mean and std.

        Parameters
        ----------
        data_dir : pathlib.Path | str
            Base directory containing the SUSY.csv. When downloading, the results are written to this directory.
        split_seed : int | np.random.RandomState
            Random seed used for the train test split.
        train_split : float
            Relative size of the train set.
        stratified_train_test : bool
            Whether to stratify the train-test split with the class labels.
        normalize : bool
            Whether to normalize the features with the train mean and std.
        """
        self.raw_data_path = pathlib.Path(data_dir) / "SUSY.csv"
        if not self.raw_data_path.exists():
            self.download()

        self._raw_data = pandas.read_csv(self.raw_data_path).values
        self.x = self._raw_data[:, 1:].astype(np.float32)
        self.y = self._raw_data[:, 0].astype(np.int32)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=1 - train_split,
            stratify=self.y if stratified_train_test else None,
            random_state=split_seed,
        )

        if normalize:
            train_mean = self.x_train.mean(axis=0)
            train_std = self.x_train.std(axis=0)
            self.x_train = self.normalize(self.x_train, train_mean, train_std)
            self.x_test = self.normalize(self.x_test, train_mean, train_std)

    @staticmethod
    def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Normalize a feature array (#samples, #features) with the given mean and std (#features,).

        Parameters
        ----------
        x : np.ndarray
            The data to normalize.
        mean : np.ndarray
            The mean to normalize with.
        std : np.ndarray
            The standard deviation to normalize with.

        Returns
        -------
        np.ndarray
            The normalized data.
        """
        return (x - mean[None, :]) / std[None, :]

    def download(self) -> None:
        """Download the SUSY.csv.gz and unpack the SUSY.csv raw data into the data directory."""
        log.info(f"Downloading SUSY dataset from {self.URL}.")
        # download data from URL and extract to self.raw_data_path
        http_response = urllib.request.urlopen(self.URL)
        gz_file_path = self.raw_data_path.with_suffix(".csv.gz")
        with open(gz_file_path, "wb") as gz_file:
            gz_file.write(http_response.read())
        log.debug("Download done, unpacking.")

        with gzip.open(gz_file_path, "rb") as gz_file:
            with open(self.raw_data_path, "wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)
        log.debug("Unpacking done.")
