import copy
import glob
import logging
import pathlib
import pickle

import joblib
import numpy as np
import sklearn.tree
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_random_state

from specialcouscous.utils import get_pickled_size

log = logging.getLogger(__name__)  # Get logger instance.
"""Logger."""


class DistributedRandomForest:
    """
    Distributed random forest class.

    Attributes
    ----------
    acc_global : float
        The global accuracy of the random forest.
    acc_local : float
        The rank-local accuracy of the random forest.
    local_clf : sklearn.ensemble.RandomForestClassifier
        The rank-local random forest classifier.
    comm : MPI.Comm
        The MPI communicator.
    shared_global_model : bool
        Whether the local models are all-gathered to one global model shared by all ranks after training.
    n_trees_base : int
        The base number of rank-local trees.
    n_trees_global : int
        The number of trees in the global random forest model.
    n_trees_local : int
        The final number of rank-local trees.
    n_trees_remainder : int
        The remaining number of trees to distribute.
    random_state : int | None
        The rank-local random state of each local random forest classifier.
    trees : list[sklearn.tree.DecisionTreeClassifier]
        A list of all trees in the global random forest model.

    Methods
    -------
    train()
        Train random forest in parallel.
    test()
        Test the trained global random forest model.
    """

    def __init__(
        self,
        n_trees_global: int,
        comm: MPI.Comm,
        random_state: int | None = None,
        shared_global_model: bool = True,
        add_rank: bool = False,
        node_local_jobs: int = -1,
    ) -> None:
        """
        Initialize a distributed random forest object.

        Parameters
        ----------
        n_trees_global : int
            The number of trees in the global forest.
        comm : MPI.Comm
            The MPI communicator to use.
        random_state : int
            The base random state for the ``sklearn.ensemble.RandomForestClassifier`` objects.``
        shared_global_model : bool
            Whether the local models are all-gathered to one global model shared by all ranks after training.
        add_rank : bool
            Whether to use random integers generated from the model base seed (False) or the sum of the base seed and
            the local rank to initialize the local random forest model (True). Default is False.
        node_local_jobs : int
            The number of jobs to train the local classifiers with, passed as n_jobs to the underlying local
            classifiers. Default is -1 to use all available cores.
        """
        self.comm = comm  # Communicator to use
        # --- MODEL SETTINGS ---
        self.node_local_jobs = node_local_jobs
        self.n_trees_global = (
            n_trees_global  # Number of trees in global random forest model
        )
        # Distribute trees over available ranks in a load-balanced fashion.
        (
            self.n_trees_base,
            self.n_trees_remainder,
            self.n_trees_local,
        ) = self._distribute_trees()
        if random_state is not None:  # Particular random state is provided
            if add_rank:
                # Add the local rank to the provided base seed to obtain a rank-specific integer seed. Convert this seed to
                # a rank-specific ``np.random.RandomState`` instance. This ensures that each local subforest is different.
                local_seed = random_state + self.comm.rank
                log.info(
                    f"[{self.comm.rank}/{self.comm.size}] Use {local_seed} to seed the rank-local `RandomState` instance."
                )
                self.random_state = check_random_state(local_seed)
            else:
                # Convert the model base seed into a ``np.random.RandomState`` instance (the same on each rank). This
                # ``RandomState`` instance is used to generate a sequence of ``self.comm.size`` random integers. The random
                # integer at position ``self.comm.rank`` is used to seed a rank-local ``RandomState`` instance, ensuring
                # that each rank-local subforest is different.
                base_random_state = check_random_state(random_state)
                local_seeds = base_random_state.randint(
                    low=0, high=2**32 - 1, size=self.comm.size
                )  # NOTE: The seed range here is the maximum range possible.
                local_seed = local_seeds[self.comm.rank]  # Extract rank-local seed.
                log.info(
                    f"[{self.comm.rank}/{self.comm.size}] Use {local_seed} to seed the rank-local `RandomState` instance."
                )
                self.random_state = check_random_state(local_seed)
            log.debug(
                f"Random state of model is: {self.random_state.get_state(legacy=True)}"
            )
        else:  # Random state is None
            self.random_state = random_state

        self.shared_global_model = shared_global_model
        self.local_clf: RandomForestClassifier  # Local random forest classifier
        # Global random forest classifier, only with shared_global_model
        self.global_clf: RandomForestClassifier | None = None
        self.trees: list[
            sklearn.tree.DecisionTreeClassifier
        ]  # Global classifier as a list of all local trees

        # --- EVALUATION METRICS ---
        self.confusion_matrix_local: np.ndarray
        self.confusion_matrix_global: np.ndarray
        self.acc_global: float = (
            np.nan
        )  # Accuracy of global classifier on global evaluation dataset
        self.acc_local: float = (
            np.nan
        )  # Accuracy of rank-local classifier on local evaluation dataset
        # Only relevant for private test set + global model: Accuracy of global classifier on local evaluation dataset
        self.acc_global_local: float = np.nan

    def _distribute_trees(self) -> tuple[int, int, int]:
        """
        Distribute trees evenly over all processors.

        Returns
        -------
        int
            The base number of rank-local trees.
        int
            The remaining number of trees to distribute.
        int
            the final number of rank-local trees.
        """
        size, rank = self.comm.size, self.comm.rank
        n_trees_base = self.n_trees_global // size  # Base number of rank-local trees
        n_trees_remainder = (
            self.n_trees_global % size
        )  # Remaining number of trees to distribute
        n_trees_local = (
            n_trees_base + 1 if rank < n_trees_remainder else n_trees_base
        )  # Final number of local trees
        return n_trees_base, n_trees_remainder, n_trees_local

    def _train_local_classifier(
        self, train_samples: np.ndarray, train_targets: np.ndarray
    ) -> RandomForestClassifier:
        """
        Train random forest classifier.

        Params
        ------
        train_samples : numpy.ndarray
            The samples of the train dataset.
        train_targets : numpy.ndarray
            The targets of the train dataset.

        Returns
        -------
        sklearn.ensemble.RandomForestClassifier
            The trained model.
        """
        # Set up and train local subforest using rank-specific random state.
        clf = RandomForestClassifier(
            n_estimators=self.n_trees_local,
            random_state=self.random_state,
            n_jobs=self.node_local_jobs,
        )
        expected_n_jobs = 1 if clf.n_jobs is None else clf.n_jobs
        if expected_n_jobs < 0:
            expected_n_jobs = joblib.cpu_count() + 1 + expected_n_jobs
        log.info(f"Training local random forest with {expected_n_jobs} jobs.")
        clf.fit(train_samples, train_targets)
        return clf

    def load_checkpoints(
        self, checkpoint_path: str | pathlib.Path, uid: str = ""
    ) -> None:
        """
        Initialize rank-local subforests from pickled model checkpoints.

        Parameters
        ----------
        checkpoint_path : str | pathlib.Path
            Path to the checkpoint directory containing the local model pickle files to load. There should only be one
            valid checkpoint file for each rank, i.e., one file with the suffix "_rank_{comm.rank}.pickle}".
        uid : str
            The considered run's unique identifier
        """
        checkpoints = glob.glob(
            str(checkpoint_path) + f"/*{uid}_classifier_rank_{self.comm.rank}.pickle"
        )
        if len(checkpoints) > 1:
            raise ValueError(
                f"More than one local checkpoints found for rank {self.comm.rank}!"
            )
        if len(checkpoints) == 0:
            raise ValueError(f"No local checkpoints found for rank {self.comm.rank}!")
        with open(checkpoints[0], "rb") as f:
            self.local_clf = pickle.load(f)
            log.info(f"[{self.comm.rank}/{self.comm.size}]: Loaded {checkpoints[0]}.")

    @staticmethod
    def predict_histogram(
        classifier: RandomForestClassifier,
        samples: np.ndarray,
        n_classes: int | None = None,
    ) -> np.ndarray:
        """
        Perform tree-wise prediction with the given classifier and samples and aggregate the results as histogram.

        Similar to the class probabilities from sklearns predict_proba but without normalization.

        Parameters
        ----------
        classifier : RandomForestClassifier
            A local random forest.
        samples : numpy.ndarray
            The samples to perform prediction for.
        n_classes : int | None
            If given, correct the histogram to the given number of classes even when not all classes are present for the
            given classifier.

        Returns
        -------
        np.ndarray
            The number of votes per class for each input sample.
        """
        class_probabilities = classifier.predict_proba(samples)
        prediction_counts = np.round(
            class_probabilities * len(classifier.estimators_)
        ).astype(int)
        if n_classes is None:
            return prediction_counts
        histogram = np.zeros((len(samples), n_classes), dtype=int)
        histogram[:, np.round(classifier.classes_).astype(int)] = prediction_counts
        return histogram

    def predict_local_histogram(
        self, samples: np.ndarray, n_classes: int | None = None
    ) -> np.ndarray:
        """
        Perform local tree-wise prediction on the given samples and aggregate the results as histogram.

        Parameters
        ----------
        samples : numpy.ndarray
            The samples to perform prediction for.
        n_classes : int | None
            The number of classes in the global dataset.

        Returns
        -------
        np.ndarray
            The number of votes per class for each input sample.
        """
        return self.predict_histogram(self.local_clf, samples, n_classes)

    def predict_global_histogram(
        self, samples: np.ndarray, n_classes: int | None = None
    ) -> np.ndarray:
        """
        Perform global tree-wise prediction on the given samples and aggregate the results as histogram.

        Corresponds to the class probabilities from sklearns predict_proba.
        With a shared global model, this is identical to predict_local_histogram (as the local and global models are
        identical). Without a global model, the predictions are gathered from all ranks.

        Parameters
        ----------
        samples : numpy.ndarray
            The samples to perform prediction for.
        n_classes : int | None
            The number of classes in the global dataset.

        Returns
        -------
        np.ndarray
            The number of votes per class for each input sample.
        """
        if self.shared_global_model:
            # for a shared global model, predict directly on the global model
            return self.predict_histogram(self.global_clf, samples, n_classes)
        else:  # otherwise, aggregate the local predictions across ranks via all reduce
            local_histogram = self.predict_local_histogram(samples, n_classes)
            global_histogram = np.zeros_like(local_histogram)
            log.info(
                f"[{self.comm.rank}/{self.comm.size}]: before all-reduce {local_histogram.shape=}"
            )
            self.comm.Allreduce(local_histogram, global_histogram)
            message_size = get_pickled_size(local_histogram)
            log.info(
                f"All-reduce histogram: total message size send from rank {self.comm.rank} is {message_size} bytes"
                f"({message_size / len(samples)} bytes per sample at {len(samples)} samples)."
            )
            return global_histogram

    @staticmethod
    def predict_from_histogram(prediction_histogram: np.ndarray) -> np.ndarray:
        """
        Get the majority prediction (argmax) from a sample x class histogram.

        Parameters
        ----------
        prediction_histogram : np.ndarray
            The number of votes per class for each input sample.

        Returns
        -------
        np.ndarray
            The predicted class for each input sample.
        """
        return np.argmax(prediction_histogram, axis=1)

    def predict_local(
        self, samples: np.ndarray, n_classes: int | None = None
    ) -> np.ndarray:
        """
        Predict the class of the given samples using the local model.

        Parameters
        ----------
        samples : numpy.ndarray
            The samples to predict the classes for.
        n_classes : int | None
            The number of classes in the global dataset.

        Returns
        -------
        numpy.ndarray
            The predicted class for each input sample.
        """
        local_histogram = self.predict_local_histogram(samples, n_classes)
        return self.predict_from_histogram(local_histogram)

    def predict(self, samples: np.ndarray, n_classes: int | None = None) -> np.ndarray:
        """
        Predict the class of the given samples using the global model.

        Parameters
        ----------
        samples : numpy.ndarray
            The samples to predict the classes for.
        n_classes : int | None
            The number of classes in the global dataset.

        Returns
        -------
        numpy.ndarray
            The predicted class for each input sample.
        """
        global_histogram = self.predict_global_histogram(samples, n_classes)
        return self.predict_from_histogram(global_histogram)

    def _allgather_subforests_tree_by_tree(
        self,
    ) -> list[sklearn.tree.DecisionTreeClassifier]:
        """
        All-gather locally trained subforests tree by tree so that each processor finally holds complete global model.

        Returns
        -------
        list[sklearn.tree.DecisionTreeClassifier]
            A list of all trees in the global random forest model.
        """
        rank = self.comm.rank
        trees = []
        total_message_size = 0
        log.info(
            f"[{self.comm.rank}/{self.comm.size}] All-gathering subforests with {self.n_trees_base=}"
        )

        # All-gather first `n_trees_base` local trees.
        for t in range(self.n_trees_base):  # Loop over local trees.
            pickled_size = get_pickled_size(self.local_clf.estimators_[t])
            total_message_size += pickled_size
            log.info(
                f"[{self.comm.rank}/{self.comm.size}] sending tree {t}/{self.n_trees_base}. {pickled_size=}."
            )
            _trees = self.comm.allgather(self.local_clf.estimators_[t])
            log.info(
                f"[{self.comm.rank}/{self.comm.size}] done all-gathering step {t}/{self.n_trees_base}."
            )
            trees.append(_trees)

        log.info(
            f"[{self.comm.rank}/{self.comm.size}] Broadcasting remainder trees with {self.n_trees_remainder}"
        )
        # Broadcast remainder trees.
        if self.n_trees_remainder > 0:
            for r in range(self.n_trees_remainder):
                tree_temp = self.local_clf.estimators_[-1] if rank == r else None
                total_message_size += (
                    get_pickled_size(tree_temp) if tree_temp is not None else 0
                )
                trees.append([self.comm.bcast(tree_temp, root=r)])

        log.info(
            f"All-gather subforests: total message size send from {rank=} is {total_message_size} bytes."
        )
        return [tree for sublist in trees for tree in sublist]

    def train(
        self,
        train_samples: np.ndarray,
        train_targets: np.ndarray,
    ) -> None:
        """
        Train distributed random forest model in parallel.

        Parameters
        ----------
        train_samples : numpy.ndarray
            The rank-local train samples.
        train_targets : numpy.ndarray
            The corresponding train targets.
        """
        # Set up communicator.
        rank, size = self.comm.rank, self.comm.size
        # Set up and train local forest.
        log.info(
            f"[{rank}/{size}]: Set up and train rank-local random forest with {self.n_trees_local} trees."
        )
        self.local_clf = self._train_local_classifier(
            train_samples=train_samples,
            train_targets=train_targets,
        )

    def build_shared_global_model(self) -> None:
        """Build global shared random forest model from rank-local classifiers."""
        # Set up communicator.
        rank, size = self.comm.rank, self.comm.size
        log.info(
            f"[{rank}/{size}]: Sync global forest by all-gathering local forests tree by tree."
        )
        # Construct shared global model as globally shared list of all trees.
        self.trees = self._allgather_subforests_tree_by_tree()
        if self.global_clf is None:
            self.global_clf = copy.copy(self.local_clf)
        self.global_clf.estimators_ = self.trees
        log.info(f"[{rank}/{size}]: {len(self.trees)} trees in global forest.")

    def score(self, samples: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the mean global accuracy using .predict(), akin to sklearn .score().

        Parameters
        ----------
        samples : numpy.ndarray
            The samples to predict the classes for.
        targets : numpy.ndarray
            The corresponding targets.

        Returns
        -------
        float
            The mean accuracy over the given samples.
        """
        predictions = self.predict(samples)
        return float(sum(predictions == targets) / len(targets))

    def evaluate(
        self,
        samples: np.ndarray,
        targets: np.ndarray,
        shared_global_model: bool = False,
        n_classes: int | None = None,
    ) -> None:
        """
        Evaluate the trained global random forest.

        Parameters
        ----------
        samples : numpy.ndarray
            The rank-local samples to evaluate on.
        targets : numpy.ndarray
            The corresponding targets.
        shared_global_model : bool
            Whether the global model is shared among all ranks (True) or not (False). Default is False.
        n_classes : int | None
            The number of classes in the global dataset.
        """
        # compute predictions on local and global model
        local_predictions = self.predict_local(samples, n_classes)
        global_predictions = self.predict(samples, n_classes)

        # compute local and global accuracy and confusion matrices
        self.acc_local = (targets == local_predictions).mean()

        self.confusion_matrix_local = sklearn.metrics.confusion_matrix(
            targets, local_predictions
        )
        self.confusion_matrix_global = sklearn.metrics.confusion_matrix(
            targets, global_predictions
        )

        if shared_global_model:
            # with a shared global model, we can additionally compute the accuracy over potentially differing local data
            # count the number of samples and correct predictions of the global (shared) model on the local data
            n_correct_and_samples_local_data = np.array(
                [(targets == global_predictions).sum(), targets.shape[0]]
            )
            n_correct_and_samples_global_data = np.zeros_like(
                n_correct_and_samples_local_data
            )
            self.comm.Allreduce(
                n_correct_and_samples_local_data, n_correct_and_samples_global_data
            )
            n_correct, n_samples = n_correct_and_samples_global_data
            self.acc_global = n_correct / n_samples
            self.acc_global_local = (targets == global_predictions).mean()
        else:
            self.acc_global = (targets == global_predictions).mean()
            self.acc_global_local = (
                np.nan
            )  # without a shared global model, all test data must be shared

        log.info(
            f"[{self.comm.rank}/{self.comm.size}]: Accuracy of local model on local data is {self.acc_local}.\n"
            f"Accuracy of global model on local data is {self.acc_global_local}.\n"
            f"Accuracy of global model on global data is {self.acc_global}."
        )
