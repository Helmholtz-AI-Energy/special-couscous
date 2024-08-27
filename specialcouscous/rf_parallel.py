import logging

import numpy as np
import sklearn.tree
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_random_state

from specialcouscous.utils.timing import MPITimer

log = logging.getLogger(__name__)  # Get logger instance.


class DistributedRandomForest:
    """
    Distributed random forest class.

    Attributes
    ----------
    acc_global : float
        The global accuracy of the random forest.
    acc_local : float
        The rank-local accuracy of the random forest.
    clf : sklearn.ensemble.RandomForestClassifier
        The rank-local random forest classifier.
    comm : MPI.Comm
        The MPI communicator.
    global_model : bool
        Whether the local models are all-gathered to one global model shared by all ranks after training.
    n_trees_base : int
        The base number of rank-local trees.
    n_trees_global : int
        The number of trees in the global random forest model.
    n_trees_local : int
        The final number of rank-local trees.
    n_trees_remainder : int
        The remaining number of trees to distribute.
    random_state : int
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
        random_state: int,
        global_model: bool = True,
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
        global_model : bool
            Whether the local models are all-gathered to one global model shared by all ranks after training.
        """
        self.n_trees_global = n_trees_global
        self.comm = comm
        # Distribute trees over available ranks in a load-balanced fashion.
        (
            self.n_trees_base,
            self.n_trees_remainder,
            self.n_trees_local,
        ) = self._distribute_trees()
        # Convert the model base seed into a ``np.random.RandomState`` instance (the same on each rank). This
        # ``RandomState`` instance is used to generate a sequence of ``self.comm.size`` random integers. The random
        # integer at position ``self.comm.rank`` is used to seed a rank-local ``RandomState`` instance, ensuring that
        # each rank-local subforest is different.
        base_random_state = check_random_state(random_state)
        local_seeds = base_random_state.randint(
            low=0, high=2**32 - 1, size=self.comm.size
        )
        local_seed = local_seeds[self.comm.rank]  # Extract rank-local seed.
        log.info(
            f"[{self.comm.rank}/{self.comm.size}] Use {local_seed} to seed the rank-local `RandomState` instance."
        )
        self.random_state = check_random_state(local_seed)
        log.debug(
            f"Random state of model is: {self.random_state.get_state(legacy=True)}"
        )
        self.global_model = global_model
        self.clf: RandomForestClassifier  # Local random forest classifier
        self.trees: list[
            sklearn.tree.DecisionTreeClassifier
        ]  # Global classifier as a list of all local trees
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
            n_estimators=self.n_trees_local, random_state=self.random_state
        )
        clf.fit(train_samples, train_targets)
        return clf

    def _predict_tree_by_tree(self, samples: np.ndarray) -> np.ndarray:
        """
        Get predictions of all sub estimators in the random forest.

        Params
        ------
        samples : numpy.ndarray
            The samples whose class to predict.

        Returns
        -------
        numpy.ndarray
            The predictions of each sub estimator on the samples.
        """
        return np.array([tree.predict(samples) for tree in self.trees], dtype="H")

    @staticmethod
    def _calc_majority_vote(tree_wise_predictions: np.ndarray) -> np.ndarray:
        """
        Calculate majority vote from tree-wise predictions.

        Params
        ------
        numpy.ndarray
            The array with tree-wise predictions. Contains a vector for each tree in the forest with the classes it
            predicted for all samples.

        Returns
        -------
        numpy.ndarray
            The majority vote.
        """
        # Determine sample-wise predictions by transposing tree-wise predictions.
        # This array contains one vector for each sample with the predicted classes from all trees in the subforest.
        sample_wise_predictions = tree_wise_predictions.transpose()
        majority_vote: list = []
        # Loop over predictions of all trees for each sample.
        for sample_predictions in sample_wise_predictions:
            # Determine predicted classes and how often each class was predicted for the considered sample.
            class_values, class_counts = np.unique(
                sample_predictions, return_counts=True
            )
            # Take the majority vote for that sample, i.e., choose the class that was predicted most often.
            majority_vote.append(class_values[np.argmax(class_counts)])
        return np.array(majority_vote)

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

        # All-gather first `n_trees_base` local trees.
        for t in range(self.n_trees_base):  # Loop over local trees.
            trees.append(self.comm.allgather(self.clf.estimators_[t]))

        # Broadcast remainder trees.
        if self.n_trees_remainder > 0:
            for r in range(self.n_trees_remainder):
                tree_temp = self.clf.estimators_[-1] if rank == r else None
                trees.append([self.comm.bcast(tree_temp, root=r)])

        return [tree for sublist in trees for tree in sublist]

    def _predict_locally(self, samples: np.ndarray) -> np.ndarray:
        """
        Get predictions of all sub estimators in random forest.

        Parameters
        ----------
        samples : numpy.ndarray
            The samples whose class to predict.

        Returns
        -------
        numpy.ndarray
            The predictions of each sub estimator on the samples.
        """
        return np.array(
            [tree.predict(samples) for tree in self.clf.estimators_], dtype="H"
        )

    def _predicted_class_hist(
        self, tree_wise_predictions: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """
        Calculate global sample-wise distributions of predicted classes from rank-local tree-wise predictions.

        Parameters
        ----------
        tree_wise_predictions : numpy.ndarray
            The tree-wise predictions.
        n_classes : int
            The number of classes in the dataset.

        Returns
        -------
        numpy.ndarray
            The sample-wise distributions of the predicted classes.
        """
        log.debug(
            f"[{self.comm.rank}/{self.comm.size}] tree-wise predictions shape: {tree_wise_predictions.shape}"
        )
        sample_wise_predictions = tree_wise_predictions.transpose()
        predicted_class_hists_local = np.array(
            [
                np.bincount(sample_pred, minlength=n_classes)
                for sample_pred in sample_wise_predictions
            ]
        )
        # The hist arrays have `n_test_samples` entries with `n_classes` elements each.
        # The local hist holds the class prediction distribution for each test sample over the local forest.
        # Now we want to sum up those sample-wise distributions to obtain the global hist over the global forest.
        # From this global hist, we can determine the global majority vote for each test sample.
        predicted_class_hists_global = np.zeros_like(predicted_class_hists_local)
        log.debug(
            f"[{self.comm.rank}/{self.comm.size}]: predicted_class_hists_local: {predicted_class_hists_local.shape}, "
            f"{type(predicted_class_hists_local)}, {predicted_class_hists_local.dtype}\n"
            f"predicted_class_hists_global: {predicted_class_hists_global.shape}, "
            f"{type(predicted_class_hists_global)}, {predicted_class_hists_global.dtype}"
        )
        self.comm.Allreduce(predicted_class_hists_local, predicted_class_hists_global)
        return predicted_class_hists_global

    @staticmethod
    def _calc_majority_vote_hist(predicted_class_hists: np.ndarray) -> np.ndarray:
        """
        Calculate majority vote from sample-wise histograms of predicted classes.

        Parameters
        ----------
        predicted_class_hists : numpy.ndarray
            The sample-wise histograms of the predicted classes.

        Returns
        -------
        numpy.ndarray
            The majority votes for all samples in the histogram input.
        """
        return np.array(
            [np.argmax(sample_hist) for sample_hist in predicted_class_hists]
        )

    def train(
        self,
        train_samples: np.ndarray,
        train_targets: np.ndarray,
        global_model: bool = True,
    ) -> None | MPITimer:
        """
        Train random forest model in parallel.

        Parameters
        ----------
        train_samples : numpy.ndarray
            The rank-local train samples.
        train_targets : numpy.ndarray
            The corresponding train targets.
        global_model : bool
            Whether the global model shall be shared among all ranks (True) or not (False).

        Returns
        -------
        None | MPITimer
            A distributed MPI timer (if ``global_model`` is True).
        """
        # Set up communicator.
        rank, size = self.comm.rank, self.comm.size
        # Set up and train local forest.
        log.info(
            f"[{rank}/{size}]: Set up and train local random forest with {self.n_trees_local} trees."
        )
        self.clf = self._train_local_classifier(
            train_samples=train_samples,
            train_targets=train_targets,
        )
        if not global_model:
            return None
        with MPITimer(self.comm, name="all-gathering model") as timer:
            log.info(
                f"[{rank}/{size}]: Sync global forest by all-gathering local forests tree by tree."
            )
            trees = self._allgather_subforests_tree_by_tree()
            log.info(f"[{rank}/{size}]: {len(trees)} trees in global forest.")
            self.trees = trees
            return timer

    def evaluate(
        self,
        samples: np.ndarray,
        targets: np.ndarray,
        n_classes: int,
        global_model: bool = True,
    ) -> None:
        """
        Evaluate the trained global random forest.

        Parameters
        ----------
        samples : numpy.ndarray
            The rank-local samples to evaluate on.
        targets : numpy.ndarray
            The corresponding targets.
        n_classes : int
            The number of classes in the dataset.
        global_model : bool
            Whether the global model is shared among all ranks (True) or not (False). Default is True.
        """
        rank, size = self.comm.rank, self.comm.size
        if global_model:  # Global model is shared.
            tree_predictions = self._predict_tree_by_tree(samples)
            # Array with one vector for each tree in global RF with predictions for all test samples.
            # Final prediction of parallel RF is majority vote over all sub estimators.
            # Calculate majority vote.
            log.info(f"[{rank}/{size}]: Calculate majority vote.")
            majority_vote = self._calc_majority_vote(tree_predictions)
            # Calculate accuracy of global model over all local datasets.
            n_correct_local = (targets == majority_vote).sum()
            n_samples_local = targets.shape[0]
            log.info(
                f"[{rank}/{size}]: Fraction of correctly predicted samples on this rank: "
                f"{n_correct_local} / {n_samples_local}."
            )
            n_correct = self.comm.allreduce(n_correct_local)
            n_samples = self.comm.allreduce(targets.shape[0])
            log.info(
                f"[{rank}/{size}]: Fraction of correctly predicted samples overall:"
                f" {n_correct}/ {n_samples}"
            )
            # Calculate accuracy of global model on global dataset.
            self.acc_global = n_correct / n_samples
            # Calculate accuracy of global model on local dataset.
            # Note that this metric can only be calculated if the global model is shared among all ranks.
            self.acc_global_local = (targets == majority_vote).mean()

        else:  # Global model is not shared. Note that the dataset to be tested must be shared among all ranks.
            # Get class predictions of sub estimators in each forest.
            log.info(f"[{rank}/{size}]: Get predictions of individual sub estimators.")
            tree_predictions_local = self._predict_locally(samples)
            log.info(f"[{rank}/{size}]: Calculate majority vote via histograms.")
            predicted_class_hists = self._predicted_class_hist(
                tree_predictions_local, n_classes
            )
            majority_vote = self._calc_majority_vote_hist(predicted_class_hists)
            # Calculate global accuracy as accuracy of global model on local dataset which must be the same on each rank
            # if the model is not shared globally. The accuracy of the global model on the global dataset is not
            # accessible for this configuration and set to NaN. It should equal the accuracy of the global model on the
            # local dataset as the local dataset is the same as the global dataset.
            self.acc_global = (targets == majority_vote).mean()
            self.acc_global_local = np.nan

        # Calculate accuracy of local model on local dataset.
        self.acc_local = self.clf.score(samples, targets)
        log.info(
            f"[{rank}/{size}]: Accuracy of local model on local data is {self.acc_local}.\n"
            f"Accuracy of global model on local data is {self.acc_global_local}.\n"
            f"Accuracy of global model on global data is {self.acc_global}."
        )

    def test(
        self,
        test_samples: np.ndarray,
        test_targets: np.ndarray,
        n_classes: int,
        global_model: bool = True,
    ) -> None:
        """
        Test the trained global random forest.

        Parameters
        ----------
        test_samples : numpy.ndarray
            The rank-local test samples.
        test_targets : numpy.ndarray
            The corresponding test targets.
        n_classes : int
            The number of classes in the dataset.
        global_model : bool
            Whether the global model shall be shared among all ranks (True) or not (False).
        """
        rank, size = self.comm.rank, self.comm.size
        if global_model:
            tree_predictions = self._predict_tree_by_tree(test_samples)
            # Array with one vector for each tree in global RF with predictions for all test samples.
            # Final prediction of parallel RF is majority vote over all sub estimators.
            # Calculate majority vote.
            log.info(f"[{rank}/{size}]: Calculate majority vote.")
            majority_vote = self._calc_majority_vote(tree_predictions)
        else:
            # Get class predictions of sub estimators in each forest.
            log.info(f"[{rank}/{size}]: Get predictions of individual sub estimators.")
            tree_predictions_local = self._predict_locally(test_samples)
            log.info(f"[{rank}/{size}]: Calculate majority vote via histograms.")
            predicted_class_hists = self._predicted_class_hist(
                tree_predictions_local, n_classes
            )
            majority_vote = self._calc_majority_vote_hist(predicted_class_hists)
        # Calculate accuracies.
        self.acc_global = (test_targets == majority_vote).mean()
        self.acc_local = self.clf.score(test_samples, test_targets)
        log.info(
            f"[{rank}/{size}]: Local accuracy is {self.acc_local}, global accuracy is {self.acc_global}."
        )
