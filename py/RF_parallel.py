from mpi4py import MPI
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class DistributedRandomForest:
    """
    Distributed random forest class.
    """
    def __init__(
            self, 
            n_trees_global: int,
            comm: MPI.Comm,
            random_state: int,
            global_model: bool = True
    ) -> None:
        """
        Parameters
        ----------
        n_trees_global : int
                         number of trees in global forest
        comm : MPI.Comm
               MPI communicator to use
        random_state : int
                       base random state for sklearn RF
        global_model : bool
                       Whether the local models are all-gathered to one global model shared by all ranks after training.
        """
        self.n_trees_global = n_trees_global
        self.comm = comm
        self.n_trees_base, self.n_trees_remainder, self.n_trees_local = self._distribute_trees()
        self.random_state = random_state + self.comm.rank
        self.global_model = global_model
        self.clf = None  # local random forest classifier
        self.trees = None  # global classifier as a list of all local trees
        self.acc_global = None  # accuracy of global classifier
        self.acc_local = None  # accuracy of each rank-local classifier
        return

    def _distribute_trees(self) -> (int, int, int):
        """
        Distribute trees evenly over all processors.

        Returns
        -------
        int : base number of rank-local trees
        int : remaining number of trees to distribute
        int : final number of rank-local trees
        """
        size, rank = self.comm.size, self.comm.rank
        n_trees_base = self.n_trees_global // size  # base number of rank-local trees
        n_trees_remainder = self.n_trees_global % size  # remaining number of trees to distribute
        n_trees_local = n_trees_base + 1 if rank < n_trees_remainder else n_trees_base  # final number of local trees
        return n_trees_base, n_trees_remainder, n_trees_local
        
    def _train_local_classifier(
            self,
            train_samples: np.ndarray,
            train_targets: np.ndarray
    ) -> RandomForestClassifier:
        """
        Train random forest classifier.

        Params
        ------
        train_samples : numpy.ndarray
                        samples of train dataset
        train_targets : numpy.ndarray
                        targets of train dataset
        Returns
        -------
        sklearn.ensemble.RandomForestClassifier : trained model
        """
        clf = RandomForestClassifier(
            n_estimators=self.n_trees_local, 
            random_state=self.random_state
        )
        _ = clf.fit(train_samples, train_targets)
        return clf 

    def _predict_tree_by_tree(
            self,
            samples: np.ndarray
    ) -> np.ndarray:
        """
        Get predictions of all sub estimators in random forest.
        Params
        ------
        samples : numpy.ndarray
                  samples whose class to predict
        Returns
        -------
        numpy.ndarray : predictions of each sub estimator on samples
        """
        return np.array([tree.predict(samples) for tree in self.trees], dtype="H")

    @staticmethod
    def _calc_majority_vote(tree_wise_predictions: np.ndarray) -> np.ndarray:
        """
        Calculate majority vote from tree-wise predictions.
        Params
        ------
        numpy.ndarray : array with tree-wise predictions
        """
        sample_wise_predictions = tree_wise_predictions.transpose()
        majority_vote = []
        for sample_preds in sample_wise_predictions:
            class_values, class_counts = np.unique(sample_preds, return_counts=True)
            majority_vote.append(class_values[np.argmax(class_counts)])
        return np.array(majority_vote)

    def _allgather_subforests_tree_by_tree(self) -> list:
        """
        All-gather locally trained subforests tree by tree so that each processor finally holds complete global model.
        """
        rank, size = self.comm.rank, self.comm.size
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

    def _predict_locally(
            self,
            samples: np.ndarray
    ) -> np.ndarray:
        """
        Get predictions of all sub estimators in random forest.
        Params
        ------
        samples : numpy.ndarray
                  samples whose class to predict
        Returns
        -------
        numpy.ndarray : predictions of each sub estimator on samples
        """
        return np.array([tree.predict(samples) for tree in self.clf.estimators_], dtype="H")

    def _predicted_class_hist(
            self,
            tree_wise_predictions: np.ndarray,
            n_classes: int
    ) -> np.ndarray:
        """
        Calculate global sample-wise distributions of predicted classes from rank-local tree-wise predictions.
        
        Params
        ------
        tree_predictions : numpy.ndarray
                           tree-wise predictions
        n_classes : int
                    number of classes in dataset
        Returns
        -------
        numpy.ndarray : sample-wise distributions of predicted classes
        """
        sample_wise_predictions = tree_wise_predictions.transpose()
        predicted_class_hists_local = np.array(
            [np.bincount(sample_pred, minlength=n_classes) for sample_pred in sample_wise_predictions]
        )
        # The hist arrays have `n_test_samples` entries with `n_classes` elements each.
        # The local hist holds the class prediction distribution for each test sample over the local forest.
        # Now we want to sum up those sample-wise distributions to obtain the global hist over the global forest.
        # From this global hist, we can determine the global majority vote for each test sample.
        predicted_class_hists_global = np.zeros_like(predicted_class_hists_local)
        self.comm.Allreduce(predicted_class_hists_local, predicted_class_hists_global)
        return predicted_class_hists_global

    @staticmethod
    def _calc_majority_vote_hist(predicted_class_hists: np.ndarray) -> np.ndarray:
        """
        Calculate majority vote from sample-wise histograms of predicted classes.
    
        Params
        ------
        predicted_class_hists : numpy.ndarray
                                sample-wise histograms of predicted classes
        Returns
        -------
        numpy.ndarray : majority votes for all samples in histogram input
        """
        return np.array([np.argmax(sample_hist) for sample_hist in predicted_class_hists])

    def train(
        self,
        train_samples: np.ndarray,
        train_targets: np.ndarray,
        global_model: bool = True,
    ) -> None:
        """
        Train random forest in parallel.
    
        Params
        ------
        train_samples : numpy.ndarray
                        rank-local train samples
        train_targets : numpy.ndarray
                        corresponding train targets
        """
        # Set up communicator.
        rank, size = self.comm.rank, self.comm.size
        # Set up and train local forest.
        print(f"[{rank}/{size}]: Set up and train local random forest with "
              f"{self.n_trees_local} trees and random state {self.random_state}.")
        self.clf = self._train_local_classifier(
            train_samples=train_samples, 
            train_targets=train_targets,
        )
        if global_model:
            print(f"[{rank}/{size}]: Sync global forest by all-gathering local forests tree by tree.")
            trees = self._allgather_subforests_tree_by_tree()
            print(f"[{rank}/{size}]: {len(trees)} trees in global forest:\n{trees}")
            self.trees = trees
        return

    def test(
        self,
        test_samples: np.ndarray,
        test_targets: np.ndarray,
        n_classes: int,
        global_model: bool = True,
    ) -> None:
        """
        Test trained global random forest.
    
        Params
        ------
        test_samples : numpy.ndarray
                       rank-local test samples
        test_targets : numpy.ndarray
                       corresponding test targets
        n_classes : int
                    number of classes in dataset
        """
        rank, size = self.comm.rank, self.comm.size
        if global_model:
            tree_predictions = self._predict_tree_by_tree(test_samples)
            # Array with one vector for each tree in global RF with predictions for all test samples.
            # Final prediction of parallel RF is majority vote over all sub estimators.
            # Calculate majority vote.
            print(f"[{rank}/{size}]: Calculate majority vote.")
            majority_vote = self._calc_majority_vote(tree_predictions)
        else:
            # Get class predictions of sub estimators in each forest.
            print(f"[{rank}/{size}]: Get predictions of individual sub estimators.")
            tree_predictions_local = self._predict_locally(test_samples)
            print(f"[{rank}/{size}]: Calculate majority vote via histograms.")
            predicted_class_hists = self._predicted_class_hist(tree_predictions_local, n_classes)
            majority_vote = self._calc_majority_vote_hist(predicted_class_hists)
        # Calculate accuracies. 
        self.acc_global = (test_targets == majority_vote).mean()
        self.acc_local = self.clf.score(test_samples, test_targets)
        print(f"[{rank}/{size}]: Local accuracy is {self.acc_local}, global accuracy is {self.acc_global}.")
        return
