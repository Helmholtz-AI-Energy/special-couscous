import numpy as np
import pytest
import sklearn.metrics

from specialcouscous import evaluation_metrics


@pytest.mark.parametrize("n_classes", [2, 10, 100])
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
class TestEvaluationMetrics:
    """Test class to test all multi-class evaluation metrics for different numbers of classes (at least two)."""

    @staticmethod
    def first_and_fill_rest(
        first: float, fill: float, total_length: int
    ) -> np.ndarray[float]:
        """
        Create a numpy array (1D) for testing.

        The array has ``length total_length`` elements where the first value is `` first and all remaining values are
        filled with the specified fill value ``fill``.

        Parameters
        ----------
        first : float
            The value to set the first array element to.
        fill : float
            The value to set all other array elements to.
        total_length : int
            The total length of the array (including the first value).

        Returns
        -------
        np.ndarray[float]
            The created array [first, fill, ... fill] of ``length total_length``.
        """
        return np.concat([np.array([first]), np.full(total_length - 1, fill)])

    def test_accuracy_score(self, n_classes: int) -> None:
        """
        Test the accuracy score metric for a variable number of classes in both balanced and unbalanced cases.

        Comparing both to a manual expected value and the ``sklearn`` equivalent.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # Balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, 1),  # All correct
            (y_true, (y_true + 1) % n_classes, 0),  # All false
            (
                y_true,
                np.zeros_like(y_true),
                1 / n_classes,
            ),  # All zero = only first correct
        ]
        # Imbalanced case: Each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, 1),  # All correct
            (y_true, (y_true + 1) % n_classes, 0),  # All false
            (
                y_true,
                np.zeros_like(y_true),
                (1 + n_classes) / (2 * n_classes),
            ),  # All zero = only first correct
        ]
        for (
            y_true,
            y_pred,
            expected_accuracy_manual,
        ) in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            expected_accuracy_sklearn = sklearn.metrics.accuracy_score(y_true, y_pred)
            actual_accuracy = evaluation_metrics.accuracy_score(confusion_matrix)
            assert actual_accuracy == expected_accuracy_manual
            assert actual_accuracy == expected_accuracy_sklearn

    def test_balanced_accuracy_score(self, n_classes: int) -> None:
        """
        Test the balanced accuracy score metric for a variable number of classes in both balanced and imbalanced cases.

        Comparing both to a manual expected value and the ``sklearn`` equivalent.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # Balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, 1),  # All correct
            (y_true, (y_true + 1) % n_classes, 0),  # All false
            (
                y_true,
                np.zeros_like(y_true),
                1 / n_classes,
            ),  # All zero = only first correct
        ]
        # Imbalanced case: Each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, 1),  # All correct
            (y_true, (y_true + 1) % n_classes, 0),  # All false
            (
                y_true,
                np.zeros_like(y_true),
                1 / n_classes,
            ),  # All zero = only first correct
        ]
        for (
            y_true,
            y_pred,
            expected_accuracy_manual,
        ) in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            expected_accuracy_sklearn = sklearn.metrics.balanced_accuracy_score(
                y_true, y_pred
            )
            actual_accuracy = evaluation_metrics.balanced_accuracy_score(
                confusion_matrix
            )
            assert actual_accuracy == expected_accuracy_manual
            assert actual_accuracy == expected_accuracy_sklearn

    def test_precision_recall_fscore__totally_balanced(self, n_classes: int) -> None:
        """
        Test the ``precision_recall_fscore`` metric in the 100% balanced case.

        All classes have equal share of the labels and equal class wise accuracy. Comparing both to a manual expected
        value and the ``sklearn`` (near-)equivalent ``precision_recall_fscore_support``.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # 100% balanced case: All classes have equal share of the labels and equal class wise accuracy.
        classes = np.arange(n_classes)
        y_true = np.tile(classes, 5)  # Each class appears 5 times.
        # Each class is predicted correctly 3 / 5 times -> class-wise accuracy is 60% for all classes.
        y_pred = np.concat([np.tile(classes, 3), (np.tile(classes, 2) + 1) % n_classes])

        expected_class_wise_accuracy = 0.6
        expected_class_wise_accuracy_array = np.full(
            n_classes, expected_class_wise_accuracy
        )
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

        # No average = class-wise scores; all scores are identical because everything is balanced.
        actual = evaluation_metrics.precision_recall_fscore(confusion_matrix)
        expected_sklearn = sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred
        )
        for actual_score_array, expected_sklearn_array in zip(actual, expected_sklearn):
            np.testing.assert_array_equal(
                actual_score_array, expected_class_wise_accuracy_array, strict=True
            )
            np.testing.assert_array_equal(
                actual_score_array, expected_sklearn_array, strict=True
            )

        # All averages are identical because everything is balanced.
        for average in ["micro", "macro", "weighted"]:
            actual_scores = evaluation_metrics.precision_recall_fscore(
                confusion_matrix, average=average
            )
            expected_scores_sklearn = sklearn.metrics.precision_recall_fscore_support(
                y_true, y_pred, average=average
            )
            for actual_score, expected_score_sklearn in zip(
                actual_scores, expected_scores_sklearn
            ):
                assert actual_score == pytest.approx(expected_class_wise_accuracy, 1e-6)
                assert actual_score == pytest.approx(expected_score_sklearn, 1e-6)

    def test_precision_recall_fscore__balanced_labels_imbalanced_predictions(
        self, n_classes: int
    ) -> None:
        """
        Test the ``precision_recall_fscore metric`` with balanced class labels but different class-wise accuracies.

        Comparing both to a manual expected value and the ``sklearn`` (near-)equivalent
        ``precision_recall_fscore_support``.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # Balanced labels but imbalanced accuracy: Class labels are balanced but different class-wise accuracies.
        y_true = np.arange(n_classes).repeat(
            n_classes
        )  # Each class appears `n_classes` times, consecutively.
        # Class i is predicted correctly (n_classes - i) times (i.e., the larger i, the lower the recall,
        # class 0 has recall 1). All incorrect predictions predict class 0 instead (i.e., class 0 has low precision,
        # all other classes have precision 1).
        # To achieve this, we interpret the labels as square matrix (n_classes x n_classes, each row corresponds to one
        # class) and take the upper triangular matrix by setting all values below the diagonal to zero. Flattening this
        # matrix results in predicting zero the first i times for class i and i the remaining n_classes - i times.
        y_pred = np.triu(y_true.reshape(n_classes, n_classes)).flatten()

        incorrect_predictions = n_classes * (n_classes - 1) / 2
        total_predictions = n_classes * n_classes
        correct_predictions = total_predictions - incorrect_predictions
        precision_class_zero = n_classes / (n_classes + incorrect_predictions)

        expected_class_wise_precision = self.first_and_fill_rest(
            precision_class_zero, 1, n_classes
        )
        expected_class_wise_recall = (n_classes - np.arange(n_classes)) / n_classes

        nominator = expected_class_wise_precision * expected_class_wise_recall
        denominator = expected_class_wise_precision + expected_class_wise_recall
        expected_class_wise_f1 = 2 * nominator / denominator
        expected_class_wise_precision_recall_f1 = [
            expected_class_wise_precision,
            expected_class_wise_recall,
            expected_class_wise_f1,
        ]

        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

        # No average = class-wise scores, all scores are identical because everything is balanced.
        actual_precision_recall_f1 = evaluation_metrics.precision_recall_fscore(
            confusion_matrix
        )
        expected_scores_sklearn = sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred
        )
        for actual, expected_manual, expected_sklearn in zip(
            actual_precision_recall_f1,
            expected_class_wise_precision_recall_f1,
            expected_scores_sklearn,
        ):
            np.testing.assert_allclose(actual, expected_manual, atol=1e-6, strict=True)
            np.testing.assert_allclose(actual, expected_sklearn, atol=1e-6, strict=True)

        # Micro average of recall, precision, and F1 are all identical to the overall accuracy.
        expected_overall_accuracy = correct_predictions / total_predictions
        actual_precision_recall_f1 = evaluation_metrics.precision_recall_fscore(
            confusion_matrix, average="micro"
        )
        expected_scores_sklearn = sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred, average="micro"
        )
        for actual, expected_sklearn in zip(
            actual_precision_recall_f1, expected_scores_sklearn
        ):
            assert actual == pytest.approx(expected_overall_accuracy, 1e-6)
            assert actual == pytest.approx(expected_sklearn, 1e-6)

        # Macro average: Mean of class-wise scores, weighted average identical since true class distribution is balanced
        for average in ["macro", "weighted"]:
            actual_precision_recall_f1 = evaluation_metrics.precision_recall_fscore(
                confusion_matrix, average=average
            )
            expected_scores_sklearn = sklearn.metrics.precision_recall_fscore_support(
                y_true, y_pred, average=average
            )
            for actual, expected_class_wise, expected_sklearn in zip(
                actual_precision_recall_f1,
                expected_class_wise_precision_recall_f1,
                expected_scores_sklearn,
            ):
                expected_manual = expected_class_wise.mean()
                assert actual == pytest.approx(expected_manual, 1e-6)
                assert actual == pytest.approx(expected_sklearn, 1e-6)

    def test_precision_recall_fscore__imbalanced(self, n_classes: int) -> None:
        """
        Test ``precision_recall_fscore`` in the imbalanced case with both imbalanced class labels and class accuracies.

        Comparing both to a manual expected value and the ``sklearn`` (near-)equivalent
        ``precision_recall_fscore_support``.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # Completely imbalanced: Both class labels and class accuracies are imbalanced.
        # Class i appears (i + 1) * 2 times, consecutively.
        y_true = np.concat([np.full((i + 1) * 2, i) for i in range(n_classes)])
        # Class i is predicted correctly (i + 1) times (-> recall 50%).
        # When class is not predicted correctly, class (i + 1) % n_classes is predicted instead
        # (-> class i is predicted (i + 1) times correctly and ((i - 1) % n_classes + 1) times incorrectly, when
        # class (i - 1) % n_classes should have been predicted instead.
        # -> precision (i + 1) / (2i + 1) for i > 0 and 1 / (n + 1) for i = 0)
        # To achieve this, the first half of occurrences for each class are predicted correctly while for the second
        # half, the next class is predicted.
        y_pred = np.concat(
            [
                x
                for i in range(n_classes)
                for x in [np.full((i + 1), i), np.full((i + 1), (i + 1) % n_classes)]
            ]
        )

        total_predictions = len(y_true)
        correct_predictions = (y_true == y_pred).sum()

        classes = np.arange(n_classes)
        class_weights = (classes + 1) * 2
        expected_class_wise_precision = (classes + 1) / (
            classes + 2 + (classes - 1) % n_classes
        )
        expected_class_wise_recall = np.full(n_classes, 0.5)
        nominator = expected_class_wise_precision * expected_class_wise_recall
        denominator = expected_class_wise_precision + expected_class_wise_recall
        expected_class_wise_f1 = 2 * nominator / denominator
        expected_class_wise_precision_recall_f1 = [
            expected_class_wise_precision,
            expected_class_wise_recall,
            expected_class_wise_f1,
        ]

        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

        # no average = class-wise scores, all scores are identical because everything is balanced
        actual_precision_recall_f1 = evaluation_metrics.precision_recall_fscore(
            confusion_matrix
        )
        expected_scores_sklearn = sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred
        )
        for actual, expected_manual, expected_sklearn in zip(
            actual_precision_recall_f1,
            expected_class_wise_precision_recall_f1,
            expected_scores_sklearn,
        ):
            np.testing.assert_allclose(actual, expected_manual, atol=1e-6, strict=True)
            np.testing.assert_allclose(actual, expected_sklearn, atol=1e-6, strict=True)

        # Micro average of recall, precision, and F1 are all identical to the overall accuracy.
        expected_overall_accuracy = correct_predictions / total_predictions
        actual_precision_recall_f1 = evaluation_metrics.precision_recall_fscore(
            confusion_matrix, average="micro"
        )
        expected_scores_sklearn = sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred, average="micro"
        )
        for actual, expected_sklearn in zip(
            actual_precision_recall_f1, expected_scores_sklearn
        ):
            assert actual == pytest.approx(expected_overall_accuracy, 1e-6)
            assert actual == pytest.approx(expected_sklearn, 1e-6)

        # Macro average: Mean of class-wise scores
        actual_precision_recall_f1 = evaluation_metrics.precision_recall_fscore(
            confusion_matrix, average="macro"
        )
        expected_scores_sklearn = sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )
        for actual, expected_class_wise, expected_sklearn in zip(
            actual_precision_recall_f1,
            expected_class_wise_precision_recall_f1,
            expected_scores_sklearn,
        ):
            expected_manual = expected_class_wise.mean()
            assert actual == pytest.approx(expected_manual, 1e-6)
            assert actual == pytest.approx(expected_sklearn, 1e-6)

        # Weighted average: Mean of class-wise scores
        actual_precision_recall_f1 = evaluation_metrics.precision_recall_fscore(
            confusion_matrix, average="weighted"
        )
        expected_scores_sklearn = sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )
        for actual, expected_class_wise, expected_sklearn in zip(
            actual_precision_recall_f1,
            expected_class_wise_precision_recall_f1,
            expected_scores_sklearn,
        ):
            expected_manual = (
                expected_class_wise * class_weights
            ).sum() / class_weights.sum()
            assert actual == pytest.approx(expected_manual, 1e-6)
            assert actual == pytest.approx(expected_sklearn, 1e-6)

    def test_precision_recall_fscore__invalid_average(self, n_classes: int) -> None:
        """
        Test ``precision_recall_fscore`` with invalid average parameters. Should raise a ``ValueError``.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        y_true = np.arange(n_classes).repeat(5)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_true)
        invalid_averages = ["invalid_average", 1, True, 0.01234]
        for invalid_average in invalid_averages:
            with pytest.raises(ValueError):
                evaluation_metrics.precision_recall_fscore(
                    confusion_matrix,
                    average=invalid_average,  # type: ignore
                )

    def test_precision_score(self, n_classes: int) -> None:
        """
        Test the precision score metric for a variable number of classes in both balanced and unbalanced cases.

        Comparing both to a manual expected value and the ``sklearn`` equivalent.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # Balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, np.ones(n_classes)),  # All correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # All false
            # All zero: for first class: 5 correct predictions, but also 5 * (n_classes - 1) incorrect predictions
            # -> 1 / n_classes, no prediction at all for all other classes -> nan
            (
                y_true,
                np.zeros_like(y_true),
                self.first_and_fill_rest(1 / n_classes, np.nan, n_classes),
            ),
        ]
        # Imbalanced case: Each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        n_samples = 2 * n_classes
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, np.ones(n_classes)),  # All correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # All false
            # All zero: for first class: 5 * (1 + n_classes) correct predictions, but also n_classes - 1 incorrect
            # predictions -> (1 + n_classes) / (2 * n_classes), no prediction at all for all other classes -> nan
            (
                y_true,
                np.zeros_like(y_true),
                self.first_and_fill_rest(
                    (1 + n_classes) / n_samples, np.nan, n_classes
                ),
            ),
        ]
        for (
            y_true,
            y_pred,
            expected_precision_manual,
        ) in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_precision = evaluation_metrics.precision_score(confusion_matrix)
            expected_precision_sklearn = sklearn.metrics.precision_score(
                y_true, y_pred, average=None, zero_division=np.nan
            )
            np.testing.assert_array_equal(
                actual_precision, expected_precision_manual, strict=True
            )
            np.testing.assert_array_equal(
                actual_precision, expected_precision_sklearn, strict=True
            )

    def test_recall_score(self, n_classes: int) -> None:
        """
        Test the recall score metric for a variable number of classes in both balanced and unbalanced cases.

        Comparing both to a manual expected value and the sklearn equivalent.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # Balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, np.ones(n_classes)),  # all correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # all false
            # All zero: only first class correct
            (
                y_true,
                np.zeros_like(y_true),
                self.first_and_fill_rest(1.0, 0.0, n_classes),
            ),
        ]
        # Imbalanced case: Each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, np.ones(n_classes)),  # All correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # All false
            # All zero: only first class correct
            (
                y_true,
                np.zeros_like(y_true),
                self.first_and_fill_rest(1.0, 0.0, n_classes),
            ),
        ]
        for (
            y_true,
            y_pred,
            expected_recall_manual,
        ) in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_recall = evaluation_metrics.recall_score(confusion_matrix)
            expected_recall_sklearn = sklearn.metrics.recall_score(
                y_true, y_pred, average=None
            )
            np.testing.assert_array_equal(
                actual_recall, expected_recall_manual, strict=True
            )
            np.testing.assert_array_equal(
                actual_recall, expected_recall_sklearn, strict=True
            )

    @pytest.mark.parametrize("beta", [0.5, 1, 10, 100])
    def test_fbeta_score(self, n_classes: int, beta: float) -> None:
        """
        Test the balanced accuracy score in both balanced and imbalanced cases.

        Comparing both to a manual expected value and the ``sklearn`` equivalent.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        beta : float
            The beta parameter of the F-beta score.
        """
        # Balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, np.ones(n_classes)),  # All correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # All false
        ]
        # All zero: Only first class correct
        precision = 1 / n_classes
        recall = 1
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        expected = self.first_and_fill_rest(f_beta, 0, n_classes)
        labels_prediction_and_expected_accuracy += [
            (y_true, np.zeros_like(y_true), expected)
        ]

        # Imbalanced case: Each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, np.ones(n_classes)),  # All correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # All false
        ]
        # All zero: Only first class correct
        precision = (1 + n_classes) / (2 * n_classes)
        recall = 1
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        expected = self.first_and_fill_rest(f_beta, 0, n_classes)
        labels_prediction_and_expected_accuracy += [
            (y_true, np.zeros_like(y_true), expected)
        ]

        for (
            y_true,
            y_pred,
            expected_fbeta_manual,
        ) in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_fbeta = evaluation_metrics.fbeta_score(confusion_matrix, beta=beta)
            expected_fbeta_sklearn = sklearn.metrics.fbeta_score(
                y_true, y_pred, beta=beta, average=None, zero_division=np.nan
            )
            np.testing.assert_allclose(
                actual_fbeta, expected_fbeta_manual, atol=1e-6, strict=True
            )
            np.testing.assert_allclose(
                actual_fbeta, expected_fbeta_sklearn, atol=1e-6, strict=True
            )

    def test_f1_score(self, n_classes: int) -> None:
        """
        Test the F1 score metric for a variable number of classes in both balanced and unbalanced cases.

        Comparing both to a manual expected value and the ``sklearn`` equivalent.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # Balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_and_predictions = [
            (y_true, y_true),  # All correct
            (y_true, (y_true + 1) % n_classes),  # All false
            (y_true, np.zeros_like(y_true)),  # All zero: Only first class correct
        ]

        # Imbalanced case: Each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_and_predictions += [
            (y_true, y_true),  # All correct
            (y_true, (y_true + 1) % n_classes),  # All false
            (y_true, np.zeros_like(y_true)),  # All zero: Only first class correct
        ]

        for y_true, y_pred in labels_and_predictions:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_f1 = evaluation_metrics.f1_score(confusion_matrix)
            # We expect f1 to be identical to fbeta with beta = 1.
            expected_f1_manual = evaluation_metrics.fbeta_score(
                confusion_matrix, beta=1
            )
            expected_f1_sklearn = sklearn.metrics.f1_score(
                y_true, y_pred, average=None, zero_division=np.nan
            )
            np.testing.assert_array_equal(actual_f1, expected_f1_manual, strict=True)
            np.testing.assert_array_equal(actual_f1, expected_f1_sklearn, strict=True)

    def test_cohen_kappa_score(self, n_classes: int) -> None:
        """
        Test the Cohen's kappa score metric for a variable number of classes in both balanced and imbalanced cases.

        Comparing both to a manual expected value and the sklearn equivalent.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # Balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_and_predictions = [
            (y_true, y_true),  # All correct
            (y_true, (y_true + 1) % n_classes),  # All false
            (y_true, np.zeros_like(y_true)),  # All zero: Only first class correct
        ]

        # Imbalanced case: Each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_and_predictions += [
            (y_true, y_true),  # All correct
            (y_true, (y_true + 1) % n_classes),  # All false
            (y_true, np.zeros_like(y_true)),  # All zero: Only first class correct
        ]

        for y_true, y_pred in labels_and_predictions:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_kappa = evaluation_metrics.cohen_kappa_score(confusion_matrix)
            expected_kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
            assert actual_kappa == pytest.approx(expected_kappa, 1e-6)

    def test_matthews_corrcoef(self, n_classes: int) -> None:
        """
        Test the ``matthews_corrcoef`` metric for a variable number of classes in both balanced and imbalanced cases.

        Comparing both to a manual expected value and the ``sklearn`` equivalent.

        Parameters
        ----------
        n_classes : int
            The number of classes in the dataset generated for testing the metric.
        """
        # Balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_and_predictions = [
            (y_true, y_true),  # All correct
            (y_true, (y_true + 1) % n_classes),  # All false
            (y_true, np.zeros_like(y_true)),  # All zero: Only first class correct
        ]

        # Imbalanced case: Each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_and_predictions += [
            (y_true, y_true),  # All correct
            (y_true, (y_true + 1) % n_classes),  # All false
            (y_true, np.zeros_like(y_true)),  # All zero: Only first class correct
        ]

        for y_true, y_pred in labels_and_predictions:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_kappa = evaluation_metrics.matthews_corrcoef(confusion_matrix)
            expected_kappa = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
            assert actual_kappa == pytest.approx(expected_kappa, 1e-6)
