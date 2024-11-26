import numpy as np
import pytest
import sklearn.metrics

from specialcouscous import evaluation_metrics


class TestEvaluationMetrics:
    @staticmethod
    def first_and_fill_rest(first: float, fill: float, total_length: int) -> np.ndarray[float]:
        """
        Create a numpy array (1D) of length total_length where the first value is first and all remaining values are
        filled with the specified fill value.

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
            The created array [first, fill, ... fill] of length total_length.
        """
        return np.concat([np.array([first]), np.full(total_length - 1, fill)])

    def test_accuracy_score(self, n_classes=10):
        # balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, 1),  # all correct
            (y_true, (y_true + 1) % n_classes, 0),  # all false
            (y_true, np.zeros_like(y_true), 1 / n_classes),  # all zero = only first correct
        ]
        # imbalanced case: each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, 1),  # all correct
            (y_true, (y_true + 1) % n_classes, 0),  # all false
            (y_true, np.zeros_like(y_true), (1 + n_classes) / (2 * n_classes)),  # all zero = only first correct
        ]
        for y_true, y_pred, expected_accuracy in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            assert evaluation_metrics.accuracy_score(confusion_matrix) == expected_accuracy

    def test_balanced_accuracy_score(self, n_classes=10):
        # balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, 1),  # all correct
            (y_true, (y_true + 1) % n_classes, 0),  # all false
            (y_true, np.zeros_like(y_true), 1 / n_classes),  # all zero = only first correct
        ]
        # imbalanced case: each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, 1),  # all correct
            (y_true, (y_true + 1) % n_classes, 0),  # all false
            (y_true, np.zeros_like(y_true), 1 / n_classes),  # all zero = only first correct
        ]
        for y_true, y_pred, expected_accuracy in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            assert evaluation_metrics.balanced_accuracy_score(confusion_matrix) == expected_accuracy

    def test_precision_recall_fscore__totally_balanced(self, n_classes=10):
        # 100% balanced case: all classes have equal share of the labels and equal class wise accuracy
        classes = np.arange(n_classes)
        y_true = np.tile(classes, 5)  # each class appears 5 times
        # each class is predicted correcting 3 / 5 times -> class-wise accuracy is 60% for all classes
        y_pred = np.concat([np.tile(classes, 3), (np.tile(classes, 2) + 1) % n_classes])

        expected_class_wise_accuracy = 0.6
        expected_class_wise_accuracy_array = np.full(n_classes, expected_class_wise_accuracy)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

        # no average = class-wise scores, all scores are identical because everything is balanced
        actual = evaluation_metrics.precision_recall_fscore(confusion_matrix)
        for actual_score_array in actual:
            np.testing.assert_array_equal(actual_score_array, expected_class_wise_accuracy_array, strict=True)

        # all averages are identical because everything is balanced
        for average in ["micro", "macro", "weighted"]:
            actual_scores = evaluation_metrics.precision_recall_fscore(confusion_matrix, average=average)
            for actual_score in actual_scores:
                assert actual_score == pytest.approx(expected_class_wise_accuracy, 1e-6)

    @pytest.mark.skip("Test not yet implemented.")
    def test_precision_recall_fscore__balanced_labels_imbalanced_predictions(self):
        # balanced labels but imbalanced accuracy: class labels are balanced but different class-wise accuracies
        pass  # TODO: implement this test

    @pytest.mark.skip("Test not yet implemented.")
    def test_precision_recall_fscore__imbalanced(self):
        # completely imbalanced: both class labels and class accuracies are imbalanced
        pass  # TODO: implement this test

    def test_precision_score(self, n_classes=10):
        # balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, np.ones(n_classes)),  # all correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # all false
            # all zero: for first class: 5 correct predictions, but also 5 * (n_classes - 1) incorrect predictions
            # -> 1 / n_classes, no prediction at all for all other classes -> nan
            (y_true, np.zeros_like(y_true), self.first_and_fill_rest(1 / n_classes, np.nan, n_classes)),
        ]
        # imbalanced case: each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        n_samples = 2 * n_classes
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, np.ones(n_classes)),  # all correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # all false
            # all zero: for first class: 5 * (1 + n_classes) correct predictions, but also n_classes - 1 incorrect
            # predictions -> (1 + n_classes) / (2 * n_classes), no prediction at all for all other classes -> nan
            (y_true, np.zeros_like(y_true), self.first_and_fill_rest((1 + n_classes) / n_samples, np.nan, n_classes)),
        ]
        for y_true, y_pred, expected_precision in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_precision = evaluation_metrics.precision_score(confusion_matrix)
            np.testing.assert_array_equal(actual_precision, expected_precision, strict=True)

    def test_recall_score(self, n_classes=10):
        # balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, np.ones(n_classes)),  # all correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # all false
            # all zero: only first class correct
            (y_true, np.zeros_like(y_true), self.first_and_fill_rest(1., 0., n_classes)),
        ]
        # imbalanced case: each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, np.ones(n_classes)),  # all correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # all false
            # all zero: only first class correct
            (y_true, np.zeros_like(y_true), self.first_and_fill_rest(1., 0., n_classes)),
        ]
        for y_true, y_pred, expected_recall in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_recall = evaluation_metrics.recall_score(confusion_matrix)
            np.testing.assert_array_equal(actual_recall, expected_recall, strict=True)

    def test___f_score_from_precision_and_recall(self):
        # scalar inputs
        precision_recall_beta_expected_fscore = [  # if precision == recall, fscore == precision == recall
            (precision_recall, precision_recall, beta, precision_recall)
            for precision_recall in [0, 0.5, 1] for beta in [0.1, 1, 10]
        ] + [
            (1, 0, 1, 0),
            (0, 1, 1, 0),
            (0.25, 0.75, 1, 0.375),
            (0.75, 0.25, 1, 0.375),
            (0.25, 0.75, 10, 0.735),  # epsilon = 1e-2
            (0.75, 0.25, 10, 0.252),  # epsilon = 1e-2
        ]
        epsilon = 1e-2
        for precision, recall, beta, expected_fscore in precision_recall_beta_expected_fscore:
            actual_fscore = evaluation_metrics._f_score_from_precision_and_recall(precision, recall, beta)
            assert actual_fscore == pytest.approx(expected_fscore, epsilon)

        # array inputs
        precisions, recalls, betas, expected_fscores = [
            np.array(values) for values in zip(*precision_recall_beta_expected_fscore)]
        actual_fscores = evaluation_metrics._f_score_from_precision_and_recall(precisions, recalls, betas)
        np.testing.assert_allclose(actual_fscores, expected_fscores, atol=epsilon, strict=True)

    def test_fbeta_score(self, n_classes=10, beta=2):
        # balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_prediction_and_expected_accuracy = [
            (y_true, y_true, np.ones(n_classes)),  # all correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # all false
        ]
        # all zero: only first class correct
        precision = 1 / n_classes
        recall = 1
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        expected = self.first_and_fill_rest(f_beta, np.nan, n_classes)
        labels_prediction_and_expected_accuracy += [(y_true, np.zeros_like(y_true), expected)]

        # imbalanced case: each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_prediction_and_expected_accuracy += [
            (y_true, y_true, np.ones(n_classes)),  # all correct
            (y_true, (y_true + 1) % n_classes, np.zeros(n_classes)),  # all false
        ]
        # all zero: only first class correct
        precision = (1 + n_classes) / (2 * n_classes)
        recall = 1
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        expected = self.first_and_fill_rest(f_beta, np.nan, n_classes)
        labels_prediction_and_expected_accuracy += [(y_true, np.zeros_like(y_true), expected)]

        for y_true, y_pred, expected_fbeta in labels_prediction_and_expected_accuracy:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_fbeta = evaluation_metrics.fbeta_score(confusion_matrix, beta=beta)
            np.testing.assert_array_equal(actual_fbeta, expected_fbeta, strict=True)

    def test_f1_score(self, n_classes=10):
        # balanced case
        y_true = np.arange(n_classes).repeat(5)
        labels_and_predictions = [
            (y_true, y_true),  # all correct
            (y_true, (y_true + 1) % n_classes),  # all false
            (y_true, np.zeros_like(y_true)),  # all zero: only first class correct
        ]

        # imbalanced case: each class 5 times, except for first class: 5 * (1 + n_classes) times
        y_true = np.concat([np.arange(n_classes), np.zeros(n_classes)]).repeat(5)
        labels_and_predictions += [
            (y_true, y_true),  # all correct
            (y_true, (y_true + 1) % n_classes),  # all false
            (y_true, np.zeros_like(y_true))  # all zero: only first class correct
        ]

        for y_true, y_pred in labels_and_predictions:
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            actual_f1 = evaluation_metrics.f1_score(confusion_matrix)
            # we expect f1 to be identical to fbeta with beta = 1
            expected_f1 = evaluation_metrics.fbeta_score(confusion_matrix, beta=1)
            np.testing.assert_array_equal(actual_f1, expected_f1, strict=True)

    @pytest.mark.skip("Test not yet implemented.")
    def test_cohen_kappa_score(self):
        pass  # TODO: implement this test

    @pytest.mark.skip("Test not yet implemented.")
    def test_matthews_corrcoef(self):
        pass  # TODO: implement this test
