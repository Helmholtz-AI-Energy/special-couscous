import numpy as np
import sklearn.metrics

from specialcouscous import evaluation_metrics


class TestEvaluationMetrics:
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

