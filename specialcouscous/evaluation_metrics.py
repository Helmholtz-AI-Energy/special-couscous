import numpy as np


def accuracy_score(confusion_matrix: np.ndarray) -> float:
    """
    Compute overall accuracy from the given confusion matrix, i.e., #correct predictions / #total samples.

    Based on ``sklearn.metrics.accuracy_score`` but computed from the confusion matrix instead of using the
    sample-wise labels and predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The multi-class classification confusion matrix (non-normalized).

    Returns
    -------
    float
        The overall accuracy computed for the given confusion matrix.
    """
    return confusion_matrix.trace() / confusion_matrix.sum()


def balanced_accuracy_score(confusion_matrix: np.ndarray) -> float:
    """
    Compute the balanced accuracy score as average of the class-wise recalls.

    Based on ``sklearn.metrics.balanced_accuracy_score`` but computed from the confusion matrix instead of using the
    sample-wise labels and predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The multi-class classification confusion matrix (non-normalized).

    Returns
    -------
    float
        The balanced accuracy computed for the given confusion matrix.
    """
    # Macro average = first compute class-wise recall, then average.
    return recall_score(confusion_matrix, average="macro")


def precision_recall_fscore(
    confusion_matrix: np.ndarray, beta: float = 1.0, average: str | None = None
) -> tuple[
    float | np.ndarray[float], float | np.ndarray[float], float | np.ndarray[float]
]:
    """
    Compute precision, recall, and f-beta score for the given confusion matrix of a multi-class classification model.

    The three metrics are either returned as class-wise values (if ``average == None``) or averaged using one of the
    following methods:
    - ``"micro"``: Metrics are computed globally, i.e., count total true/false positives/negatives for all samples,
      independent of class. This gives **equal importance to each sample** and should result in
      precision == recall == f-score == global accuracy.
    - ``"macro"``: Metrics are first computed independently for each class, the class-wise metrics are then averaged
      over all classes. This gives **equal importance to each class** but minority classes can outweigh majority
      classes. With balanced classes, ``"micro"`` and ``"macro"`` should be identical.
    - ``"weighted"``: Metrics are first computed independently for each class (as for macro), the class-wise metrics are
      then averaged over all classes, each **weighted by their support**. Note that this can result in an F-score that
      is not between precision and recall.
    This function is for multi-class but not multi-label classification, thus the average options ``"binary"`` and
    ``"samples"`` are not included.

    Based on ``sklearn.metrics.precision_recall_fscore_support`` but computed from the confusion matrix instead of using
    the sample-wise labels and predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The multi-class classification confusion matrix (non-normalized).
    beta : float
        The weight of recall in the F score (default: 1.0).
    average : str | None
        How to aggregate over classes. If ``None`` (default), the scores for each class are returned as array.
        Otherwise, the scores are aggregated to a single average score. Available averaging methods are: ``"micro"``,
        ``"macro"``, ``"weighted"``, and ``None`` for no averaging.

    Returns
    -------
    float | np.ndarray[float]
        The precision score either class-wise (if ``average == None``) or averaged over all classes using the specified
        averaging method.
    float | np.ndarray[float]
        The recall score either class-wise (if ``average == None``) or averaged over all classes using the specified
        averaging method.
    float | np.ndarray[float]
        The f-beta score either class-wise (if ``average == None``) or averaged over all classes using the specified
        averaging method.
    """
    n_samples = confusion_matrix.sum()
    n_correct = confusion_matrix.trace()

    supported_averages = ["micro", "macro", "weighted", None]
    if average not in supported_averages:
        raise ValueError(
            f"Invalid {average=}. Supported averages are: {supported_averages}."
        )

    if average == "micro":  # Compute metrics globally.
        accuracy = n_correct / n_samples
        return (
            accuracy,
            accuracy,
            accuracy,
        )  # Precision, recall, and F score are all the same.

    predicted_samples_per_class = confusion_matrix.sum(axis=0)
    true_samples_per_class = confusion_matrix.sum(axis=1)
    correct_predictions_per_class = confusion_matrix.diagonal()  # True positives
    false_positives_per_class = (
        predicted_samples_per_class - correct_predictions_per_class
    )
    false_negatives_per_class = true_samples_per_class - correct_predictions_per_class

    precision_per_class = correct_predictions_per_class / predicted_samples_per_class
    recall_per_class = correct_predictions_per_class / true_samples_per_class
    # Using the F-score definition (1+β²) TP / ((1+β²) TP + β² FN + FP):
    nominator = (1 + beta**2) * correct_predictions_per_class  # (1+β²) TP
    denominator = (  # ((1+β²) TP + β² FN + FP)
        (1 + beta**2) * correct_predictions_per_class
        + beta**2 * false_negatives_per_class
        + false_positives_per_class
    )
    f_score_per_class = nominator / denominator

    if average is None:  # Return raw metrics per class without aggregation.
        return precision_per_class, recall_per_class, f_score_per_class

    if (
        average == "weighted"
    ):  # Average metrics, class weighted by number of true samples with that label.
        class_weights = true_samples_per_class
    elif average == "macro":  # Average metrics, all classes have the same weight.
        class_weights = np.ones_like(true_samples_per_class)
    else:
        raise ValueError(f"No class weights supported for {average=}.")

    def average_with_weights(
        weights: np.ndarray, values: np.ndarray
    ) -> tuple[
        float | np.ndarray[float], float | np.ndarray[float], float | np.ndarray[float]
    ]:
        """
        Calculate average with provided weights.

        Parameters
        ----------
        weights : np.ndarray
            The weights.
        values : np.ndarray
            The values to average.

        Returns
        -------
        float
            The weighted average.
        """
        return (weights * values).sum() / weights.sum()

    precision = average_with_weights(class_weights, precision_per_class)
    recall = average_with_weights(class_weights, recall_per_class)
    f_score = average_with_weights(class_weights, f_score_per_class)
    return precision, recall, f_score


def precision_score(
    confusion_matrix: np.ndarray, average: str | None = None
) -> float | np.ndarray[float]:
    """
    Compute the precision score for the given confusion matrix of a multi-class classification model.

    The result is either returned as class-wise values (if ``average == None``) or averaged.
    Based on ``sklearn.metrics.precision_score`` but computed from the confusion matrix instead of using the
    sample-wise labels and predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The multi-class classification confusion matrix (non-normalized).
    average : str | None
        How to aggregate over classes. If ``None`` (default), the scores for each class are returned as array.
        Otherwise, the scores are aggregated to a single average score. See ``precision_recall_fscore`` for more details
        on the available averaging methods.

    Returns
    -------
    float | np.ndarray[float]
        The precision score either class-wise (if ``average == None``) or averaged over all classes using the specified
        averaging method.
    """
    precision, _, _ = precision_recall_fscore(confusion_matrix, average=average)
    return precision


def recall_score(
    confusion_matrix: np.ndarray, average: str | None = None
) -> float | np.ndarray[float]:
    """
    Compute the recall score for the given confusion matrix of a multi-class classification model.

    The result is either returned as class-wise values (if ``average == None``) or averaged.
    Based on ``sklearn.metrics.recall_score`` but computed from the confusion matrix instead of using the
    sample-wise labels and predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The multi-class classification confusion matrix (non-normalized).
    average : str | None
        How to aggregate over classes. If ``None`` (default), the scores for each class are returned as array.
        Otherwise, the scores are aggregated to a single average score. See ``precision_recall_fscore`` for more details
        on the available averaging methods.

    Returns
    -------
    float | np.ndarray[float]
        The recall score either class-wise (if average == None) or averaged over all classes using the specified
        averaging method.
    """
    _, recall, _ = precision_recall_fscore(confusion_matrix, average=average)
    return recall


def fbeta_score(
    confusion_matrix: np.ndarray, beta: float, average: str | None = None
) -> float | np.ndarray[float]:
    """
    Compute the F-beta score for the given confusion matrix of a multi-class classification model.

    The result is either returned as class-wise values (if ``average == None``) or averaged.
    Based on ``sklearn.metrics.fbeta_score`` but computed from the confusion matrix instead of using the
    sample-wise labels and predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The multi-class classification confusion matrix (non-normalized).
    beta : float
        The weight of recall in the F score.
    average : str | None
        How to aggregate over classes. If ``None`` (default), the scores for each class are returned as array.
        Otherwise, the scores are aggregated to a single average score. See precision_recall_fscore for more details on
        the available averaging methods.

    Returns
    -------
    float | np.ndarray[float]
        The f-beta score either class-wise (if ``average == None``) or averaged over all classes using the specified
        averaging method.
    """
    _, _, f_score = precision_recall_fscore(
        confusion_matrix, beta=beta, average=average
    )
    return f_score


def f1_score(
    confusion_matrix: np.ndarray, average: str | None = None
) -> float | np.ndarray[float]:
    """
    Compute the F1 score for the given confusion matrix of a multi-class classification model.

    The result is either returned as class-wise values (if ``average == None``) or averaged.
    Based on ``sklearn.metrics.f1_score`` but computed from the confusion matrix instead of using the
    sample-wise labels and predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The multi-class classification confusion matrix (non-normalized).
    average : str | None
        How to aggregate over classes. If ``None`` (default), the scores for each class are returned as array.
        Otherwise, the scores are aggregated to a single average score. See ``precision_recall_fscore`` for more details
        on the available averaging methods.

    Returns
    -------
    float | np.ndarray[float]
        The F1 score either class-wise (if ``average == None``) or averaged over all classes using the specified
        averaging method.
    """
    return fbeta_score(confusion_matrix, beta=1, average=average)


def cohen_kappa_score(confusion_matrix: np.ndarray) -> float:
    """
    Compute Cohen’s kappa, a measure for agreement between two annotators on a classification problem.

    Based on ``sklearn.metrics.cohen_kappa_score`` but computed from the confusion matrix instead of using the
    sample-wise labels and predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The multi-class classification confusion matrix (non-normalized).

    Returns
    -------
    float
        Cohen’s kappa computed for the given confusion matrix.
    """
    n_samples = confusion_matrix.sum()

    predicted_samples_per_class = np.sum(confusion_matrix, axis=0)
    true_samples_per_class = np.sum(confusion_matrix, axis=1)
    expected_confusion_matrix = (
        np.outer(predicted_samples_per_class, true_samples_per_class) / n_samples
    )

    expected_accuracy = (
        expected_confusion_matrix.diagonal().sum() / n_samples
    )  # = expected agreement p_e
    observed_accuracy = (
        confusion_matrix.diagonal().sum() / n_samples
    )  # = observed agreement p_o

    return (observed_accuracy - expected_accuracy) / (
        1 - expected_accuracy
    )  # = Cohen's kappa (p_o - p_e) / (1 - p_e)


def matthews_corrcoef(confusion_matrix: np.ndarray) -> float:
    """
    Compute Matthews correlation coefficient (MCC) for the given confusion matrix.

    Based on ``sklearn.metrics.matthews_corrcoef`` but computed from the confusion matrix instead of using the
    sample-wise labels and predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The multi-class classification confusion matrix (non-normalized).

    Returns
    -------
    float
        The Matthews correlation coefficient computed for the given confusion matrix.
    """
    predicted_samples_per_class = confusion_matrix.sum(axis=0)  # = p_k
    true_samples_per_class = confusion_matrix.sum(axis=1)  # = t_k
    n_samples = confusion_matrix.sum()  # = s
    n_correct = confusion_matrix.trace()  # = c

    # MCC = (c * s - t • p) / (sqrt(s^2 - p • p) * sqrt(s^2 - t • t))
    nominator_tp = n_correct * n_samples - np.dot(
        true_samples_per_class, predicted_samples_per_class
    )  # c * s - t•p
    denominator_predicted = n_samples**2 - np.dot(
        predicted_samples_per_class, predicted_samples_per_class
    )  # s^2 - p•p
    denominator_true = n_samples**2 - np.dot(
        true_samples_per_class, true_samples_per_class
    )  # s^2 - t•t
    denominator = np.sqrt(
        denominator_predicted * denominator_true
    )  # sqrt(s^2 - p • p) * sqrt(s^2 - t • t)

    return (
        0 if denominator == 0 else nominator_tp / denominator
    )  # MCC = (c*s - t•p) / sqrt((s^2 - p•p) * (s^2 - t•t))
