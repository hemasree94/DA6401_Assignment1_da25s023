import numpy as np


def accuracy_score(y_true, y_pred):
    """Compute the classification accuracy."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, labels=None):
    """Compute confusion matrix.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    labels : list or array, optional
        If provided, defines the label ordering.

    Returns
    -------
    np.ndarray
        Shape (n_labels, n_labels), where rows are true labels and cols are predicted.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.array(labels)

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred):
        if t not in label_to_index or p not in label_to_index:
            continue
        cm[label_to_index[t], label_to_index[p]] += 1

    return cm


def precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0):
    """Compute precision, recall, f1-score.

    This implementation supports the `average` modes used by the project.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    average : {None, "macro", "micro", "weighted"}
    zero_division : int or float
        Value to return when a metric is undefined (e.g., division by zero).

    Returns
    -------
    precision, recall, f1_score, support
        If `average` is not None, precision/recall/f1_score are scalars.
        `support` is always an array of per-class support counts.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)  # true (actual) counts
    predicted = cm.sum(axis=0).astype(float)  # predicted counts

    # per-class precision/recall
    with np.errstate(divide="ignore", invalid="ignore"):
        precision_per_class = np.where(predicted == 0, zero_division, tp / predicted)
        recall_per_class = np.where(support == 0, zero_division, tp / support)

    f1_per_class = np.where(
        (precision_per_class + recall_per_class) == 0,
        zero_division,
        2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class),
    )

    if average is None:
        return precision_per_class, recall_per_class, f1_per_class, support

    average = average.lower()
    if average == "micro":
        total_tp = tp.sum()
        total_pred = predicted.sum()
        total_support = support.sum()

        if total_pred == 0:
            precision = zero_division
        else:
            precision = total_tp / total_pred

        if total_support == 0:
            recall = zero_division
        else:
            recall = total_tp / total_support

        if precision + recall == 0:
            f1 = zero_division
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1, support

    if average == "weighted":
        weights = support / support.sum() if support.sum() != 0 else np.zeros_like(support)
        precision = float((precision_per_class * weights).sum())
        recall = float((recall_per_class * weights).sum())
        f1 = float((f1_per_class * weights).sum())
        return precision, recall, f1, support

    # Default to macro
    precision = float(np.mean(precision_per_class))
    recall = float(np.mean(recall_per_class))
    f1 = float(np.mean(f1_per_class))
    return precision, recall, f1, support
