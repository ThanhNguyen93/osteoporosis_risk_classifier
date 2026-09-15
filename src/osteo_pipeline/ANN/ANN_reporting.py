"""
Out-of-fold reporting.

Every row is scored by the one fold that held it out, so the metrics below are
computed once over the whole frame rather than averaged across folds.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def REPORT_OOF(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
    train_accuracy: Optional[Sequence[float]] = None,
    digits: int = 3,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Print and return the out-of-fold summary.

    AUC is computed from the PROBABILITY, not the 0/1 prediction: ``roc_curve``
    on hard predictions gives the curve a single interior point instead of a
    full sweep across thresholds.

    ``train_accuracy`` is the per-fold training accuracy, reported alongside the
    out-of-fold accuracy so the train/test gap is readable in one place.

    Note on the confusion matrix: sklearn's convention is row = actual,
    column = predicted, which is the transpose of most textbooks and of R's
    ``caret``.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics = {
        "auc": float(auc(fpr, tpr)),
        "oof_accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if train_accuracy is not None:
        metrics["mean_train_accuracy"] = float(np.mean(train_accuracy))

    if verbose:
        if train_accuracy is not None:
            print("mean train accuracy: ", round(metrics["mean_train_accuracy"], 4))
        print("out-of-fold accuracy:", round(metrics["oof_accuracy"], 4))
        print("AUC of whole model:  ", round(metrics["auc"], 4), "\n")
        print(
            "classification_report of whole model:\n",
            classification_report(y_true, y_pred, digits=digits),
            "\n",
        )
        print(
            "confusion_matrix of whole model:\n",
            confusion_matrix(y_true, y_pred),
            "\n",
        )

    return metrics
