"""The cross-validation loop and the per-version runner.

Folds are built from `np.unique(data["Strata"])` in both versions, so fold
membership is identical and the versions are directly comparable.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, roc_curve

from ..config import N_SPLITS, SEED
from ..folds import strata_folds

from .ML_models import make_model


def TRAIN_MODEL_ML(data_ohe, classifier, feature_cols, version,
                   n_splits=N_SPLITS, seed=SEED, return_models=False):
    """Out-of-fold predictions for one model family, one column set, one version.

    With `return_models=False` (the default) returns
    `(all_preds, probability, training_acc, testing_acc)`.
    With `return_models=True` the per-fold fitted estimators are appended as a
    fifth element — the variant notebook 05 uses to persist the XGB fits.
    """
    data = data_ohe.reset_index(drop=True).copy()

    X = data[["Strata"] + feature_cols]
    Y = data[["Strata", "osteo_label"]]

    # same fold construction the ANN stage uses — `osteo_pipeline.folds`
    inner_strata, inner_fold = strata_folds(data["Strata"], n_splits=n_splits, seed=seed)

    all_preds = np.full(len(data), -1)
    probability = np.full(len(data), np.nan)

    training_acc = []
    testing_acc = []
    fold_models = []

    for train_index, test_index in inner_fold.split(inner_strata):

        train_strata = inner_strata[train_index]
        test_strata = inner_strata[test_index]

        X_train = X.loc[X["Strata"].isin(train_strata)].drop(["Strata"], axis=1)
        X_test = X.loc[X["Strata"].isin(test_strata)].drop(["Strata"], axis=1)

        y_train = Y.loc[Y["Strata"].isin(train_strata)]["osteo_label"]
        y_test = Y.loc[Y["Strata"].isin(test_strata)]["osteo_label"]

        clf = make_model(classifier, version)
        fit = clf.fit(X_train, y_train)

        all_preds[y_test.index] = clf.predict(X_test)
        probability[y_test.index] = clf.predict_proba(X_test)[:, 1]

        training_acc.append(fit.score(X_train, y_train))
        testing_acc.append(fit.score(X_test, y_test))
        fold_models.append(clf)

    if return_models:
        return all_preds, probability, training_acc, testing_acc, fold_models
    return all_preds, probability, training_acc, testing_acc


def RUN_VERSION(data_ohe, y, classifiers, feature_cols, version,
                return_models=False, quiet=True):
    """One fold loop, both versions. Only the estimator settings differ.
    version = scaling or not

    `quiet=True` suppresses `ConvergenceWarning` for the duration of the loop.
    `SVM` uses `loss='hinge'` with `max_iter=5000` and does not converge on this
    frame in either version, so liblinear raises once per fold and the output is
    mostly warning text. Nothing else is filtered, and the fits themselves are
    unchanged — the solver still stops early either way. Pass `quiet=False` to
    see the warnings.
    """
    rows = []

    with warnings.catch_warnings():
        if quiet:
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for classifier in classifiers:
            result = TRAIN_MODEL_ML(
                data_ohe, classifier, feature_cols, version,
                return_models=return_models
            )

            if return_models:
                oof_pred, oof_prob, train_acc, test_acc, fold_models = result
            else:
                oof_pred, oof_prob, train_acc, test_acc = result

            fpr, tpr, threshold = roc_curve(y, oof_prob)

            row = {
                'classifier':   classifier,
                'FPR':          fpr,
                'TPR':          tpr,
                'AUC':          roc_auc_score(y, oof_prob),
                'training_acc': train_acc,
                'testing_acc':  test_acc,
                'y_pred':       oof_pred,
                'y_prob':       oof_prob,
            }
            if return_models:
                row['fold_models'] = fold_models

            rows.append(row)

    out = pd.DataFrame(rows).set_index('classifier')
    return out
