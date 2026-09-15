"""
Strata-based cross-validation loop for the ANN.

The loop itself is thin: it asks `osteo_pipeline.folds` for the strata blocks —
the same fold assignment the ML stage uses — `scaling` for fold-local
standardization, the builder for a fresh network, and `reporting` for the
summary. Everything it owns is bookkeeping: masks, write-back, and the per-fold
log line.

Fitting defaults live here rather than in the shared config because they are
ANN-only; `SEED`, `N_SPLITS`, and `VAL_FRAC` come from `osteo_pipeline.config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from ..checks import check_all_predicted
from ..config import LABEL_COL, N_SPLITS, SEED, STRATA_COL, VAL_FRAC
from ..folds import strata_splits


from .scaling import standardize

from .ANN_plotting import PLOT_FOLD_HISTORY
from .ANN_reporting import REPORT_OOF

if TYPE_CHECKING:  # keeps TensorFlow out of the import path of this module
    from .nets import Builder

EPOCHS = 30
BATCH_SIZE = 1024
THRESHOLD = 0.5  # probability cutoff for the 0/1 prediction

PROBABILITY_COL = "probability"
PREDICTION_COL = "all_preds"


@dataclass(frozen=True)
class TrainConfig:
    """Everything the loop needs that is not the data, the builder, or the columns."""

    n_splits: int = N_SPLITS
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    val_frac: float = VAL_FRAC
    threshold: float = THRESHOLD
    seed: int = SEED
    label_col: str = LABEL_COL
    strata_col: str = STRATA_COL
    verbose: int = 0  # Keras verbosity inside model.fit


@dataclass
class ANNRun:
    """Result of one cross-validation run."""

    predictions: pd.DataFrame  # input frame plus `probability` and `all_preds`
    histories: List[Dict[str, List[float]]] = field(default_factory=list)
    fold_train_accuracy: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Optional[TrainConfig] = None

    @property
    def auc(self) -> float:
        return self.metrics["auc"]

    @property
    def oof_accuracy(self) -> float:
        return self.metrics["oof_accuracy"]


def TRAIN_MODEL_ANN(
    ohe_data: pd.DataFrame,
    builder: "Builder",
    model_cols: Sequence[str],
    config: Optional[TrainConfig] = None,
    plot_folds: bool = False,
    save_plots_to: Optional[Path] = None,
) -> ANNRun:
    """
    Cross-validate the ANN with folds assigned over unique strata.

    Parameters
    ----------
    builder
        A builder from :func:`osteo_pipeline.ann.nets.make_ann`, not a compiled
        model — it is called once per fold so every fold starts from fresh
        weights.

    Returns
    -------
    ANNRun
        ``predictions`` is a copy of the input frame with two added columns:
        ``probability`` (the held-out probability from the fold that scored the
        row) and ``all_preds`` (that probability thresholded).
    """
    cfg = config or TrainConfig()

    # work on a copy so re-running the cell doesn't stack columns on the frame
    ohe_data = ohe_data.copy()
    ohe_data[PROBABILITY_COL] = np.nan
    ohe_data[PREDICTION_COL] = np.nan

    model_cols = list(model_cols)
    histories: List[Dict[str, List[float]]] = []
    fold_train_accuracy: List[float] = []

    splits = strata_splits(
        ohe_data[cfg.strata_col], n_splits=cfg.n_splits,
        val_frac=cfg.val_frac, seed=cfg.seed
    )
    for fold_number, blocks in enumerate(splits, start=1):
        # one boolean mask per block, defined once and reused. Reusing the mask
        # is what keeps X rows aligned with y rows and with the write-back.
        train_mask = ohe_data[cfg.strata_col].isin(blocks.train)
        val_mask = ohe_data[cfg.strata_col].isin(blocks.validation)
        test_mask = ohe_data[cfg.strata_col].isin(blocks.test)

        X_train, X_val, X_test = standardize(
            ohe_data.loc[train_mask, model_cols],
            ohe_data.loc[val_mask, model_cols],
            ohe_data.loc[test_mask, model_cols],
        )
        y_train = ohe_data.loc[train_mask, cfg.label_col].to_numpy(dtype="float32")
        y_val = ohe_data.loc[val_mask, cfg.label_col].to_numpy(dtype="float32")

        model = builder(len(model_cols))
        history = model.fit(
            X_train,
            y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=cfg.verbose,
            validation_data=(X_val, y_val),
        )
        histories.append(history.history)

        # Keras returns probabilities with shape (n, 1); ravel() flattens to (n,)
        test_prob = model.predict(X_test, verbose=0).ravel()
        ohe_data.loc[test_mask, PROBABILITY_COL] = test_prob
        ohe_data.loc[test_mask, PREDICTION_COL] = (test_prob > cfg.threshold).astype(int)

        train_prob = model.predict(X_train, verbose=0).ravel()
        train_acc = accuracy_score(y_train, (train_prob > cfg.threshold).astype(int))
        fold_train_accuracy.append(train_acc)

        print(
            f"fold {fold_number:>2}"
            f"  train strata {len(blocks.train):>6,}"
            f"  val strata {len(blocks.validation):>6,}"
            f"  test strata {len(blocks.test):>6,}"
            f"  train acc {train_acc:.4f}"
        )

        if plot_folds or save_plots_to is not None:
            PLOT_FOLD_HISTORY(
                history.history,
                fold_number,
                save_to=save_plots_to,
                show=plot_folds,
            )

    check_all_predicted(ohe_data[PROBABILITY_COL])
    ohe_data[PREDICTION_COL] = ohe_data[PREDICTION_COL].astype(int)

    print()
    metrics = REPORT_OOF(
        ohe_data[cfg.label_col],
        ohe_data[PREDICTION_COL],
        ohe_data[PROBABILITY_COL],
        train_accuracy=fold_train_accuracy,
    )

    return ANNRun(
        predictions=ohe_data,
        histories=histories,
        fold_train_accuracy=fold_train_accuracy,
        metrics=metrics,
        config=cfg,
    )
