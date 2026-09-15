"""
Fold-local standardization.

Mean and standard deviation are computed from TRAINING rows only and then
applied unchanged to validation and test — fitting the scaler on the full frame
would leak test information into the fit.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd


class Scaler(NamedTuple):
    """Column means and standard deviations learned from the training rows."""

    mean: pd.Series
    std: pd.Series


def fit_scaler(X_train: pd.DataFrame) -> Scaler:
    """
    Learn column means and SDs from the training rows.

    A constant column has SD 0 and would divide by zero; its SD is set to 1, so
    the column comes out as a constant 0 after centering rather than NaN.
    """
    return Scaler(mean=X_train.mean(), std=X_train.std().replace(0, 1))


def apply_scaler(
    X: pd.DataFrame, scaler: Scaler, dtype: str = "float32"
) -> np.ndarray:
    """Apply a fitted scaler and return an array in the dtype Keras casts to."""
    return ((X - scaler.mean) / scaler.std).to_numpy(dtype=dtype)


def standardize(
    X_train: pd.DataFrame, *others: pd.DataFrame, dtype: str = "float32"
) -> Tuple[np.ndarray, ...]:
    """
    Fit on ``X_train``, transform ``X_train`` and every frame in ``others``.

    Returned in the order given, so the call site reads::

        X_train, X_val, X_test = standardize(train_df, val_df, test_df)
    """
    scaler = fit_scaler(X_train)
    return tuple(apply_scaler(X, scaler, dtype=dtype) for X in (X_train, *others))
