"""Frame checks that run before and after a fold loop.

Each raises rather than warns. A ten-fold run is expensive enough that a bad
frame should stop it at second zero, not at minute forty.
"""

import pandas as pd

from .config import LABEL_COL, STRATA_COL


def check_frame(df, model_cols, label_col=LABEL_COL, strata_col=STRATA_COL,
                verbose=True):
    """Verify the frame satisfies what a fold loop assumes.

    Raises
    ------
    KeyError
        The label or strata column is missing.
    ValueError
        The label or strata column is inside the feature set, or a feature
        column contains missing values.
    TypeError
        A feature column is not numeric.
    """
    missing = [c for c in (label_col, strata_col) if c not in df.columns]
    if missing:
        raise KeyError(f"missing required column(s): {missing}")

    leaked = [c for c in model_cols if c in (label_col, strata_col)]
    if leaked:
        raise ValueError(f"label/strata column inside the feature set: {leaked}")

    non_numeric = df[list(model_cols)].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise TypeError(f"non-numeric feature column(s): {non_numeric[:10]}")

    n_missing = int(df[list(model_cols)].isna().sum().sum())
    if n_missing:
        raise ValueError(f"{n_missing} missing values in the feature columns")

    if verbose:
        print(f"frame OK — {len(df):,} rows, "
              f"{df[strata_col].nunique():,} strata, "
              f"{len(model_cols):,} features, "
              f"label counts {df[label_col].value_counts().to_dict()}")


def check_all_predicted(probability):
    """Verify every row landed in exactly one test fold.

    NaN is the sentinel for "not yet predicted" — a numeric sentinel such as 100
    could be mistaken for a probability or a class label.
    """
    n_unfilled = int(pd.Series(probability).isna().sum())
    if n_unfilled:
        raise RuntimeError(f"{n_unfilled} rows never landed in a test fold")
