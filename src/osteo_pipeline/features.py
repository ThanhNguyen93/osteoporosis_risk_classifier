"""One-hot encoding, pair integrity, and the `_Ever` / no-`_Ever` column split."""

import pandas as pd


def build_ohe_frame(one, KEYS, GROUPS):
    """One-hot encode the demographics block and assemble the model frame.

    Returns `(data_OHE, DEMO_COLS, MODEL_COLS)`.
    """
    ALL_VARS = [c for cols in GROUPS.values() for c in cols]
    study_vars = one[KEYS + ALL_VARS]

    numeric_subset = study_vars.select_dtypes("number")
    categorical_subset = pd.get_dummies(study_vars[GROUPS["demographics"]], dtype=int)

    data_OHE = pd.concat([numeric_subset, categorical_subset], axis=1)
    DEMO_COLS = list(categorical_subset.columns)
    MODEL_COLS = [c for c in data_OHE.columns if c not in KEYS]

    return data_OHE, DEMO_COLS, MODEL_COLS


def keep_intact_pairs(data_OHE):
    """Keep only complete rows, then only strata that are still a clean
    1-case / 1-control pair."""
    data_OHE = data_OHE.dropna()

    pair = data_OHE.groupby("Strata")["osteo_label"]
    data_OHE = data_OHE[pair.transform("size").eq(2) & pair.transform("sum").eq(1)]

    return data_OHE.reset_index(drop=True)


def split_ever_columns(MODEL_COLS, GROUPS_RAW):
    """Split `MODEL_COLS` into the `_Ever` columns and everything else.

    The name rule picks up any column containing `_Ever`; the cross-check walks
    `GROUPS_RAW["ever_worst"]`, whose base names generate the `_decile` /
    `_measured` children via `startswith()`. The two must agree.
    """
    # name rule
    EVER_COLS = []
    for col in MODEL_COLS:
        if "_Ever" in col:
            EVER_COLS.append(col)

    # cross-check against config: GROUPS_RAW["ever_worst"] holds the base names,
    # startswith() picks up the _decile / _measured children built from each base
    config_ever = []
    for base in GROUPS_RAW["ever_worst"]:
        for col in MODEL_COLS:
            if col.startswith(base):
                config_ever.append(col)

    assert set(EVER_COLS) == set(config_ever), "name rule and config disagree"

    NO_EVER_COLS = []
    for col in MODEL_COLS:
        if col not in EVER_COLS:
            NO_EVER_COLS.append(col)

    assert len(NO_EVER_COLS) + len(EVER_COLS) == len(MODEL_COLS)

    return EVER_COLS, NO_EVER_COLS
