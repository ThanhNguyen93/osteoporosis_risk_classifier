"""Seeds, paths, the frame contract, and the column groups read from `config_ml.yaml`.

Shared by both stages: `osteo_pipeline.ml` and `osteo_pipeline.ann` read their
seed and fold count from here, so changing `SEED` once changes every run.
"""

from pathlib import Path

import yaml

SEED = 12345
N_SPLITS = 10
VAL_FRAC = 0.15  # share of TRAIN strata held out for validation (ANN only)

# --- frame contract ---------------------------------------------------------
# The two columns every stage assumes exist. `Strata` carries the matched pair,
# `osteo_label` the outcome.
STRATA_COL = "Strata"
LABEL_COL = "osteo_label"

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"

SOURCE_FILE = DATA_DIR / "osteo_features.csv"
OHE_FILE = DATA_DIR / "05_data_OHE.csv"  # to export later


def load_column_groups(config_file):
    """Read `config_ml.yaml` and return `(KEYS, GROUPS_RAW, GROUPS)`.

    `GROUPS_RAW` holds the study variable names as the config writes them; 
    the `_Ever` section matches on these. 
    `GROUPS` holds the engineered column names the models actually see: the two lab groups (`calcium_sodium`, `ever_worst`)
    are replaced by their `_decile` and `_measured` children.
    """
    cfg = yaml.safe_load(open(config_file))

    KEYS = cfg["keys"]
    GROUPS_RAW = {name: g["columns"] for name, g in cfg["groups"].items()}

    GROUPS = dict(GROUPS_RAW)

    LAB_COLS = GROUPS.pop("calcium_sodium") + GROUPS.pop("ever_worst")

    GROUPS["labs_decile"] = [c + "_decile" for c in LAB_COLS]
    GROUPS["labs_measured"] = [c + "_measured" for c in LAB_COLS]

    return KEYS, GROUPS_RAW, GROUPS


def make_group_of(GROUPS):
    """Build the reverse lookup `frame column -> config group`, dummy names included.

    Returns a `group_of(col)` function closed over that lookup.
    """
    COL_TO_GROUP = {c: g for g, cols in GROUPS.items() for c in cols}

    def group_of(col):
        if col in COL_TO_GROUP:
            return COL_TO_GROUP[col]
        for base, g in COL_TO_GROUP.items():  # "sex_F" -> "sex" -> demographics
            if col.startswith(base + "_"):
                return g
        return "unknown"

    return group_of
