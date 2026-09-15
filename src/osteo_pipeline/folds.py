"""Cross-validation folds — split on strata, not on rows.

One definition of a fold for both stages. `strata_folds` is the primitive: the
sorted unique strata and a `KFold` over them. The ML loop consumes it directly;
the ANN loop goes through `strata_splits`, which adds a validation block carved
out of train. Because both start from the same `strata_folds` call with the same
seed, the test blocks are identical across stages and the runs are comparable.
"""

from typing import Iterator, NamedTuple

import numpy as np
from sklearn.model_selection import KFold

from .config import N_SPLITS, SEED, VAL_FRAC


def strata_folds(strata, n_splits=N_SPLITS, seed=SEED):
    """Return `(uniq, fold)`: the unique strata and the `KFold` splitter over them.

    Built once and used by both versions, so all four runs see the same folds.
    """
    uniq = np.unique(strata)
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return uniq, fold


def check_pairs_intact(strata, uniq, fold):
    """Verify no pair straddles a fold boundary before fitting anything."""
    for tr_u, te_u in fold.split(uniq):
        tr = np.isin(strata, uniq[tr_u])
        te = np.isin(strata, uniq[te_u])
        assert not (set(strata[tr]) & set(strata[te]))
        assert tr.sum() % 2 == 0 and te.sum() % 2 == 0


class FoldStrata(NamedTuple):
    """The three disjoint strata blocks that make up one fold."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def strata_splits(
    strata, n_splits=N_SPLITS, val_frac=VAL_FRAC, seed=SEED
) -> Iterator[FoldStrata]:
    """Yield one `FoldStrata` per fold, with a validation block carved out of train.

    The test block is whatever `strata_folds` assigns — untouched, so it matches
    the ML stage fold for fold. The validation strata come out of TRAIN. `KFold`
    returns `train_index` sorted ascending, so slicing it directly would always
    take the lowest-numbered strata as validation; the train block is shuffled
    first. The generator is created once, outside the loop, so each fold gets a
    different shuffle while the whole run still reproduces from `seed`.

    `val_frac=0` gives the plain train/test split the ML stage uses.
    """
    uniq, fold = strata_folds(strata, n_splits=n_splits, seed=seed)
    rng = np.random.default_rng(seed)

    for train_index, test_index in fold.split(uniq):
        shuffled = rng.permutation(uniq[train_index])
        n_val = int(round(len(shuffled) * val_frac))
        yield FoldStrata(
            train=shuffled[n_val:],
            validation=shuffled[:n_val],
            test=uniq[test_index],
        )
