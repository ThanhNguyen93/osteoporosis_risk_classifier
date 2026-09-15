"""
Network architecture.

The cross-validation loop needs a brand-new network every fold, so nothing here
hands back a compiled model: :func:`make_ann` returns a *builder*, and the loop
calls it once per fold with the feature count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras import layers

Builder = Callable[[int], tf.keras.Model]


@dataclass(frozen=True)
class ANNSpec:
    """
    Two hidden layers, dropout after the first, L2 on the second.

    Parameters
    ----------
    h1, h2
        Units in the first and second hidden layer.
    dropout
        Dropout rate applied after the first hidden layer.
    learning_rate
        Adam learning rate.
    l2
        Kernel regularizer on the second hidden layer. Any Keras regularizer or
        its string alias; ``None`` disables it.
    seed
        Weight-initialization seed. ``None`` leaves initialization to the global
        TensorFlow seed, which means fold results reproduce only when the loop
        runs start to finish in a fresh kernel. Set an integer to make each fold
        reproducible on its own — note this changes the numbers relative to a
        ``None`` run.
    """

    h1: int = 120
    h2: int = 100
    dropout: float = 0.5
    learning_rate: float = 0.001
    l2: Optional[str] = "l2"
    seed: Optional[int] = None

    def with_seed(self, seed: Optional[int]) -> "ANNSpec":
        """Copy of this spec with a different initialization seed."""
        return replace(self, seed=seed)


def make_ann(spec: Optional[ANNSpec] = None) -> Builder:
    """
    Return a builder ``build(n_features) -> compiled model``.

    Examples
    --------
    >>> builder = make_ann(ANNSpec(h1=120, h2=100))
    >>> model = builder(n_features=254)
    """
    spec = spec or ANNSpec()

    def build(n_features: int) -> tf.keras.Model:
        initializer = (
            tf.keras.initializers.GlorotUniform(seed=spec.seed)
            if spec.seed is not None
            else "glorot_uniform"
        )

        model = tf.keras.Sequential(
            [
                layers.Input(shape=(n_features,)),
                layers.Dense(
                    spec.h1,
                    activation="relu",
                    kernel_initializer=initializer,
                    name="h1",
                ),
                layers.Dropout(spec.dropout, seed=spec.seed, name="dropout_hidden"),
                layers.Dense(
                    spec.h2,
                    activation="relu",
                    kernel_regularizer=spec.l2,
                    kernel_initializer=initializer,
                    name="h2",
                ),
                layers.Dense(
                    1,
                    activation="sigmoid",
                    kernel_initializer=initializer,
                    name="output_layer",
                ),
            ],
            name="osteo_ann",
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=spec.learning_rate),
            metrics=["accuracy"],
        )
        return model

    build.spec = spec  # so a caller can read back what the builder builds
    return build


def summarize_ann(spec: ANNSpec, n_features: int) -> tf.keras.Model:
    """Build one throwaway model and print its layer table. Returns the model."""
    model = make_ann(spec)(n_features)
    model.summary()
    return model
