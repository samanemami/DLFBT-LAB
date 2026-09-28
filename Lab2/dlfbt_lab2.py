# ===============================================================================
# DLFBT 2026/2027
# Lab assignment 2
# Authors:
#   Name1 NIA1 (complete your name and NIA here)
#   Name2 NIA2 (complete your name and NIA here)
# ===============================================================================

import numpy as np
import tensorflow as tf
from time import perf_counter
from dataclasses import dataclass
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def set_reproducible(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_phoneme():
    """
    Load the phoneme dataset from the follwing directory:
    "../data/lab2/phoneme.csv"

    The CSV contains:
    - several feature columns;
    - one final binary target column.

    Returns
    -------
    X:
        Feature matrix of shape (n_samples, n_features).
    y:
        Binary target vector of shape (n_samples,).

    TODO
    ----
    Load the dataset and split the last column from the remaining columns.
    """

    # TODO: load the CSV.
    # dataset = ...

    # TODO: perform a shape / validity check.

    # TODO:
    # X = ...
    # y = ...

    raise NotImplementedError("TODO: implement load_phoneme")


def dataset_overview(X, y):
    labels, counts = np.unique(y, return_counts=True)
    return {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "class_distribution": {int(k): int(v) for k, v in zip(labels, counts)},
        "feature_mean": np.mean(X, axis=0),
        "feature_std": np.std(X, axis=0),
    }


@dataclass
class DataSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    scaler: Optional[StandardScaler]


def prepare_data(
    X,
    y,
    *,
    test_size=0.20,
    val_size=0.20,
    normalize=True,
    random_state=RANDOM_STATE,
):
    """
    Create stratified training, validation and test partitions.

    Important note:

        FIT THE SCALER USING ONLY THE TRAINING DATA.

    Note that;
    ----
    If information from validation or test samples is used to compute
    preprocessing statistics, information leaks into the training pipeline.
    That makes the reported evaluation too optimistic.

    Parameters
    ----------
    X, y:
        Full dataset.
    test_size:
        Fraction of the full dataset assigned to the test partition.
    val_size:
        Fraction of the full dataset assigned to validation.
    normalize:
        If True, standardize features using StandardScaler.
    random_state:
        Seed passed to train_test_split.

    Returns
    -------
    DataSplit
        Object containing the three partitions and the fitted scaler.

    TODO
    ----
    Implement the full preprocessing pipeline.
    """

    # TODO: validate test_size and val_size.

    # TODO: first split -> train+validation and test.

    # TODO: compute the validation fraction relative to train+validation.

    # TODO: second split -> train and validation.

    # TODO: optionally fit StandardScaler ONLY on X_train.

    # TODO: return DataSplit(...)

    raise NotImplementedError("TODO: implement prepare_data")


def build_baseline_model(input_dim, hidden_units=8):
    """
    Build the small baseline multilayer perceptron used in Part 1.

    Architecture
    ------------
    Input
      -> Dense(hidden_units, activation="relu")
      -> Dense(1, activation="sigmoid")

    Important
    ---------
    This function should BUILD the model, but it should NOT compile it.

    TODO
    ----
    Create and return the Keras Sequential model.
    """

    # TODO:
    # model = tf.keras.Sequential([...])

    raise NotImplementedError("TODO: implement build_baseline_model")


def compile_binary_model(
    model,
    optimizer="adam",
    learning_rate=None,
):
    """
    Compile a Keras model for binary classification.

    Required settings
    -----------------
    loss:
        "binary_crossentropy"

    metric:
        "accuracy"

    optimizer:
        May be either:
        - a ready-created Keras optimizer object; or
        - one of the supported strings:
              "sgd"
              "adagrad"
              "rmsprop"
              "adam"

    TODO
    ----
    Compile the model and return it.
    """

    # TODO:
    # If optimizer is a string AND learning_rate is provided,
    # create the appropriate tf.keras.optimizers.* object.

    # TODO:
    # model.compile(
    #
    # )

    raise NotImplementedError("TODO: implement compile_binary_model")


def make_early_stopping(patience=20):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )


def train_model(
    model,
    split,
    *,
    epochs=250,
    batch_size=32,
    patience=20,
    verbose=0,
):
    """
    Train a model using the explicit training and validation partitions.

    ------------------
    Use:

        split.X_train
        split.y_train

    for fitting and:

        split.X_val
        split.y_val

    as validation_data.

    Also include the EarlyStopping callback returned by
    `make_early_stopping(patience)`.

    Returns
    -------
    history:
        The Keras History object returned by `model.fit`.

    TODO
    ----
    Call `model.fit(...)` with the appropriate arguments.
    """

    # TODO:
    # history = model.fit(...)

    raise NotImplementedError("TODO: implement train_model")


def evaluate_model(model, split):
    """
    Evaluate the trained model ONCE on the held-out test set.

    The returned dictionary must contain:

        "test_loss"
        "test_accuracy"

    Use:
        split.X_test
        split.y_test

    TODO
    ----
    Evaluate the model and return the metrics as normal Python floats.
    """

    # TODO:
    # loss, accuracy = model.evaluate(...)

    raise NotImplementedError("TODO: implement evaluate_model")


def build_improved_model(input_dim):
    """
    Build a modestly improved network.

    Required architecture
    ---------------------
    Input
      -> Dense(32, relu)
      -> Dropout(0.15)
      -> Dense(16, relu)
      -> Dense(1, sigmoid)

    Compile it with Adam and a learning rate of 1e-3.

    The intention is NOT to build a huge model. We want a small, controlled
    change that can be compared with the baseline.

    TODO
    ----
    Build, compile and return the model.
    """

    # TODO: build the architecture.

    # TODO: compile it, preferably by reusing compile_binary_model(...).

    raise NotImplementedError("TODO: implement build_improved_model")


def build_regularized_model(input_dim, l2_strength=1e-4):
    """
    Build an MLP using L2 kernel regularization.

    Required architecture
    ---------------------
    Dense(32, relu, kernel_regularizer=L2)
      -> Dense(16, relu, kernel_regularizer=L2)
      -> Dense(1, sigmoid)

    Important
    ---------
    This function only builds the model. The notebook will compile it later.

    TODO
    ----
    Create the L2 regularizer and apply it to both hidden Dense layers.
    """

    # TODO:
    # reg = tf.keras.regularizers.l2(...)

    # TODO: build and return the Sequential model.

    raise NotImplementedError("TODO: implement build_regularized_model")


def build_dropout_model(input_dim, rate=0.25):
    """
    Build an MLP that uses dropout.

    Required architecture
    ---------------------
    Dense(32, relu)
      -> Dropout(rate)
      -> Dense(16, relu)
      -> Dense(1, sigmoid)

    Important
    ---------
    This function only builds the model. The notebook will compile it later.

    TODO
    ----
    Create and return the model.
    """

    # TODO: build and return the Sequential model.

    raise NotImplementedError("TODO: implement build_dropout_model")


def optimizer_from_name(name, learning_rate=1e-3):
    """
    Construct one of the optimizers studied in Part 2.

    Supported names
    ---------------
    "sgd"
        Standard stochastic gradient descent.

    "momentum"
        SGD with momentum=0.9.

    "nesterov"
        SGD with momentum=0.9 and nesterov=True.

    "adagrad"
        AdaGrad.

    "rmsprop"
        RMSprop.

    "adam"
        Adam.

    All optimizers must use the provided learning rate.

    Hints
    -----
    You may normalize the input string using:

        name.lower()
        .replace(" ", "")
        .replace("-", "")

    Raise ValueError for an unknown optimizer.

    TODO
    ----
    Implement the optimizer selection logic.
    """

    # TODO: return the matching tf.keras.optimizers optimizer.

    raise NotImplementedError("TODO: implement optimizer_from_name")


def run_optimizer_experiment(
    split,
    optimizer_name,
    *,
    epochs=150,
    batch_size=32,
    learning_rate=1e-3,
    patience=15,
    seed=RANDOM_STATE,
):
    """
    Train ONE optimizer under controlled conditions.

    Fair-comparison requirements
    ----------------------------
    Keep these fixed:
    - data split;
    - architecture;
    - seed;
    - batch size;
    - stopping rule.

    Only the optimizer should change.


    The notebook expects:

        "optimizer"
        "epochs_run"
        "seconds"
        "test_loss"
        "test_accuracy"
        "history"
        "model"

    TODO
    ----
    Implement one complete experiment.
    """

    # TODO: make the run reproducible.

    # TODO: build model.

    # TODO: construct optimizer.

    # TODO: compile model.

    # TODO: record start time.

    # TODO: train.

    # TODO: compute elapsed time.

    # TODO: evaluate.

    # TODO: return the result dictionary.

    raise NotImplementedError("TODO: implement run_optimizer_experiment")


def compare_optimizers(
    split,
    names=(
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
    ),
    **kwargs,
):
    """
    Run `run_optimizer_experiment` for every requested optimizer.

    Parameters
    ----------
    split:
        The common DataSplit used for every experiment.
    names:
        Iterable containing optimizer names.
    **kwargs:
        Additional keyword arguments forwarded to run_optimizer_experiment.

    Returns
    -------
    list
        One result dictionary per optimizer.

    TODO
    ----
    A compact loop or list comprehension is sufficient.
    """

    # TODO: call run_optimizer_experiment once per optimizer name.

    raise NotImplementedError("TODO: implement compare_optimizers")


def available_devices():
    return {
        "CPU": [d.name for d in tf.config.list_physical_devices("CPU")],
        "GPU": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
