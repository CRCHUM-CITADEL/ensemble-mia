# Standard library
import sys

sys.path.insert(0, "..")
from typing import Tuple


# 3rd party packages
import numpy as np
import pandas as pd

# Local
from clover.metrics.privacy.membership import Logan


def prepare_data(
    df_synth_train: pd.DataFrame,
    df_synth_2nd: pd.DataFrame,
    size: int,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare training data for LOGAN

    :param df_synth_train: the 1st generation synthetic train data
    :param df_synth_2nd: the 2nd generation synthetic data
    :param size: the number of the samples in the 1st generation synthetic train data to be used
    :param seed: for reproduction

    :return: the features and label to train LOGAN
    """

    df_synth_train_sample = df_synth_train.sample(
        n=size, replace=False, ignore_index=True, random_state=seed
    )

    df_train_logan = pd.concat(
        [
            df_synth_train_sample,
            df_synth_2nd,
        ],
        axis=0,
        ignore_index=True,
    )

    # Label 1 for 1st generation synthetic data used to generate 2nd generation synthetic data and
    # 0 for 2nd generation synthetic data
    y_train_logan = np.array([1] * len(df_synth_train_sample) + [0] * len(df_synth_2nd))

    return df_train_logan, y_train_logan


def fit_pred(
    df_train_logan: pd.DataFrame,
    y_train_logan: np.ndarray,
    df_test: pd.DataFrame,
    cont_cols: list,
    cat_cols: list,
    iteration: int,
) -> list:
    """Fit LOGAN and output the prediction on the test set

    :param df_train_logan: the training data for LOGAN
    :param y_train_logan: the training label
    :param df_test: the test data without label
    :param cont_cols: the name(s) of the continuous variable(s)
    :param cat_cols: the name(s) of the categorical variable(s)
    :param iteration: the number of time to train the model


    :return: the predicted probabilities
    """

    logan = Logan(
        num_kfolds=5,
        num_optuna_trials=20,
        use_gpu=True,
    )

    pred_proba_logan = []

    for i in range(iteration):
        pipe_logan = logan.fit(
            df_train=df_train_logan,
            y_train=y_train_logan,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
        )

        y_pred_proba = pipe_logan.predict_proba(df_test)[:, 1]
        pred_proba_logan.append(y_pred_proba)

    return pred_proba_logan
