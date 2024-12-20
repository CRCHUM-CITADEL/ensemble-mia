# Standard library
from typing import Tuple

# 3rd party packages
import numpy as np
import pandas as pd


def prepare_data(
    df_real_train: pd.DataFrame,
    df_real_control_val: pd.DataFrame,
    df_real_control_test: pd.DataFrame,
    cat_cols: list,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Prepare validation and test data for ensemble model

    :param df_real_train: the real train data
    :param df_real_control_val: the real control data for validation
    :param df_real_control_test: the real control data for final evaluation
    :param cat_cols: the name(s) of the categorical variable(s)
    :param seed: for reproduction

    :return: the features and label for validation and test sets
    """

    # Construct validation set for ensemble model
    df_real_train_sample = df_real_train.sample(
        n=len(df_real_control_val) + len(df_real_control_test),
        replace=False,
        ignore_index=True,
        random_state=seed,
    )

    df_real_train_val = df_real_train_sample.iloc[: len(df_real_control_val), :]

    df_val = pd.concat(
        [df_real_train_val, df_real_control_val],
        axis=0,
        ignore_index=True,
    )

    df_val[cat_cols] = df_val[cat_cols].astype("object")

    #  Label 1 for real records used to generate 1st generation synthetic data and 0 for control
    y_val = np.array([1] * len(df_real_train_val) + [0] * len(df_real_control_val))

    # Test set: can be used to evaluate all the models
    df_real_train_test = df_real_train_sample.iloc[len(df_real_control_val) :, :]

    df_test = pd.concat(
        [df_real_train_test, df_real_control_test],
        axis=0,
        ignore_index=True,
    )

    df_test[cat_cols] = df_test[cat_cols].astype("object")

    #  Label 1 for real records used to generate 1st generation synthetic data and 0 for control.
    y_test = np.array([1] * len(df_real_train_test) + [0] * len(df_real_control_test))

    return df_val, y_val, df_test, y_test
