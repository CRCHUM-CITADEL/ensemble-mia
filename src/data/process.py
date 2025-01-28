# Standard library
from typing import Tuple, Union

# 3rd party packages
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_real_data(
    df_real: pd.DataFrame,
    save_folder: Union[Path, str],
    var_to_stratify: str = None,
    proportion: dict = None,
    seed: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the real data into train, val and test set and save the data into files

    :param df_real: the real data
    :param save_folder: the name of the path to save the result
    :param var_to_stratify: variable used for stratification
    :param proportion: the proportion of the train and validation set in the format of {"train": xx, "val": xx}
    :param seed: for reproduction

    :return: the partitioned data
    """

    if proportion is None:
        proportion = {"train": 0.5, "val": 0.25}

    # Split the real data into train and control
    df_real_train, df_real_control = train_test_split(
        df_real,
        test_size=1 - proportion["train"],
        random_state=seed,
        stratify=df_real[var_to_stratify],
    )

    # Further split the control into val and test set:
    df_real_val, df_real_test = train_test_split(
        df_real_control,
        test_size=(1 - proportion["train"] - proportion["val"])
        / (1 - proportion["train"]),
        random_state=seed,
        stratify=df_real_control[var_to_stratify],
    )

    save_folder = Path(save_folder)

    df_real_train.to_csv(
        save_folder / "real_train.csv",
        index=False,
    )

    df_real_val.to_csv(
        save_folder / "real_val.csv",
        index=False,
    )

    df_real_test.to_csv(
        save_folder / "real_test.csv",
        index=False,
    )

    return (
        df_real_train,
        df_real_val,
        df_real_test,
    )


def generate_val_test(
    df_real_train: pd.DataFrame,
    df_real_control_val: pd.DataFrame,
    df_real_control_test: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Generate the validation and test set with labels

    :param df_real_train: the real train data
    :param df_real_control_val: the real control data for validation
    :param df_real_control_test: the real control data for final evaluation
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

    #  Label 1 for real records used to generate 1st generation synthetic data and 0 for control
    y_val = np.array([1] * len(df_real_train_val) + [0] * len(df_real_control_val))

    # Test set: can be used to evaluate all the models
    df_real_train_test = df_real_train_sample.iloc[len(df_real_control_val) :, :]

    df_test = pd.concat(
        [df_real_train_test, df_real_control_test],
        axis=0,
        ignore_index=True,
    )

    #  Label 1 for real records used to generate 1st generation synthetic data and 0 for control.
    y_test = np.array([1] * len(df_real_train_test) + [0] * len(df_real_control_test))

    return df_val, y_val, df_test, y_test
