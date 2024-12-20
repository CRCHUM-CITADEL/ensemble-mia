# Standard library
from typing import Tuple, Union

# 3rd party packages
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_real_data(
    data_path: Union[Path, str], seed: int, var_to_predict: str, save_folder: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the real data

    :param data_path: the path of the data in csv
    :param seed: for reproduction
    :param var_to_predict: variable to predict
    :param save_folder: the name of the sub-folder to save the result

    :return: the partitioned data
    """

    # Load the real data
    df_real = pd.read_csv(data_path)

    # Split the real data into train and control
    df_real_train, df_real_control = train_test_split(
        df_real,
        test_size=0.6,
        random_state=seed,
        stratify=df_real[var_to_predict],
    )

    # Further split the control:
    #   - One portion used to train detector
    #   - One portion used as part of the validation set for ensemble model
    #   - One portion used to construct the test set to evaluate the final model
    df_real_control_detector, df_real_control_rest = train_test_split(
        df_real_control,
        test_size=1 / 3,
        random_state=seed,
        stratify=df_real_control[var_to_predict],
    )

    df_real_control_val, df_real_control_test = train_test_split(
        df_real_control_rest,
        test_size=0.5,
        random_state=seed,
        stratify=df_real_control_rest[var_to_predict],
    )

    df_real_train.to_csv(
        save_folder / "real_train.csv",
        index=False,
    )

    df_real_control_detector.to_csv(
        save_folder / "real_control_detector.csv",
        index=False,
    )

    df_real_control_val.to_csv(
        save_folder / "real_control_val.csv",
        index=False,
    )

    df_real_control_test.to_csv(
        save_folder / "real_control_test.csv",
        index=False,
    )

    return (
        df_real_train,
        df_real_control_detector,
        df_real_control_val,
        df_real_control_test,
    )
