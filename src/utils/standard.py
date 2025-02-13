# Standard library
import os
import zipfile
from pathlib import Path
from typing import Union

# 3rd party packages
import pandas as pd


def create_directory(path: Union[Path, str]) -> None:
    """
    Create directory if it does not exist

    :param path: the directory to be created
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def trans_type(df: pd.DataFrame, col_type: dict, decimal: int) -> pd.DataFrame:
    """
    Transform the column type of the dataframe

    :param df: the dataframe to be transformed
    :param col_type: a dictionary to specify the desired type for each column,
        i.e., {"float": ["col1", "col2"], "int": ["col3", "col4"]}
    :param decimal: decimal places to be preserved for all the float type
    :return: the transformed dataframe
    """
    df_trans = df.copy()

    for col in col_type["int"]:
        df_trans[col] = df_trans[col].astype(int)

    for col in col_type["float"]:
        df_trans[col] = df_trans[col].astype(float).round(decimal)

    return df_trans


def zip_files(model_name: str, tasks: list, file_path: Union[str, Path]) -> None:
    """
    Create a zip file to store all the predictions for all the dev and final challenge points

    :param model_name: the name of the attack model
    :param tasks: the performed task, i.e., ["tabddpm_black_box", "tabsyn_black_box"] or one of the two
    :param file_path: the path where the predictions are stored (in csv files) for different attack models
    :return: None
    """

    output_dir = Path(file_path)

    with zipfile.ZipFile(
        output_dir / model_name / f"black_box_single_table_submission.zip", "w"
    ) as zipf:
        for phase in ["dev", "final"]:
            for task in tasks:
                root = output_dir / model_name / task / phase
                model_folders = [
                    item for item in os.listdir(root) if os.path.isdir(root / item)
                ]
                for model_folder in sorted(
                    model_folders, key=lambda d: int(d.split("_")[1])
                ):
                    path = root / model_folder
                    if not os.path.isdir(path):
                        continue

                    file = path / "prediction.csv"
                    if os.path.exists(file):
                        # Use `arcname` to remove the base directory and phase directory from the zip path
                        arcname = os.path.relpath(
                            file, os.path.dirname(output_dir / model_name / task)
                        )
                        zipf.write(file, arcname=str(arcname))
                    else:
                        raise FileNotFoundError(
                            f"`prediction.csv` not found in {path}."
                        )
