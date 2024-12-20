# Standard library
# import sys
#
# sys.path.insert(0, "../../../../..")
from pathlib import Path
from typing import Tuple


# 3rd party packages
import numpy as np
import pandas as pd

# Local
from metrics.privacy.membership import AttackModel, Logan
from mia_ensemble.src.utils import draw


def prepare_data(
    df_synth_train: pd.DataFrame,
    df_synth_2nd: pd.DataFrame,
    size: int,
    cat_cols: list,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare training data for LOGAN

    :param df_synth_train: the 1st generation synthetic train data
    :param df_synth_2nd: the 2nd generation synthetic data
    :param size: the number of the samples in the 1st generation synthetic train data to be used
    :param cat_cols: the name(s) of the categorical variable(s)
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

    df_train_logan[cat_cols] = df_train_logan[cat_cols].astype("object")

    # Label 1 for 1st generation synthetic data used to generate 2nd generation synthetic data and
    # 0 for 2nd generation synthetic data
    y_train_logan = np.array([1] * len(df_synth_train_sample) + [0] * len(df_synth_2nd))

    return df_train_logan, y_train_logan


def model(
    df_real_train: pd.DataFrame,
    df_synth_train: pd.DataFrame,
    df_synth_test: pd.DataFrame,
    df_train_logan: pd.DataFrame,
    y_train_logan: np.ndarray,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    cont_cols: list,
    cat_cols: list,
    iteration: int,
    save_path: Path,
    seed: int,
) -> Tuple[list, list]:
    """Membership inference attack with LOGAN

    :param df_real_train: the real train data
    :param df_synth_train: the 1st generation synthetic train data
    :param df_synth_test: the 1st generation synthetic test data
    :param df_train_logan: the training data for LOGAN
    :param y_train_logan: the training label
    :param df_test: the test data
    :param y_test: the test label
    :param cont_cols: the name(s) of the continuous variable(s)
    :param cat_cols: the name(s) of the categorical variable(s)
    :param iteration: the number of time to train the model
    :param save_path: the path to save the plot
    :param seed: for reproduction

    :return: top 1% precision and top 50% precision of the predictions
    """

    logan = Logan(
        num_kfolds=5,
        num_optuna_trials=20,
        use_gpu=True,
    )

    precision_top1_logan = []
    precision_top50_logan = []

    for i in range(iteration):
        pipe_logan = logan.fit(
            df_train=df_train_logan,
            y_train=y_train_logan,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
        )

        y_pred_proba = pipe_logan.predict_proba(df_test)[:, 1]

        precision_top_1 = AttackModel.precision_top_n(
            n=1, y_true=y_test, y_pred_proba=y_pred_proba
        )
        precision_top_50 = AttackModel.precision_top_n(
            n=50, y_true=y_test, y_pred_proba=y_pred_proba
        )

        precision_top1_logan.append(precision_top_1)
        precision_top50_logan.append(precision_top_50)

        draw.prediction_vis(
            df_real_train=df_real_train,
            df_synth_train=df_synth_train,
            df_synth_test=df_synth_test,
            df_test=df_test,
            y_test=y_test,
            y_pred_proba=y_pred_proba,
            cont_col=cont_cols,
            n=1,
            save_path=save_path / f"logan_output_iter{i}.jpg",
            seed=seed,
        )

    return precision_top1_logan, precision_top50_logan
