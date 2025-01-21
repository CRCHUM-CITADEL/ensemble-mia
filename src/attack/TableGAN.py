# Standard library
import sys

sys.path.insert(0, "..")
from pathlib import Path
from typing import Tuple

# 3rd party packages
import numpy as np
import pandas as pd

# Local
from clover.metrics.privacy.membership import AttackModel, TableGan
from src.utils import draw


def prepare_data(
    df_synth_train: pd.DataFrame,
    df_synth_test: pd.DataFrame,
    df_synth_2nd: pd.DataFrame,
    size: int,
    cat_cols: list,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Prepare training data for TableGAN

    TableGAN is trained in 2 steps:
    - Train a discriminator
    - Train a classifier

    :param df_synth_train: the 1st generation synthetic train data
    :param df_synth_test: the 1st generation synthetic test data
    :param df_synth_2nd: the 2nd generation synthetic data
    :param size: the number of the samples in the 1st generation synthetic train data to be used to train the classifier
    :param cat_cols: the name(s) of the categorical variable(s)
    :param seed: for reproduction

    :return: the features and label to train the discriminator and classifier
    """

    # Split the 1st generation synthetic train set into 2 sets:
    # 1 used to train discriminator and another used to train final classifier
    df_synth_train_tablegan_classifier = df_synth_train.sample(
        n=size, replace=False, ignore_index=False, random_state=seed
    )

    df_synth_train_tablegan_discriminator = df_synth_train[
        ~df_synth_train.index.isin(df_synth_train_tablegan_classifier.index)
    ].reset_index(drop=True)

    # Construct train set to train the discriminator: 1st gen + 2nd gen synthetic sets
    df_train_tablegan_discriminator = pd.concat(
        [df_synth_train_tablegan_discriminator, df_synth_2nd],
        axis=0,
        ignore_index=True,
    )

    df_train_tablegan_discriminator[cat_cols] = df_train_tablegan_discriminator[
        cat_cols
    ].astype("object")

    # Label 1 for 1st generation synthetic data and 0 for 2nd generation synthetic data.
    y_train_tablegan_discriminator = np.array(
        [1] * len(df_synth_train_tablegan_discriminator) + [0] * len(df_synth_2nd)
    )

    # Construct the train set used to train the final classifier, which contains
    # 1st generation of synthetic data which is used to generate the 2nd generation synthetic data and control set
    df_train_tablegan_classifier = pd.concat(
        [
            df_synth_train_tablegan_classifier.reset_index(drop=True),
            df_synth_test,
        ],
        axis=0,
        ignore_index=True,
    )

    df_train_tablegan_classifier[cat_cols] = df_train_tablegan_classifier[
        cat_cols
    ].astype("object")

    # Label 1 for 1st gen synthetic data used to generate 2nd generation synthetic data and 0 for control.
    y_train_tablegan_classifier = np.array(
        [1] * len(df_synth_train_tablegan_classifier) + [0] * len(df_synth_test)
    )

    return (
        df_train_tablegan_discriminator,
        y_train_tablegan_discriminator,
        df_train_tablegan_classifier,
        y_train_tablegan_classifier,
    )


def model(
    df_real_train: pd.DataFrame,
    df_synth_train: pd.DataFrame,
    df_synth_test: pd.DataFrame,
    df_train_tablegan_discriminator: pd.DataFrame,
    y_train_tablegan_discriminator: np.ndarray,
    df_train_tablegan_classifier: pd.DataFrame,
    y_train_tablegan_classifier: np.ndarray,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    cont_cols: list,
    cat_cols: list,
    iteration: int,
    save_path: Path,
    seed: int,
) -> Tuple[list, list, list]:
    """Membership inference attack with TableGAN

    :param df_real_train: the real train data
    :param df_synth_train: the 1st generation synthetic train data
    :param df_synth_test: the 1st generation synthetic test data
    :param df_train_tablegan_discriminator: the data to train the discriminator
    :param y_train_tablegan_discriminator: the training label for the discriminator
    :param df_train_tablegan_classifier: the data to train the classifier
    :param y_train_tablegan_classifier: the training label for the classifier
    :param df_test: the test data
    :param y_test: the test label
    :param cont_cols: the name(s) of the continuous variable(s)
    :param cat_cols: the name(s) of the categorical variable(s)
    :param iteration: the number of time to train the model
    :param save_path: the path to save the plot
    :param seed: for reproduction

    :return: the predicted probability, top 1% precision and top 50% precision of the predictions
    """

    tablegan = TableGan(
        num_kfolds=5,
        num_optuna_trials=20,
        use_gpu=True,
    )

    pred_proba = []
    precision_top1_tablegan = []
    precision_top50_tablegan = []

    for i in range(iteration):
        pipe_tablegan_discriminator, pipe_tablegan_classifier = tablegan.fit(
            df_train_discriminator=df_train_tablegan_discriminator,
            y_train_discriminator=y_train_tablegan_discriminator,
            df_train_classifier=df_train_tablegan_classifier,
            y_train_classifier=y_train_tablegan_classifier,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
        )

        y_pred_proba = tablegan.pred_proba(
            df=df_test,
            trained_discriminator=pipe_tablegan_discriminator,
            trained_classifier=pipe_tablegan_classifier,
        )

        precision_top_1 = AttackModel.precision_top_n(
            n=1, y_true=y_test, y_pred_proba=y_pred_proba
        )
        precision_top_50 = AttackModel.precision_top_n(
            n=50, y_true=y_test, y_pred_proba=y_pred_proba
        )

        pred_proba.append(y_pred_proba)
        precision_top1_tablegan.append(precision_top_1)
        precision_top50_tablegan.append(precision_top_50)

        draw.prediction_vis(
            df_real_train=df_real_train,
            df_synth_train=df_synth_train,
            df_synth_test=df_synth_test,
            df_test=df_test,
            y_test=y_test,
            y_pred_proba=y_pred_proba,
            cont_col=cont_cols,
            n=1,
            save_path=save_path / f"tablegan_output_iter{i}.jpg",
            seed=seed,
        )

    return pred_proba, precision_top1_tablegan, precision_top50_tablegan
