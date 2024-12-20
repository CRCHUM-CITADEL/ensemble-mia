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
from metrics.privacy.membership import AttackModel, Logan, TableGan, Detector
from mia_ensemble.src.utils import draw


def model(
    df_real_train: pd.DataFrame,
    df_synth_train: pd.DataFrame,
    df_synth_test: pd.DataFrame,
    df_train_logan: pd.DataFrame,
    y_train_logan: np.ndarray,
    df_train_tablegan_discriminator: pd.DataFrame,
    y_train_tablegan_discriminator: np.ndarray,
    df_train_tablegan_classifier: pd.DataFrame,
    y_train_tablegan_classifier: np.ndarray,
    df_train_detector: pd.DataFrame,
    y_train_detector: np.ndarray,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    cont_cols: list,
    cat_cols: list,
    iteration: int,
    save_path: Path,
    seed: int,
) -> Tuple[list, list]:
    """Membership inference attack with ensemble approach - soft voting

    :param df_real_train: the real train data
    :param df_synth_train: the 1st generation synthetic train data
    :param df_synth_test: the 1st generation synthetic test data
    :param df_train_logan: the training data for LOGAN
    :param y_train_logan: the training label for LOGAN
    :param df_train_tablegan_discriminator: the data to train the TableGAN discriminator
    :param y_train_tablegan_discriminator: the training label for the TableGAN discriminator
    :param df_train_tablegan_classifier: the data to train the TableGAN classifier
    :param y_train_tablegan_classifier: the training label for the TableGAN classifier
    :param df_train_detector: the training data for Detector
    :param y_train_detector: the training label for Detector
    :param df_test: the test data
    :param y_test: the test label
    :param cont_cols: the name(s) of the continuous variable(s)
    :param cat_cols: the name(s) of the categorical variable(s)
    :param iteration: the number of time to train the model
    :param save_path: the path to save the plot
    :param seed: for reproduction

    :return: top 1% precision and top 50% precision of the predictions
    """

    # Initiate the individual attack model
    logan = Logan(
        num_kfolds=5,
        num_optuna_trials=20,
        use_gpu=True,
    )

    tablegan = TableGan(
        num_kfolds=5,
        num_optuna_trials=20,
        use_gpu=True,
    )

    detector = Detector(
        num_kfolds=5,
        num_optuna_trials=20,
        use_gpu=True,
    )

    precision_top1_voting = []
    precision_top50_voting = []

    for i in range(iteration):
        pipe_logan = logan.fit(
            df_train=df_train_logan,
            y_train=y_train_logan,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
        )

        pipe_tablegan_discriminator, pipe_tablegan_classifier = tablegan.fit(
            df_train_discriminator=df_train_tablegan_discriminator,
            y_train_discriminator=y_train_tablegan_discriminator,
            df_train_classifier=df_train_tablegan_classifier,
            y_train_classifier=y_train_tablegan_classifier,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
        )

        pipe_detector = detector.fit(
            df_train=df_train_detector,
            y_train=y_train_detector,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
        )

        y_pred_proba_logan = pipe_logan.predict_proba(df_test)[:, 1]
        y_pred_proba_tablegan = tablegan.pred_proba(
            df=df_test,
            trained_discriminator=pipe_tablegan_discriminator,
            trained_classifier=pipe_tablegan_classifier,
        )
        y_pred_proba_detector = pipe_detector.predict_proba(df_test)[:, 1]

        y_pred_proba_final = np.mean(
            [
                y_pred_proba_logan,
                y_pred_proba_tablegan,
                y_pred_proba_detector,
            ],
            axis=0,
        )

        precision_top_1 = AttackModel.precision_top_n(
            n=1, y_true=y_test, y_pred_proba=y_pred_proba_final
        )
        precision_top_50 = AttackModel.precision_top_n(
            n=50, y_true=y_test, y_pred_proba=y_pred_proba_final
        )

        precision_top1_voting.append(precision_top_1)
        precision_top50_voting.append(precision_top_50)

        draw.prediction_vis(
            df_real_train=df_real_train,
            df_synth_train=df_synth_train,
            df_synth_test=df_synth_test,
            df_test=df_test,
            y_test=y_test,
            y_pred_proba=y_pred_proba_final,
            cont_col=cont_cols,
            n=1,
            save_path=save_path / f"soft_voting_output_iter{i}.jpg",
            seed=seed,
        )

    return precision_top1_voting, precision_top50_voting
