# Standard library
import sys

sys.path.append("..")

# 3rd party packages
import numpy as np
import pandas as pd

# Local
from clover.metrics.privacy.membership import Logan, TableGan
from src.attack import domias


def fit_pred(
    df_train_logan: pd.DataFrame,
    y_train_logan: np.ndarray,
    df_train_tablegan_discriminator: pd.DataFrame,
    y_train_tablegan_discriminator: np.ndarray,
    df_train_tablegan_classifier: pd.DataFrame,
    y_train_tablegan_classifier: np.ndarray,
    df_ref: pd.DataFrame,
    df_synth: pd.DataFrame,
    df_test: pd.DataFrame,
    cont_cols: list,
    cat_cols: list,
    iteration: int,
) -> list:
    """Fit MIA with ensemble approach (soft voting) and output the prediction on the test set

    :param df_train_logan: the training data for LOGAN
    :param y_train_logan: the training label for LOGAN
    :param df_train_tablegan_discriminator: the data to train the TableGAN discriminator
    :param y_train_tablegan_discriminator: the training label for the TableGAN discriminator
    :param df_train_tablegan_classifier: the data to train the TableGAN classifier
    :param y_train_tablegan_classifier: the training label for the TableGAN classifier
    :param df_ref: the reference population data
    :param df_synth: the synthetic data
    :param df_test: the test data
    :param cont_cols: the name(s) of the continuous variable(s)
    :param cat_cols: the name(s) of the categorical variable(s)
    :param iteration: the number of time to train the model

    :return: the predicted probabilities
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

    y_pred_proba_domias = domias.fit_pred(
        df_ref=df_ref.astype(float),
        df_synth=df_synth.astype(float),
        df_test=df_test.astype(float),
    )

    pred_proba_voting = []

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

        y_pred_proba_logan = pipe_logan.predict_proba(df_test)[:, 1]
        y_pred_proba_tablegan = tablegan.pred_proba(
            df=df_test,
            trained_discriminator=pipe_tablegan_discriminator,
            trained_classifier=pipe_tablegan_classifier,
        )

        y_pred_proba_final = np.mean(
            [y_pred_proba_logan, y_pred_proba_tablegan, y_pred_proba_domias],
            axis=0,
        )

        pred_proba_voting.append(y_pred_proba_final)

    return pred_proba_voting
