# Standard library
import sys

sys.path.append("..")
from typing import Union

# 3rd party packages
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Local
from clover.metrics.privacy.membership import Logan, TableGan
from src.utils.learning import hyperparam_tuning
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
    meta_classifier: Union[LogisticRegression, Pipeline] = None,
    meta_classifier_type: str = None,
    df_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
) -> dict:
    """Fit MIA with ensemble approach (stacking) and output the prediction on the test set and the meta classifier

    * The meta classifier is trained with a hold-out (validation) set
    * Logistic regression or XGBoost can be used as the level-1 model
    * Prediction from each individual model is used as input of the meta classifier

    :param df_train_logan: the training data for LOGAN
    :param y_train_logan: the training label for LOGAN
    :param df_train_tablegan_discriminator: the data to train the TableGAN discriminator
    :param y_train_tablegan_discriminator: the training label for the TableGAN discriminator
    :param df_train_tablegan_classifier: the data to train the TableGAN classifier
    :param y_train_tablegan_classifier: the training label for the TableGAN classifier
    :param df_ref: the reference population data
    :param df_synth: the synthetic data
    :param df_test: the features of the test set
    :param cont_cols: the name(s) of the continuous variable(s)
    :param cat_cols: the name(s) of the categorical variable(s)
    :param iteration: the number of time to train the model
    :param meta_classifier: the trained meta classifier. If the trained meta classifier is not provided,
        meta_classifier_type, df_val and y_val should be provided to train a meta classifier.
    :param meta_classifier_type: type of the meta classifier, lr or xgb
    :param df_val: the features of the validation set
    :param y_val: the labels of the validation set

    :return: the predicted probabilities and the trained meta classifier
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

    # Only need to train domias once
    if meta_classifier is None:
        y_val_pred_proba_domias = domias.fit_pred(
            df_ref=df_ref.astype(float),
            df_synth=df_synth.astype(float),
            df_test=df_val.astype(float),
        )

        # Convert prediction to pandas DataFrame
        y_val_pred_proba_domias = pd.DataFrame(
            y_val_pred_proba_domias, columns=["pred_proba_domias"]
        )
    else:
        y_val_pred_proba_domias = None

    y_test_pred_proba_domias = domias.fit_pred(
        df_ref=df_ref.astype(float),
        df_synth=df_synth.astype(float),
        df_test=df_test.astype(float),
    )
    y_test_pred_proba_domias = pd.DataFrame(
        y_test_pred_proba_domias, columns=["pred_proba_domias"]
    )

    pred_proba_stacking = []
    meta_classifier_stacking = []

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

        if meta_classifier is not None:
            meta_classifier = meta_classifier
        else:
            ##################################################
            # Train meta classifier
            ##################################################

            # Predictions of the individual model on validation set
            y_val_pred_proba_logan = pipe_logan.predict_proba(df_val)[:, 1]
            y_val_pred_proba_tablegan = tablegan.pred_proba(
                df=df_val,
                trained_discriminator=pipe_tablegan_discriminator,
                trained_classifier=pipe_tablegan_classifier,
            )

            # Convert prediction to pandas DataFrame
            y_val_pred_proba_logan = pd.DataFrame(
                y_val_pred_proba_logan, columns=["pred_proba_logan"]
            )
            y_val_pred_proba_tablegan = pd.DataFrame(
                y_val_pred_proba_tablegan, columns=["pred_proba_tablegan"]
            )

            # Prepare training data for meta classifier
            df_val_meta = pd.concat(
                [
                    y_val_pred_proba_logan,
                    y_val_pred_proba_tablegan,
                    y_val_pred_proba_domias,
                ],
                axis=1,
            )

            if meta_classifier_type == "lr":  # Logistic Regression Model training
                meta_classifier = LogisticRegression(max_iter=1000)
                meta_classifier.fit(df_val_meta, y_val)
            else:  # XGBoost
                meta_classifier = hyperparam_tuning(
                    x=df_val_meta,
                    y=y_val,
                    continuous_cols=list(df_val_meta.columns),
                    categorical_cols=[],
                    bounds={},
                    num_optuna_trials=20,
                    num_kfolds=5,
                    use_gpu=True,
                )

        ##################################################
        # Evaluate meta classifier on test set
        ##################################################

        # Prediction of individual model on test set
        y_test_pred_proba_logan = pipe_logan.predict_proba(df_test)[:, 1]
        y_test_pred_proba_tablegan = tablegan.pred_proba(
            df=df_test,
            trained_discriminator=pipe_tablegan_discriminator,
            trained_classifier=pipe_tablegan_classifier,
        )

        y_test_pred_proba_logan = pd.DataFrame(
            y_test_pred_proba_logan, columns=["pred_proba_logan"]
        )
        y_test_pred_proba_tablegan = pd.DataFrame(
            y_test_pred_proba_tablegan, columns=["pred_proba_tablegan"]
        )

        # Prepare test data for meta classifier
        df_test_meta = pd.concat(
            [
                y_test_pred_proba_logan,
                y_test_pred_proba_tablegan,
                y_test_pred_proba_domias,
            ],
            axis=1,
        )

        y_pred_proba_final = meta_classifier.predict_proba(df_test_meta)[:, 1]

        pred_proba_stacking.append(y_pred_proba_final)
        meta_classifier_stacking.append(meta_classifier)

    return {
        "pred_proba": pred_proba_stacking,
        "meta_classifier": meta_classifier_stacking,
    }
