# Standard library
import sys

sys.path.insert(0, "..")
from pathlib import Path
from typing import Tuple

# 3rd party packages
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Local
from clover.metrics.privacy.membership import AttackModel, Logan, TableGan, Detector
from src.utils import draw


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
    df_val: pd.DataFrame,
    y_val: np.ndarray,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    cont_cols: list,
    cat_cols: list,
    iteration: int,
    save_path: Path,
    seed: int,
) -> Tuple[list, list]:
    """Membership inference attack with ensemble approach - blending

    The original features are also used as input to train the meta classifier

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
    :param df_val: the features of the validation set
    :param y_val: the labels of the validation set
    :param df_test: the features of the test set
    :param y_test: the labels of the test set
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

    precision_top1_blending = []
    precision_top50_blending = []

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

        # Make predictions on validation set (to be used to train logistic regression)
        y_val_pred_proba_logan = pipe_logan.predict_proba(df_val)[:, 1]
        y_val_pred_proba_tablegan = tablegan.pred_proba(
            df=df_val,
            trained_discriminator=pipe_tablegan_discriminator,
            trained_classifier=pipe_tablegan_classifier,
        )
        y_val_pred_proba_detector = pipe_detector.predict_proba(df_val)[:, 1]

        y_val_pred_proba_logan = pd.DataFrame(
            y_val_pred_proba_logan, columns=["pred_proba_logan"]
        )
        y_val_pred_proba_tablegan = pd.DataFrame(
            y_val_pred_proba_tablegan, columns=["pred_proba_tablegan"]
        )
        y_val_pred_proba_detector = pd.DataFrame(
            y_val_pred_proba_detector, columns=["pred_proba_detector"]
        )

        # Test set (to be used to evaluation the logistic regression)
        y_test_pred_proba_logan = pipe_logan.predict_proba(df_test)[:, 1]
        y_test_pred_proba_tablegan = tablegan.pred_proba(
            df=df_test,
            trained_discriminator=pipe_tablegan_discriminator,
            trained_classifier=pipe_tablegan_classifier,
        )
        y_test_pred_proba_detector = pipe_detector.predict_proba(df_test)[:, 1]

        y_test_pred_proba_logan = pd.DataFrame(
            y_test_pred_proba_logan, columns=["pred_proba_logan"]
        )
        y_test_pred_proba_tablegan = pd.DataFrame(
            y_test_pred_proba_tablegan, columns=["pred_proba_tablegan"]
        )
        y_test_pred_proba_detector = pd.DataFrame(
            y_test_pred_proba_detector, columns=["pred_proba_detector"]
        )

        # Training set for the meta classifier
        df_val_lr = pd.concat(
            [
                df_val,
                y_val_pred_proba_logan,
                y_val_pred_proba_tablegan,
                y_val_pred_proba_detector,
            ],
            axis=1,
        )

        # Test set for the meta classifier
        df_test_lr = pd.concat(
            [
                df_test,
                y_test_pred_proba_logan,
                y_test_pred_proba_tablegan,
                y_test_pred_proba_detector,
            ],
            axis=1,
        )

        # Logistic Regression Model Pipeline
        preprocessing = ColumnTransformer(
            [
                ("continuous", StandardScaler(), cont_cols),
                (
                    "categorical",
                    OneHotEncoder(
                        categories=[df_real_train[cat].unique() for cat in cat_cols],
                        handle_unknown="ignore",
                    ),
                    cat_cols,
                ),
            ],
            verbose_feature_names_out=False,
            remainder="passthrough",  # Not transform the predictions of the individual attack model
        )

        pipe_lr = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                (
                    "lr",
                    LogisticRegression(max_iter=1000),
                ),
            ]
        )

        pipe_lr.fit(df_val_lr, y_val)

        y_pred_proba_final = pipe_lr.predict_proba(df_test_lr)[:, 1]

        precision_top_1 = AttackModel.precision_top_n(
            n=1, y_true=y_test, y_pred_proba=y_pred_proba_final
        )
        precision_top_50 = AttackModel.precision_top_n(
            n=50, y_true=y_test, y_pred_proba=y_pred_proba_final
        )

        precision_top1_blending.append(precision_top_1)
        precision_top50_blending.append(precision_top_50)

        draw.prediction_vis(
            df_real_train=df_real_train,
            df_synth_train=df_synth_train,
            df_synth_test=df_synth_test,
            df_test=df_test,
            y_test=y_test,
            y_pred_proba=y_pred_proba_final,
            cont_col=cont_cols,
            n=1,
            save_path=save_path / f"blending_output_iter{i}.jpg",
            seed=seed,
        )

    return precision_top1_blending, precision_top50_blending
