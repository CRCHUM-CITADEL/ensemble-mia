# Standard library
import sys

sys.path.insert(0, "..")

# 3rd party packages
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Local
from clover.metrics.privacy.membership import Logan, TableGan


def fit_pred(
    df_train_logan: pd.DataFrame,
    y_train_logan: np.ndarray,
    df_train_tablegan_discriminator: pd.DataFrame,
    y_train_tablegan_discriminator: np.ndarray,
    df_train_tablegan_classifier: pd.DataFrame,
    y_train_tablegan_classifier: np.ndarray,
    df_test: pd.DataFrame,
    cont_cols: list,
    cat_cols: list,
    iteration: int,
    meta_classifier: Pipeline = None,
    df_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    bounds: dict = None,
) -> dict:
    """Fit MIA with ensemble approach (blending) and output the prediction on the test set and the meta classifier

    The original features are also used as input to train the meta classifier

    :param df_train_logan: the training data for LOGAN
    :param y_train_logan: the training label for LOGAN
    :param df_train_tablegan_discriminator: the data to train the TableGAN discriminator
    :param y_train_tablegan_discriminator: the training label for the TableGAN discriminator
    :param df_train_tablegan_classifier: the data to train the TableGAN classifier
    :param y_train_tablegan_classifier: the training label for the TableGAN classifier
    :param df_test: the features of the test set
    :param cont_cols: the name(s) of the continuous variable(s)
    :param cat_cols: the name(s) of the categorical variable(s)
    :param iteration: the number of time to train the model
    :param meta_classifier: the trained meta classifier. If the trained meta classifier is not provided,
        df_val, y_val and bounds should be provided to train a meta classifier.
    :param df_val: the features of the validation set
    :param y_val: the labels of the validation set
    :param bounds: the unique values of each categorical variables, in the format of
        {"cat_var_name": {"categories": [XX, XX, ...]}, ...}

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

    pred_proba_blending = []
    meta_classifier_blending = []

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
            pipe_lr = meta_classifier
        else:
            ##################################################
            # Train meta classifier
            ##################################################

            # Make predictions on validation set (to be used to train logistic regression)
            y_val_pred_proba_logan = pipe_logan.predict_proba(df_val)[:, 1]
            y_val_pred_proba_tablegan = tablegan.pred_proba(
                df=df_val,
                trained_discriminator=pipe_tablegan_discriminator,
                trained_classifier=pipe_tablegan_classifier,
            )

            y_val_pred_proba_logan = pd.DataFrame(
                y_val_pred_proba_logan, columns=["pred_proba_logan"]
            )
            y_val_pred_proba_tablegan = pd.DataFrame(
                y_val_pred_proba_tablegan, columns=["pred_proba_tablegan"]
            )

            # Training set for the meta classifier
            df_val_lr = pd.concat(
                [df_val, y_val_pred_proba_logan, y_val_pred_proba_tablegan],
                axis=1,
            )

            # Logistic Regression Model Pipeline
            preprocessing = ColumnTransformer(
                [
                    ("continuous", StandardScaler(), cont_cols),
                    (
                        "categorical",
                        OneHotEncoder(
                            categories=[bounds[cat]["categories"] for cat in cat_cols],
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

        ##################################################
        # Evaluate meta classifier on test set
        ##################################################

        # Test set (to be used to evaluation the logistic regression)
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

        # Test set for the meta classifier
        df_test_lr = pd.concat(
            [df_test, y_test_pred_proba_logan, y_test_pred_proba_tablegan],
            axis=1,
        )

        y_pred_proba_final = pipe_lr.predict_proba(df_test_lr)[:, 1]

        pred_proba_blending.append(y_pred_proba_final)
        meta_classifier_blending.append(pipe_lr)

    return {
        "pred_proba": pred_proba_blending,
        "meta_classifier": meta_classifier_blending,
    }
