# Standard library
import sys

sys.path.append("..")

# 3rd party packages
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Local
import src.utils.external.gower.gower_dist as gower
from src.utils.learning import hyperparam_tuning, fit_lr_pipeline
from src.attack import domias

from src import config


def fit_pred(
    df_synth: pd.DataFrame,
    df_ref: pd.DataFrame,
    df_test: pd.DataFrame,
    test_pred_proba_rmia: pd.DataFrame,
    cat_cols: list,
    iteration: int,
    meta_classifier: Pipeline = None,
    meta_classifier_type: str = None,
    df_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    val_pred_proba_rmia: pd.DataFrame = None,
) -> dict:
    """Fit MIA with ensemble approach (blending++) and output the prediction on the test set and the meta classifier

    * The original features are also used as input to train the meta classifier
    * Gower distance to the nearest neighbors, nearest neighbor distance ratio
        and number of neighbors are added as input to the meta classifier

    :param df_synth: the 1st generation synthetic data
    :param df_ref: the reference population data
    :param df_test: the features of the test set
    :param test_pred_proba_rmia: the prediction of RMIA on the test set
    :param cat_cols: the name(s) of the categorical variable(s)
    :param iteration: the number of time to train the model
    :param meta_classifier: the trained meta classifier. If the trained meta classifier is not provided,
        meta_classifier_type, df_val, y_val and val_pred_proba_rmia should be provided to train a meta classifier.
    :param meta_classifier_type: type of the meta classifier, lr or xgb
    :param df_val: the features of the validation set
    :param y_val: the labels of the validation set
    :param val_pred_proba_rmia: the prediction of RMIA on the validation set

    :return: the predicted probabilities and the trained meta classifier
    """

    pred_proba_blending_plus_plus = []
    meta_classifier_blending_plus_plus = []

    ##################################################
    # Compute the Gower distance and # of neighbors
    ##################################################
    df_synth = df_synth.reset_index(drop=True)[df_test.columns]

    cat_features = [
        True if col in cat_cols else False for col in df_test.columns
    ]  # boolean array instead of column names

    # Compute Gower distance for validation and test set (adapted to mixed data): data_y is data to compare
    if meta_classifier is None:
        pairwise_gower_val = gower.gower_matrix(
            data_x=df_val, data_y=df_synth, cat_features=cat_features
        )
        dist_val = np.sort(pairwise_gower_val, axis=1)

        # Fetch the shortest distance for each record in the validation set
        min_dist_val = dist_val[:, 0]

        # Fetch the mean distance of the 5, 10, 50 and 100 closest records
        dcr_val_5 = dist_val[:, :5].mean(axis=1)
        dcr_val_10 = dist_val[:, :10].mean(axis=1)
        dcr_val_20 = dist_val[:, :20].mean(axis=1)
        dcr_val_30 = dist_val[:, :30].mean(axis=1)
        dcr_val_40 = dist_val[:, :40].mean(axis=1)
        dcr_val_50 = dist_val[:, :50].mean(axis=1)

        min_dist_val = pd.DataFrame(min_dist_val, columns=["min_gower_distance"])
        dcr_val_5 = pd.DataFrame(dcr_val_5, columns=["dcr_5"])
        dcr_val_10 = pd.DataFrame(dcr_val_10, columns=["dcr_10"])
        dcr_val_20 = pd.DataFrame(dcr_val_20, columns=["dcr_20"])
        dcr_val_30 = pd.DataFrame(dcr_val_30, columns=["dcr_30"])
        dcr_val_40 = pd.DataFrame(dcr_val_40, columns=["dcr_40"])
        dcr_val_50 = pd.DataFrame(dcr_val_50, columns=["dcr_50"])

        # Compute the NNDR
        nndr_val = np.divide(
            dist_val[:, 0],
            dist_val[:, 1],
            out=np.zeros_like(dist_val[:, 0]),
            where=dist_val[:, 1] != 0,
        )
        nndr_val = pd.DataFrame(nndr_val, columns=["nndr"])

        # Calculate the number of neighbors
        eps_val = np.median(dist_val[:, 0])
        num_neighbor_val = np.sum(np.where(pairwise_gower_val <= eps_val, 1, 0), axis=1)
        num_neighbor_val = pd.DataFrame(num_neighbor_val, columns=["num_of_neighbor"])

    else:
        min_dist_val = None
        dcr_val_5 = None
        dcr_val_10 = None
        dcr_val_20 = None
        dcr_val_30 = None
        dcr_val_40 = None
        dcr_val_50 = None
        nndr_val = None
        num_neighbor_val = None

    # Test set
    pairwise_gower_test = gower.gower_matrix(
        data_x=df_test, data_y=df_synth, cat_features=cat_features
    )
    dist_test = np.sort(pairwise_gower_test, axis=1)

    min_dist_test = dist_test[:, 0]
    dcr_test_5 = dist_test[:, :5].mean(axis=1)
    dcr_test_10 = dist_test[:, :10].mean(axis=1)
    dcr_test_20 = dist_test[:, :20].mean(axis=1)
    dcr_test_30 = dist_test[:, :30].mean(axis=1)
    dcr_test_40 = dist_test[:, :40].mean(axis=1)
    dcr_test_50 = dist_test[:, :50].mean(axis=1)

    min_dist_test = pd.DataFrame(min_dist_test, columns=["min_gower_distance"])
    dcr_test_5 = pd.DataFrame(dcr_test_5, columns=["dcr_5"])
    dcr_test_10 = pd.DataFrame(dcr_test_10, columns=["dcr_10"])
    dcr_test_20 = pd.DataFrame(dcr_test_20, columns=["dcr_20"])
    dcr_test_30 = pd.DataFrame(dcr_test_30, columns=["dcr_30"])
    dcr_test_40 = pd.DataFrame(dcr_test_40, columns=["dcr_40"])
    dcr_test_50 = pd.DataFrame(dcr_test_50, columns=["dcr_50"])

    nndr_test = np.divide(
        dist_test[:, 0],
        dist_test[:, 1],
        out=np.zeros_like(dist_test[:, 0]),
        where=dist_test[:, 1] != 0,
    )
    nndr_test = pd.DataFrame(nndr_test, columns=["nndr"])

    eps_test = np.median(dist_test[:, 0])
    num_neighbor_test = np.sum(np.where(pairwise_gower_test <= eps_test, 1, 0), axis=1)
    num_neighbor_test = pd.DataFrame(num_neighbor_test, columns=["num_of_neighbor"])

    ##################################################
    # Prediction with DOMIAS
    ##################################################

    # Only need to train domias once
    if meta_classifier is None:
        y_val_pred_proba_domias = domias.fit_pred(
            df_ref=df_ref, df_synth=df_synth[df_ref.columns], df_test=df_val
        )
        # Convert prediction to pandas DataFrame
        y_val_pred_proba_domias = pd.DataFrame(
            y_val_pred_proba_domias, columns=["pred_proba_domias"]
        )
    else:
        y_val_pred_proba_domias = None

    y_test_pred_proba_domias = domias.fit_pred(
        df_ref=df_ref, df_synth=df_synth[df_ref.columns], df_test=df_test
    )
    y_test_pred_proba_domias = pd.DataFrame(
        y_test_pred_proba_domias, columns=["pred_proba_domias"]
    )
    for i in range(iteration):
        if meta_classifier is not None:
            meta_classifier = meta_classifier
        else:
            ##################################################
            # Train meta classifier
            ##################################################

            df_val_meta = pd.concat(
                [
                    df_val[config.metadata["continuous"]],
                    min_dist_val,
                    dcr_val_5,
                    dcr_val_10,
                    dcr_val_20,
                    dcr_val_30,
                    dcr_val_40,
                    dcr_val_50,
                    nndr_val,
                    num_neighbor_val,
                    y_val_pred_proba_domias,
                    val_pred_proba_rmia,
                ],
                axis=1,
            )

            if meta_classifier_type == "lr":  # Logistic Regression Model Pipeline
                meta_classifier = fit_lr_pipeline(
                    x=df_val_meta,
                    y=y_val,
                    continuous_cols=list(df_val_meta.columns),
                    categorical_cols=[],
                    bounds={},
                )
            else:  # XGBoost
                meta_classifier = hyperparam_tuning(
                    x=df_val_meta,
                    y=y_val,
                    continuous_cols=list(df_val_meta.columns),
                    categorical_cols=[],
                    bounds={},
                    num_optuna_trials=1000,
                    num_kfolds=5,
                    use_gpu=True,
                )
        ##################################################
        # Evaluate meta classifier on test set
        ##################################################

        df_test_meta = pd.concat(
            [
                df_test[config.metadata["continuous"]],
                min_dist_test,
                dcr_test_5,
                dcr_test_10,
                dcr_test_20,
                dcr_test_30,
                dcr_test_40,
                dcr_test_50,
                nndr_test,
                num_neighbor_test,
                y_test_pred_proba_domias,
                test_pred_proba_rmia,
            ],
            axis=1,
        )

        y_pred_proba_final = meta_classifier.predict_proba(df_test_meta)[:, 1]

        pred_proba_blending_plus_plus.append(y_pred_proba_final)
        meta_classifier_blending_plus_plus.append(meta_classifier)

    return {
        "pred_proba": pred_proba_blending_plus_plus,
        "meta_classifier": meta_classifier_blending_plus_plus,
    }
