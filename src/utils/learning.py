# Standard library
from typing import Callable, List

# 3rd party packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer
import xgboost as xgb
from optuna.trial import Trial
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local
from .stats import get_tpr_at_fpr


def pipeline_prediction(
    trial: Trial,
    predictor: type(xgb.XGBModel),
    preprocessing: ColumnTransformer,
    loss_function: str,
    use_gpu: bool = False,
) -> Pipeline:
    """
    Create a XGBoost pipeline within an Optuna trial.

    :param trial: the Optuna trial to specify the hyperparameters to tune
    :param predictor: the predictor object
    :param preprocessing: a list of steps to perform before training the model
    :param loss_function: the loss function of the predictor
    :param use_gpu: flag to use GPU computation power to accelerate the learning

    :return: a Sklearn pipeline
    """

    pipe = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "xgboost",
                predictor(
                    n_estimators=100,
                    eta=trial.suggest_float("eta", 0.0001, 0.1, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    subsample=trial.suggest_float("subsample", 0.1, 1),
                    colsample_bytree=trial.suggest_float("colsample_bylevel", 0.5, 1),
                    reg_alpha=trial.suggest_categorical(
                        "reg_alpha",
                        [0, 0.1, 0.5, 1, 5, 10],
                    ),
                    reg_lambda=trial.suggest_categorical(
                        "reg_lambda",
                        [0, 0.1, 0.5, 1, 5, 10, 100],
                    ),
                    tree_method="auto" if not use_gpu else "gpu_hist",
                    objective=loss_function,
                    seed=np.random.randint(1000),
                    verbosity=1,
                ),
            ),
        ]
    )
    return pipe


def objective_cross_val(
    trial: Trial,
    pipeline: Callable[[Trial], Pipeline],
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    num_kfolds: int,
    scoring: str,
) -> float:
    """
    Run a k-fold cross validation within an Optuna trial.

    :param trial: the Optuna trial to specify the hyperparameters to tune
    :param pipeline: a sequence of the data transformations to apply with a final estimator
    :param df_train: the training input as a Pandas dataframe
    :param y_train: the training ground truth
    :param num_kfolds: the number of folds to tune the hyperparameters of the predictor
    :param scoring: the scoring metric to evaluate the predictor performance on the validation set

    :return: the average cross validation score across the k-folds
    """

    cv_scores = cross_val_score(
        pipeline(trial),
        df_train,
        y_train,
        cv=num_kfolds,
        scoring=scoring,
    )
    score = np.mean(cv_scores)

    return score


def hyperparam_tuning(
    x: pd.DataFrame,
    y: np.ndarray,
    continuous_cols: List[str],
    categorical_cols: List[str],
    bounds: dict,
    num_optuna_trials: int,
    num_kfolds: int,
    use_gpu: bool,
) -> Pipeline:
    """
    Train a classifier with hyperparameters tuning.

    :param x: the inputs
    :param y: the ground truth
    :param continuous_cols: the continuous columns
    :param categorical_cols: the categorical columns
    :param bounds: the unique values of each categorical variables in the format of
        {"col1": {"categories": ["a", "b",...], ...}
    :param num_optuna_trials: the number of trials of the optimization process for tuning hyperparameters
    :param num_kfolds: the number of folds to tune the hyperparameters of the classifier
    :param use_gpu: flag to use GPU computation power to accelerate the learning

    :return: the **best_model** with the optimized hyperparameters
    """

    # ColumnTransformers
    preprocessing = ColumnTransformer(
        [
            ("continuous", StandardScaler(), continuous_cols),
            (
                "categorical",
                OneHotEncoder(
                    categories=[bounds[cat]["categories"] for cat in categorical_cols],
                    handle_unknown="ignore",
                ),
                categorical_cols,
            ),
        ],
        verbose_feature_names_out=False,
        remainder="passthrough",
    )

    pipeline = lambda trial: pipeline_prediction(
        trial,
        predictor=xgb.XGBClassifier,
        preprocessing=preprocessing,
        loss_function="binary:logistic",
        use_gpu=use_gpu,
    )
    # Custom scorer
    tpr_scorer = make_scorer(get_tpr_at_fpr, needs_proba=True)

    objective = lambda trial: objective_cross_val(
        trial,
        pipeline=pipeline,
        df_train=x,
        y_train=y,
        num_kfolds=num_kfolds,
        scoring=tpr_scorer,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10, seed=np.random.randint(1000)
        ),
    )
    study.optimize(objective, n_trials=num_optuna_trials)

    # Refit model with best hyperparameters
    best_pipe = pipeline(study.best_trial)
    best_pipe.fit(x, y)

    return best_pipe


def fit_lr_pipeline(
    x: pd.DataFrame,
    y: np.ndarray,
    continuous_cols: List[str],
    categorical_cols: List[str],
    bounds: dict,
) -> Pipeline:
    """
    Transform data and fit a logistic regression pipeline

    :param x: the inputs
    :param y: the ground truth
    :param continuous_cols: the continuous columns
    :param categorical_cols: the categorical columns
    :param bounds: the unique values of each categorical variables in the format of
        {"col1": {"categories": ["a", "b",...], ...}

    :return: the fitted logistic regression pipeline
    """

    preprocessing = ColumnTransformer(
        [
            ("continuous", StandardScaler(), continuous_cols),
            (
                "categorical",
                OneHotEncoder(
                    categories=[bounds[cat]["categories"] for cat in categorical_cols],
                    handle_unknown="ignore",
                ),
                categorical_cols,
            ),
        ],
        verbose_feature_names_out=False,
        remainder="passthrough",
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "lr",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    lr_pipeline.fit(x, y)

    return lr_pipeline
