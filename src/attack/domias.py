# 3rd party packages
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler


def fit_pred(
    df_ref: pd.DataFrame,
    df_synth: pd.DataFrame,
    df_test: pd.DataFrame,
) -> np.ndarray:
    """Fit DOMIAS and output the prediction on the test set

    :param df_ref: the reference population data
    :param df_synth: the synthetic data
    :param df_test: the test data without label

    :return: the predicted probabilities
    """

    density_ref = gaussian_kde(df_ref.values.transpose(1, 0))
    density_synth = gaussian_kde(df_synth.values.transpose(1, 0))

    p_ref = density_ref(df_test.values.transpose(1, 0))
    p_g = density_synth(df_test.values.transpose(1, 0))
    domias_ratio = p_g / p_ref

    scaler = MinMaxScaler()
    pred_proba_domias = scaler.fit_transform(domias_ratio.reshape(-1, 1))

    return pred_proba_domias.flatten()
