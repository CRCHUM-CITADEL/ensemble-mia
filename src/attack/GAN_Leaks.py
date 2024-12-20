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
from metrics.privacy.membership import GANLeaks
from mia_ensemble.src.utils import draw


def model(
    df_real_train: pd.DataFrame,
    df_synth_train: pd.DataFrame,
    df_synth_test: pd.DataFrame,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    metadata: dict,
    save_path: Path,
    seed: int,
) -> Tuple[float, float]:
    """Membership inference attack with GANLeaks

    :param df_real_train: the real train data
    :param df_synth_train: the 1st generation synthetic train data
    :param df_synth_test: the 1st generation synthetic test data
    :param df_test: the test data
    :param y_test: the test label
    :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
    :param save_path: the path to save the plot
    :param seed: for reproduction

    :return: top 1% precision and top 50% precision of the predictions
    """

    ganleaks = GANLeaks()

    precision_top1_ganleaks, precision_top50_ganleaks, _, _, DCRs = ganleaks.eval(
        df_test=df_test,
        y_test=y_test,
        df_synth=df_synth_train,
        cat_cols=metadata["categorical"],
    )

    # Convert the distance to probability
    y_pred_proba = np.exp(-DCRs)

    draw.prediction_vis(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_test=df_test,
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        cont_col=metadata["continuous"],
        n=1,
        save_path=save_path / "ganleaks_output.jpg",
        seed=seed,
    )

    return precision_top1_ganleaks, precision_top50_ganleaks
