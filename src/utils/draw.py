# Standard library
from pathlib import Path
import warnings

# 3rd party packages
import numpy as np
import pandas as pd
import prince
from matplotlib import pyplot as plt


def plot_pred(
    df_test: pd.DataFrame,
    y_pred_proba: np.ndarray,
    cont_col: list,
    n: int,
    save_path: Path,
    mode: str = "infer",
    y_test: np.ndarray = None,
    seed: int = None,
) -> None:
    """Visualize the prediction of MIA on a 2D plot

    :param df_test: the features of the test data
    :param y_pred_proba: the predicted probability of the positive class
    :param cont_col: continuous variables
    :param n: top n% of the prediction to be plotted
    :param save_path: path to save the figure
    :param mode: "eval" or "infer". If set to "eval" mode y_test has to be provided.
    :param y_test: the ground truth
    :param seed: random state

    :return: None
    """

    # Sort prediction according to score
    y_pred_proba_decend_idx = y_pred_proba.argsort()[::-1]
    top_n_idx = y_pred_proba_decend_idx[: int(len(y_pred_proba_decend_idx) * (n / 100))]

    if len(top_n_idx) < 1:
        warnings.warn(f"Not enough samples in test set to extract top {n}% prediction")
        return None

    # Feature of top n prediction
    df_test_top_n = df_test.iloc[top_n_idx]

    if mode == "eval":
        # True label of top n prediction
        y_true_top_n = y_test[top_n_idx]

        # True positive from the top n prediction
        df_test_top_n_true_pos = df_test_top_n.iloc[np.where(y_true_top_n == 1)]

        # Members from the test set
        df_test_member = df_test.iloc[np.where(y_test == 1)]

        # Combine all the dataframes
        df_combined = pd.concat(
            [
                df_test,
                df_test_member,
                df_test_top_n,
                df_test_top_n_true_pos,
            ],
            axis=0,
            ignore_index=True,
        )

        # Need to convert continuous variables to float for FAMD
        df_combined[cont_col] = df_combined[cont_col].astype("float")

        # Dimension reduction with FAMD
        famd = prince.FAMD(
            n_components=2,
            n_iter=3,
            copy=True,
            check_input=True,
            random_state=seed,
            engine="sklearn",
            handle_unknown="error",  # same parameter as sklearn.preprocessing.OneHotEncoder
        )

        famd = famd.fit(df_combined)
        df_combined_trans = famd.transform(df_combined)

        plt.figure(figsize=(16, 9))
        plt.scatter(
            x=df_combined_trans.iloc[: len(df_test)][0],
            y=df_combined_trans.iloc[: len(df_test)][1],
            color="orange",
            edgecolors="none",
            alpha=1,
            s=2,
            label=f"Non-members (# of observation: {len(df_test)-len(df_test_member)})",
        )

        plt.scatter(
            x=df_combined_trans.iloc[len(df_test) : len(df_test) + len(df_test_member)][
                0
            ],
            y=df_combined_trans.iloc[len(df_test) : len(df_test) + len(df_test_member)][
                1
            ],
            color="blue",
            edgecolors="none",
            alpha=1,
            s=2,
            label=f"Members (# of observation: {len(df_test_member)})",
        )

        plt.scatter(
            x=df_combined_trans.iloc[
                len(df_test)
                + len(df_test_member) : len(df_test)
                + len(df_test_member)
                + len(df_test_top_n)
            ][0],
            y=df_combined_trans.iloc[
                len(df_test)
                + len(df_test_member) : len(df_test)
                + len(df_test_member)
                + len(df_test_top_n)
            ][1],
            color="red",
            edgecolors="none",
            alpha=1,
            s=2,
            marker="*",
            label="Wrongly predicted members",
        )

        plt.scatter(
            x=df_combined_trans.iloc[
                len(df_test) + len(df_test_member) + len(df_test_top_n) :
            ][0],
            y=df_combined_trans.iloc[
                len(df_test) + len(df_test_member) + len(df_test_top_n) :
            ][1],
            color="green",
            edgecolors="none",
            alpha=1,
            s=2,
            marker="*",
            label=f"Correctly predicted members (top {n}%; {len(df_test_top_n_true_pos)} out of {len(df_test_top_n)})",
        )

        plt.xlabel("Component 0", fontsize=18)
        plt.ylabel("Component 1", fontsize=16)
        plt.legend(markerscale=5)
        plt.savefig(save_path, dpi=1200)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    else:
        # Combine all the dataframes
        df_combined = pd.concat(
            [
                df_test,
                df_test_top_n,
            ],
            axis=0,
            ignore_index=True,
        )

        # Need to convert continuous variables to float for FAMD
        df_combined[cont_col] = df_combined[cont_col].astype("float")

        # Dimension reduction with FAMD
        famd = prince.FAMD(
            n_components=2,
            n_iter=3,
            copy=True,
            check_input=True,
            random_state=seed,
            engine="sklearn",
            handle_unknown="error",  # same parameter as sklearn.preprocessing.OneHotEncoder
        )

        famd = famd.fit(df_combined)
        df_combined_trans = famd.transform(df_combined)

        plt.figure(figsize=(16, 9))
        plt.scatter(
            x=df_combined_trans.iloc[: len(df_test)][0],
            y=df_combined_trans.iloc[: len(df_test)][1],
            color="brown",
            edgecolors="none",
            alpha=1,
            s=2,
            label=f"Test data (# of observation: {len(df_test)})",
        )

        plt.scatter(
            x=df_combined_trans.iloc[len(df_test) :][0],
            y=df_combined_trans.iloc[len(df_test) :][1],
            color="green",
            edgecolors="none",
            alpha=1,
            s=2,
            marker="*",
            label=f"Predicted members (top {n}%; # of prediction: {len(df_test_top_n)})",
        )

        plt.xlabel("Component 0", fontsize=18)
        plt.ylabel("Component 1", fontsize=16)
        plt.legend(markerscale=5)
        plt.savefig(save_path, dpi=1200)
        plt.show(block=False)
        plt.pause(3)
        plt.close()


def compare_results(
    method_names: list,
    metric_name: str,
    metrics: dict,
    save_path: Path,
) -> None:
    """Compare the performance of different MIA methods

    :param method_names: names of the MIA method
    :param metric_name: name of the metric
    :param metrics: metrics from different MIA method, the format of {"MIA method": [xx, xx, ...], ...}
    :param save_path: path to save the figure

    :return: None
    """

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.boxplot(
        [metrics[method] for method in method_names],
        labels=method_names,
        showmeans=True,
    )
    # ax.set(ylim=[0, 0.2])
    ax.legend()
    ax.set_title(metric_name)
    plt.savefig(save_path, dpi=1200)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
