# Standard library
from pathlib import Path
import warnings

# 3rd party packages
import numpy as np
import pandas as pd
import prince
from matplotlib import pyplot as plt


def prediction_vis(
    df_real_train: pd.DataFrame,
    df_synth_train: pd.DataFrame,
    df_synth_test: pd.DataFrame,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    cont_col: list,
    n: int,
    save_path: Path,
    seed: int,
) -> None:
    """Visualize the prediction for MIA on a 2D plot

    :param df_real_train: the real data that was used to generate 1st generation synthetic data
    :param df_synth_train: the 1st generation synthetic train data
    :param df_synth_test: the 1st generation synthetic test data
    :param df_test: the features of the test data
    :param y_test: the ground truth
    :param y_pred_proba: the predicted probability of the positive class
    :param cont_col: continuous variables
    :param n: top n% of the prediction to be plotted
    :param save_path: path to save the figure
    :param seed: random state

    :return: None
    """

    # Sort prediction according to score
    y_pred_proba_decend_idx = y_pred_proba.argsort()[::-1]
    top_n_idx = y_pred_proba_decend_idx[: int(len(y_pred_proba_decend_idx) * (n / 100))]

    if len(top_n_idx) < 1:
        warnings.warn(f"Not enough samples in test set to extract top {n}% prediction")
        return None

    # Feature and true label of top n prediction
    df_test_top_n = df_test.iloc[top_n_idx]
    y_true_top_n = y_test[top_n_idx]

    # True positive from the top n prediction
    df_test_top_n_true_pos = df_test_top_n.iloc[np.where(y_true_top_n == 1)]

    # True positive from the test set
    df_test_true_pos = df_test.iloc[np.where(y_test == 1)]

    # Combine all the dataframes
    df_combined = pd.concat(
        [
            df_real_train,
            df_synth_train,
            df_synth_test,
            df_test_true_pos,
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
        x=df_combined_trans.iloc[: len(df_real_train)][0],
        y=df_combined_trans.iloc[: len(df_real_train)][1],
        color="blue",
        edgecolors="none",
        alpha=0.5,
        s=2,
        label=f"Real data (# of observation: {len(df_real_train)})",
    )

    plt.scatter(
        x=df_combined_trans.iloc[
            len(df_real_train) : len(df_real_train)
            + len(df_synth_train)
            + len(df_synth_test)
        ][0],
        y=df_combined_trans.iloc[
            len(df_real_train) : len(df_real_train)
            + len(df_synth_train)
            + len(df_synth_test)
        ][1],
        color="green",
        edgecolors="none",
        alpha=0.3,
        s=2,
        label=f"Synthetic data (# of observation: {len(df_synth_train)+len(df_synth_test)})",
    )

    plt.scatter(
        x=df_combined_trans.iloc[
            len(df_real_train)
            + len(df_synth_train)
            + len(df_synth_test) : len(df_real_train)
            + len(df_synth_train)
            + len(df_synth_test)
            + len(df_test_true_pos)
        ][0],
        y=df_combined_trans.iloc[
            len(df_real_train)
            + len(df_synth_train)
            + len(df_synth_test) : len(df_real_train)
            + len(df_synth_train)
            + len(df_synth_test)
            + len(df_test_true_pos)
        ][1],
        color="orange",
        edgecolors="none",
        alpha=0.7,
        s=2,
        label=f"Test member (# of observation: {len(df_test_true_pos)})",
    )

    plt.scatter(
        x=df_combined_trans.iloc[
            len(df_real_train)
            + len(df_synth_train)
            + len(df_synth_test)
            + len(df_test_true_pos) :
        ][0],
        y=df_combined_trans.iloc[
            len(df_real_train)
            + len(df_synth_train)
            + len(df_synth_test)
            + len(df_test_true_pos) :
        ][1],
        color="red",
        edgecolors="none",
        alpha=0.5,
        s=8,
        marker="*",
        label=f"Correct prediction (top {n}%; {len(df_test_top_n_true_pos)} out of {len(df_test_top_n)})",
    )

    plt.xlabel("Component 0", fontsize=18)
    plt.ylabel("Component 1", fontsize=16)
    plt.legend(markerscale=5)
    plt.savefig(save_path, dpi=1200)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def compare_results(
    names: list,
    precision_top1_results: list,
    precision_top50_results: list,
    ganleaks_results: list,
    mcmembership_results: list,
    save_path: Path,
) -> None:
    """Compare the performance of different MIA methods

    :param names: names of the MIA method
    :param precision_top1_results: the top 1% precision of the different methods
    :param precision_top50_results: the top 50% precision of the different methods
    :param ganleaks_results: the top 1% and 50% precision for GAN-Leaks in this order
    :param mcmembership_results : the top 1% and 50% precision for MC Membership in this order
    :param save_path: path to save the figure

    :return: None
    """

    # Unpack the results for GAN-Leaks and MC Membership
    precision_top1_ganleaks, precision_top50_ganleaks = ganleaks_results
    precision_top1_mcmembership, precision_top50_mcmembership = mcmembership_results

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    axs[0].boxplot(precision_top1_results, labels=names, showmeans=True)
    axs[0].axhline(
        y=precision_top1_ganleaks,
        xmin=0,
        xmax=1,
        linestyle="--",
        linewidth=0.5,
        color="blue",
        label="GAN-Leaks",
    )
    axs[0].axhline(
        y=precision_top1_mcmembership,
        xmin=0,
        xmax=1,
        linestyle="--",
        linewidth=0.5,
        color="purple",
        label="MC Membership",
    )
    axs[0].set(ylim=[0, 1.01])
    axs[0].legend()
    axs[0].set_title("Top 1% precision")
    axs[1].boxplot(precision_top50_results, labels=names, showmeans=True)
    axs[1].axhline(
        y=precision_top50_ganleaks,
        xmin=0,
        xmax=1,
        linestyle="--",
        linewidth=0.5,
        color="blue",
        label="GAN-Leaks",
    )
    axs[1].axhline(
        y=precision_top50_mcmembership,
        xmin=0,
        xmax=1,
        linestyle="--",
        linewidth=0.5,
        color="purple",
        label="MC Membership",
    )
    axs[1].set(ylim=[0, 1.01])
    axs[1].legend()
    axs[1].set_title("Top 50% precision")
    plt.savefig(save_path, dpi=1200)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
