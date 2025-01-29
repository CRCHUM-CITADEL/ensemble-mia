# Standard library
import argparse
from pathlib import Path

# 3rd party
import numpy as np
import pandas as pd

# Local
from clover.utils.standard import load_pickle
from attack import (
    logan,
    tablegan,
    soft_voting,
    stacking,
    stacking_plus,
    blending,
    blending_plus,
)
import config
from utils import draw, standard


def main(
    meta_classifier_stacking_path: str,
    meta_classifier_stacking_plus_path: str,
    meta_classifier_blending_path: str,
    meta_classifier_blending_plus_path: str,
    dataset: str,
    is_plot: str,
) -> None:
    if is_plot == "True":
        is_plot = True
    elif is_plot == "False":
        is_plot = False
    else:
        raise ValueError("is_plot must be True or False")

    # Input and output folders
    gen_name = config.attack_type.split("_")[0]
    input_dir = config.DATA_PATH / config.attack_type / dataset
    output_dir = config.OUTPUT_PATH / "infer"

    # Load meta classifier for ensemble models
    meta_classifier_stacking = load_pickle(Path(meta_classifier_stacking_path))
    meta_classifier_stacking_plus = load_pickle(
        Path(meta_classifier_stacking_plus_path)
    )
    meta_classifier_blending = load_pickle(Path(meta_classifier_blending_path))
    meta_classifier_blending_plus = load_pickle(
        Path(meta_classifier_blending_plus_path)
    )

    # Load data for each model
    if dataset == "dev":
        infer_id = config.dev_id
    else:
        infer_id = config.final_id

    for data_id in infer_id:
        df_synth_train = pd.read_csv(
            input_dir / f"{gen_name}_{data_id}" / config.synth_train_file
        )
        df_synth_test = pd.read_csv(
            input_dir / f"{gen_name}_{data_id}" / config.synth_test_file
        )
        df_synth_2nd = pd.read_csv(
            input_dir / f"{gen_name}_{data_id}" / config.synth_2nd_file
        )

        # The challenge points
        df_test = pd.read_csv(input_dir / f"{gen_name}_{data_id}" / config.test_file)

        # Drop ids
        df_test = df_test.drop(columns=["trans_id", "account_id"])

        # Type convertion
        for col in config.metadata["categorical"]:
            df_synth_train[col] = df_synth_train[col].astype("object")
            df_synth_test[col] = df_synth_test[col].astype("object")
            df_synth_2nd[col] = df_synth_2nd[col].astype("object")
            df_test[col] = df_test[col].astype("object")

        ##################################################
        # Prepare data
        ##################################################

        # LOGAN
        df_train_logan, y_train_logan = logan.prepare_data(
            df_synth_train=df_synth_train, df_synth_2nd=df_synth_2nd
        )

        # TableGAN
        (
            df_train_tablegan_discriminator,
            y_train_tablegan_discriminator,
            df_train_tablegan_classifier,
            y_train_tablegan_classifier,
        ) = tablegan.prepare_data(
            df_synth_train=df_synth_train,
            df_synth_test=df_synth_test,
            df_synth_2nd=df_synth_2nd,
            size_1st_gen_cla=len(df_synth_test),
            size_2nd_gen_dis=len(df_synth_train) - len(df_synth_test),
            seed=config.seed,
        )

        ##################################################
        # Train and eval
        ##################################################

        # LOGAN
        pred_proba_logan = logan.fit_pred(
            df_train_logan=df_train_logan,
            y_train_logan=y_train_logan,
            df_test=df_test,
            cont_cols=config.metadata["continuous"],
            cat_cols=config.metadata["categorical"],
            iteration=1,
        )[0]

        current_logan_dir = (
            output_dir
            / "logan"
            / config.attack_type
            / dataset
            / f"{gen_name}_{data_id}"
        )
        standard.create_directory(current_logan_dir)

        np.savetxt(
            current_logan_dir / "prediction.csv",
            pred_proba_logan,
            delimiter=",",
        )

        if is_plot:
            draw.plot_pred(
                df_test=df_test,
                y_pred_proba=pred_proba_logan,
                cont_col=config.metadata["continuous"],
                n=50,
                save_path=current_logan_dir / "plot_pred.jpg",
                mode="infer",
                seed=config.seed,
            )

        # TableGAN
        pred_proba_tablegan = tablegan.fit_pred(
            df_train_tablegan_discriminator=df_train_tablegan_discriminator,
            y_train_tablegan_discriminator=y_train_tablegan_discriminator,
            df_train_tablegan_classifier=df_train_tablegan_classifier,
            y_train_tablegan_classifier=y_train_tablegan_classifier,
            df_test=df_test,
            cont_cols=config.metadata["continuous"],
            cat_cols=config.metadata["categorical"],
            iteration=1,
        )[0]

        current_tablegan_dir = (
            output_dir
            / "tablegan"
            / config.attack_type
            / dataset
            / f"{gen_name}_{data_id}"
        )
        standard.create_directory(current_tablegan_dir)

        np.savetxt(
            current_tablegan_dir / "prediction.csv",
            pred_proba_tablegan,
            delimiter=",",
        )

        if is_plot:
            draw.plot_pred(
                df_test=df_test,
                y_pred_proba=pred_proba_tablegan,
                cont_col=config.metadata["continuous"],
                n=50,
                save_path=current_tablegan_dir / "plot_pred.jpg",
                mode="infer",
                seed=config.seed,
            )

        # Soft voting
        pred_proba_soft_voting = soft_voting.fit_pred(
            df_train_logan=df_train_logan,
            y_train_logan=y_train_logan,
            df_train_tablegan_discriminator=df_train_tablegan_discriminator,
            y_train_tablegan_discriminator=y_train_tablegan_discriminator,
            df_train_tablegan_classifier=df_train_tablegan_classifier,
            y_train_tablegan_classifier=y_train_tablegan_classifier,
            df_test=df_test,
            cont_cols=config.metadata["continuous"],
            cat_cols=config.metadata["categorical"],
            iteration=1,
        )[0]

        current_soft_voting_dir = (
            output_dir
            / "soft_voting"
            / config.attack_type
            / dataset
            / f"{gen_name}_{data_id}"
        )
        standard.create_directory(current_soft_voting_dir)

        np.savetxt(
            current_soft_voting_dir / "prediction.csv",
            pred_proba_soft_voting,
            delimiter=",",
        )

        if is_plot:
            draw.plot_pred(
                df_test=df_test,
                y_pred_proba=pred_proba_soft_voting,
                cont_col=config.metadata["continuous"],
                n=50,
                save_path=current_soft_voting_dir / "plot_pred.jpg",
                mode="infer",
                seed=config.seed,
            )

        # Stacking
        output_stacking = stacking.fit_pred(
            df_train_logan=df_train_logan,
            y_train_logan=y_train_logan,
            df_train_tablegan_discriminator=df_train_tablegan_discriminator,
            y_train_tablegan_discriminator=y_train_tablegan_discriminator,
            df_train_tablegan_classifier=df_train_tablegan_classifier,
            y_train_tablegan_classifier=y_train_tablegan_classifier,
            df_test=df_test,
            cont_cols=config.metadata["continuous"],
            cat_cols=config.metadata["categorical"],
            iteration=1,
            meta_classifier=meta_classifier_stacking,
        )

        pred_proba_stacking = output_stacking["pred_proba"][0]

        current_stacking_dir = (
            output_dir
            / "stacking"
            / config.attack_type
            / dataset
            / f"{gen_name}_{data_id}"
        )
        standard.create_directory(current_stacking_dir)

        np.savetxt(
            current_stacking_dir / "prediction.csv",
            pred_proba_stacking,
            delimiter=",",
        )

        if is_plot:
            draw.plot_pred(
                df_test=df_test,
                y_pred_proba=pred_proba_stacking,
                cont_col=config.metadata["continuous"],
                n=50,
                save_path=current_stacking_dir / "plot_pred.jpg",
                mode="infer",
                seed=config.seed,
            )

        # Stacking+
        output_stacking_plus = stacking_plus.fit_pred(
            df_synth_train=df_synth_train,
            df_synth_test=df_synth_test,
            df_train_logan=df_train_logan,
            y_train_logan=y_train_logan,
            df_train_tablegan_discriminator=df_train_tablegan_discriminator,
            y_train_tablegan_discriminator=y_train_tablegan_discriminator,
            df_train_tablegan_classifier=df_train_tablegan_classifier,
            y_train_tablegan_classifier=y_train_tablegan_classifier,
            df_test=df_test,
            cont_cols=config.metadata["continuous"],
            cat_cols=config.metadata["categorical"],
            iteration=1,
            meta_classifier=meta_classifier_stacking_plus,
        )

        pred_proba_stacking_plus = output_stacking_plus["pred_proba"][0]

        current_stacking_plus_dir = (
            output_dir
            / "stacking_plus"
            / config.attack_type
            / dataset
            / f"{gen_name}_{data_id}"
        )
        standard.create_directory(current_stacking_plus_dir)

        np.savetxt(
            current_stacking_plus_dir / "prediction.csv",
            pred_proba_stacking_plus,
            delimiter=",",
        )

        if is_plot:
            draw.plot_pred(
                df_test=df_test,
                y_pred_proba=pred_proba_stacking_plus,
                cont_col=config.metadata["continuous"],
                n=50,
                save_path=current_stacking_plus_dir / "plot_pred.jpg",
                mode="infer",
                seed=config.seed,
            )

        # Blending
        output_blending = blending.fit_pred(
            df_train_logan=df_train_logan,
            y_train_logan=y_train_logan,
            df_train_tablegan_discriminator=df_train_tablegan_discriminator,
            y_train_tablegan_discriminator=y_train_tablegan_discriminator,
            df_train_tablegan_classifier=df_train_tablegan_classifier,
            y_train_tablegan_classifier=y_train_tablegan_classifier,
            df_test=df_test,
            cont_cols=config.metadata["continuous"],
            cat_cols=config.metadata["categorical"],
            iteration=1,
            meta_classifier=meta_classifier_blending,
        )

        pred_proba_blending = output_blending["pred_proba"][0]

        current_blending_dir = (
            output_dir
            / "blending"
            / config.attack_type
            / dataset
            / f"{gen_name}_{data_id}"
        )
        standard.create_directory(current_blending_dir)

        np.savetxt(
            current_blending_dir / "prediction.csv",
            pred_proba_blending,
            delimiter=",",
        )

        if is_plot:
            draw.plot_pred(
                df_test=df_test,
                y_pred_proba=pred_proba_blending,
                cont_col=config.metadata["continuous"],
                n=50,
                save_path=current_blending_dir / "plot_pred.jpg",
                mode="infer",
                seed=config.seed,
            )

        # Blending+
        output_blending_plus = blending_plus.fit_pred(
            df_synth_train=df_synth_train,
            df_synth_test=df_synth_test,
            df_train_logan=df_train_logan,
            y_train_logan=y_train_logan,
            df_train_tablegan_discriminator=df_train_tablegan_discriminator,
            y_train_tablegan_discriminator=y_train_tablegan_discriminator,
            df_train_tablegan_classifier=df_train_tablegan_classifier,
            y_train_tablegan_classifier=y_train_tablegan_classifier,
            df_test=df_test,
            cont_cols=config.metadata["continuous"],
            cat_cols=config.metadata["categorical"],
            iteration=1,
            meta_classifier=meta_classifier_blending_plus,
        )

        pred_proba_blending_plus = output_blending_plus["pred_proba"][0]

        current_blending_plus_dir = (
            output_dir
            / "blending_plus"
            / config.attack_type
            / dataset
            / f"{gen_name}_{data_id}"
        )
        standard.create_directory(current_blending_plus_dir)

        np.savetxt(
            current_blending_plus_dir / "prediction.csv",
            pred_proba_blending_plus,
            delimiter=",",
        )

        if is_plot:
            draw.plot_pred(
                df_test=df_test,
                y_pred_proba=pred_proba_blending_plus,
                cont_col=config.metadata["continuous"],
                n=50,
                save_path=current_blending_plus_dir / "plot_pred.jpg",
                mode="infer",
                seed=config.seed,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIAs model inference")

    parser.add_argument(
        "--meta_classifier_stacking_path",
        default=None,
        type=str,
        help="Full path of the trained meta classifier for stacking model",
    )

    parser.add_argument(
        "--meta_classifier_stacking_plus_path",
        default=None,
        type=str,
        help="Full path of the trained meta classifier for stacking+ model",
    )

    parser.add_argument(
        "--meta_classifier_blending_path",
        default=None,
        type=str,
        help="Full path of the trained meta classifier for blending model",
    )

    parser.add_argument(
        "--meta_classifier_blending_plus_path",
        default=None,
        type=str,
        help="Full path of the trained meta classifier for blending+ model",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Name of dataset: dev or final",
    )

    parser.add_argument(
        "--is_plot",
        default=None,
        type=str,
        help="If to plot the prediction: True or False",
    )

    args = parser.parse_args()
    main(
        meta_classifier_stacking_path=args.meta_classifier_stacking_path,
        meta_classifier_stacking_plus_path=args.meta_classifier_stacking_plus_path,
        meta_classifier_blending_path=args.meta_classifier_blending_path,
        meta_classifier_blending_plus_path=args.meta_classifier_blending_plus_path,
        dataset=args.dataset,
        is_plot=args.is_plot,
    )
