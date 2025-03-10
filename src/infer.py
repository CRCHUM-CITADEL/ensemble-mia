# Standard library
import argparse
from pathlib import Path

# 3rd party
import numpy as np
import pandas as pd

# Local
from utils.standard import load_pickle
from attack import blending_plus_plus
from utils import draw, standard
import config


def main(
    attack_model: list,
    attack_type: str,
    real_ref_path: str,
    meta_classifier_blending_plus_plus_path: str,
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
    gen_name = attack_type.split("_")[0]
    input_dir = config.DATA_PATH / attack_type / dataset
    output_dir = config.OUTPUT_PATH / "infer"
    rmia_dir = config.RMIA_PRED_PATH / attack_type / dataset

    # Load reference population data and synthetic data
    df_real_ref = pd.read_csv(Path(real_ref_path))

    # Type convertion
    for col in config.metadata["categorical"]:
        df_real_ref[col] = df_real_ref[col].astype("object")

    # Load meta classifier for ensemble models
    if "Blending++" in attack_model:
        meta_classifier_blending_plus_plus = load_pickle(
            Path(meta_classifier_blending_plus_plus_path)
        )
    else:
        meta_classifier_blending_plus_plus = None

    # Load data for each model
    if dataset == "dev":
        infer_id = config.dev_id
    else:
        infer_id = config.final_id

    for data_id in infer_id:
        print("-----------------------------")
        print(f"Predicting for {gen_name}_{data_id}")
        print("-----------------------------")
        df_synth = pd.read_csv(input_dir / f"{gen_name}_{data_id}" / config.synth_file)

        # Decimal place convertion for synthetic data
        df_synth = standard.trans_type(df=df_synth, col_type=config.col_type, decimal=1)

        df_rmia_pred = pd.read_csv(
            rmia_dir / f"{gen_name}_{data_id}" / config.rmia_file
        )

        # The challenge points
        df_test = pd.read_csv(input_dir / f"{gen_name}_{data_id}" / config.test_file)

        # Drop ids
        df_test = df_test.drop(columns=["trans_id", "account_id"])

        # Type convertion
        for col in config.metadata["categorical"]:
            df_synth[col] = df_synth[col].astype("object")
            df_test[col] = df_test[col].astype("object")

        ##################################################
        # Train and eval
        ##################################################

        # Blending++
        if "Blending++" in attack_model:
            output_blending_plus_plus = blending_plus_plus.fit_pred(
                df_synth=df_synth,
                df_ref=df_real_ref,
                df_test=df_test,
                test_pred_proba_rmia=df_rmia_pred,
                cat_cols=config.metadata["categorical"],
                iteration=1,
                meta_classifier=meta_classifier_blending_plus_plus,
            )

            pred_proba_blending_plus_plus = output_blending_plus_plus["pred_proba"][0]

            current_blending_plus_plus_dir = (
                output_dir
                / "blending_plus_plus"
                / attack_type
                / dataset
                / f"{gen_name}_{data_id}"
            )
            standard.create_directory(current_blending_plus_plus_dir)

            np.savetxt(
                current_blending_plus_plus_dir / "prediction.csv",
                pred_proba_blending_plus_plus,
                delimiter=",",
            )

            if is_plot:
                draw.plot_pred(
                    df_test=df_test,
                    y_pred_proba=pred_proba_blending_plus_plus,
                    cont_col=config.metadata["continuous"],
                    n=50,
                    save_path=current_blending_plus_plus_dir / "plot_pred.jpg",
                    mode="infer",
                    seed=config.seed,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIAs model inference")

    parser.add_argument(
        "--attack_model",
        nargs="+",
        default=[
            "LOGAN",
            "TableGAN",
            "Soft Voting",
            "Stacking",
            "Stacking+",
            "Blending",
            "Blending+",
            "Blending++",
        ],
        type=str,
        help="Name of the attack model. Available models are: LOGAN, TableGAN, Soft Voting, Stacking, Stacking+, Blending, Blending+ and Blending++",
    )

    parser.add_argument(
        "--attack_type",
        default=None,
        type=str,
        help="Setting to attack, tabddpm_black_box or tabsyn_black_box",
    )

    parser.add_argument(
        "--real_ref_path",
        default=None,
        type=str,
        help="Full path of the real reference/population data",
    )

    parser.add_argument(
        "--meta_classifier_blending_plus_plus_path",
        default=None,
        type=str,
        help="Full path of the trained meta classifier for blending++ model",
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
        attack_model=args.attack_model,
        attack_type=args.attack_type,
        real_ref_path=args.real_ref_path,
        meta_classifier_blending_plus_plus_path=args.meta_classifier_blending_plus_plus_path,
        dataset=args.dataset,
        is_plot=args.is_plot,
    )
