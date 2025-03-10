# Standard library
import argparse
from datetime import datetime
from pathlib import Path

# 3rd party
import numpy as np
import pandas as pd

# Local
from utils.standard import load_pickle
from attack import blending_plus_plus
from utils import draw, standard, stats
import config


def main(
    attack_model: list,
    attack_type: str,
    real_ref_path: str,
    meta_classifier_blending_plus_plus_path: str,
) -> None:
    # Input and output folders
    gen_name = attack_type.split("_")[0]
    input_dir = config.DATA_PATH / attack_type / "train"
    output_dir = config.OUTPUT_PATH / "eval" / attack_type
    rmia_dir = config.DATA_PATH / attack_type / "train"

    # Load reference population data
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

    # Initial metrics list
    tpr_at_fpr_dict = {"Blending++": []}

    # Load data for each shadow model
    for data_id in config.train_id:
        print("-----------------------------")
        print(f"Evaluating for {gen_name}_{data_id}")
        print("-----------------------------")
        df_synth = pd.read_csv(input_dir / f"{gen_name}_{data_id}" / config.synth_file)

        # Decimal place convertion for synthetic data
        df_synth = standard.trans_type(df=df_synth, col_type=config.col_type, decimal=1)

        df_rmia_pred = pd.read_csv(
            rmia_dir / f"{gen_name}_{data_id}" / config.rmia_file
        )

        # The challenge points
        df_test = pd.read_csv(input_dir / f"{gen_name}_{data_id}" / config.test_file)
        y_test = pd.read_csv(input_dir / f"{gen_name}_{data_id}" / config.test_label)

        # Drop ids
        df_test = df_test.drop(columns=["trans_id", "account_id"])

        # Type convertion
        for col in config.metadata["categorical"]:
            df_synth[col] = df_synth[col].astype("object")
            df_test[col] = df_test[col].astype("object")

        y_test = y_test["is_train"].to_numpy()

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

            tpr_at_fpr_blending_plus_plus = stats.get_tpr_at_fpr(
                true_membership=y_test,
                predictions=pred_proba_blending_plus_plus,
                max_fpr=0.1,
            )

            tpr_at_fpr_dict["Blending++"].append(tpr_at_fpr_blending_plus_plus)
            print(
                f"Blending++ TPR at FPR==10% for {gen_name}_{data_id}: {tpr_at_fpr_blending_plus_plus}"
            )

            standard.create_directory(
                output_dir / "blending_plus_plus" / f"{gen_name}_{data_id}"
            )
            np.savetxt(
                output_dir
                / "blending_plus_plus"
                / f"{gen_name}_{data_id}"
                / "prediction.csv",
                pred_proba_blending_plus_plus,
                delimiter=",",
            )

            draw.plot_pred(
                df_test=df_test,
                y_pred_proba=pred_proba_blending_plus_plus,
                cont_col=config.metadata["continuous"],
                n=50,
                save_path=output_dir
                / "blending_plus_plus"
                / f"{gen_name}_{data_id}"
                / "plot_pred.jpg",
                mode="eval",
                y_test=y_test,
                seed=config.seed,
            )

    # Compare all the MIA models
    current_date = datetime.now().strftime("%Y-%m-%d")

    draw.compare_results(
        method_names=attack_model,
        metric_name="TPR at 10% FPR",
        metrics=tpr_at_fpr_dict,
        save_path=output_dir / f"plot_comparison_{current_date}.jpg",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIAs model evaluation")

    parser.add_argument(
        "--attack_model",
        nargs="+",
        default=["Blending++"],
        type=str,
        help="Name of the attack model.",
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

    args = parser.parse_args()
    main(
        attack_model=args.attack_model,
        attack_type=args.attack_type,
        real_ref_path=args.real_ref_path,
        meta_classifier_blending_plus_plus_path=args.meta_classifier_blending_plus_plus_path,
    )
