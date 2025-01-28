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
from utils import draw, standard, stats


def main(
    meta_classifier_stacking_path: str,
    meta_classifier_stacking_plus_path: str,
    meta_classifier_blending_path: str,
    meta_classifier_blending_plus_path: str,
) -> None:
    # Input and output folders
    gen_name = config.attack_type.split("_")[0]
    input_dir = config.DATA_PATH / config.attack_type / "train"
    output_dir = config.OUTPUT_PATH / "eval" / config.attack_type

    # Load meta classifier for ensemble models
    meta_classifier_stacking = load_pickle(Path(meta_classifier_stacking_path))
    meta_classifier_stacking_plus = load_pickle(
        Path(meta_classifier_stacking_plus_path)
    )
    meta_classifier_blending = load_pickle(Path(meta_classifier_blending_path))
    meta_classifier_blending_plus = load_pickle(
        Path(meta_classifier_blending_plus_path)
    )

    # Initial metrics list
    tpr_at_fpr_dict = {
        "LOGAN": [],
        "TableGAN": [],
        "Soft Voting": [],
        "Stacking": [],
        "Stacking+": [],
        "Blending": [],
        "Blending+": [],
    }

    # Load data for each shadow model
    for data_id in config.train_id:
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
        y_test = pd.read_csv(input_dir / f"{gen_name}_{data_id}" / config.test_label)

        # Drop ids
        df_test = df_test.drop(columns=["trans_id", "account_id"])

        # Type convertion
        for col in config.metadata["categorical"]:
            df_synth_train[col] = df_synth_train[col].astype("object")
            df_synth_test[col] = df_synth_test[col].astype("object")
            df_synth_2nd[col] = df_synth_2nd[col].astype("object")
            df_test[col] = df_test[col].astype("object")

        y_test = y_test["is_train"].to_numpy()

        ##################################################
        # Prepare data
        ##################################################

        # LOGAN
        df_train_logan, y_train_logan = logan.prepare_data(
            df_synth_train=df_synth_train,
            df_synth_2nd=df_synth_2nd,
            size=len(df_synth_2nd),
            seed=config.seed,
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
            size=len(df_synth_test),
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

        tpr_at_fpr_logan = stats.get_tpr_at_fpr(
            true_membership=y_test,
            predictions=pred_proba_logan,
            max_fpr=0.1,
        )

        tpr_at_fpr_dict["LOGAN"].append(tpr_at_fpr_logan)
        print(f"LOGAN TPR at FPR==10% for {gen_name}_{data_id}: {tpr_at_fpr_logan}")

        standard.create_directory(output_dir / "logan" / f"{gen_name}_{data_id}")
        np.savetxt(
            output_dir / "logan" / f"{gen_name}_{data_id}" / "prediction.csv",
            pred_proba_logan,
            delimiter=",",
        )

        draw.plot_pred(
            df_test=df_test,
            y_pred_proba=pred_proba_logan,
            cont_col=config.metadata["continuous"],
            n=50,
            save_path=output_dir / "logan" / f"{gen_name}_{data_id}" / "plot_pred.jpg",
            mode="eval",
            y_test=y_test,
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

        tpr_at_fpr_tablegan = stats.get_tpr_at_fpr(
            true_membership=y_test,
            predictions=pred_proba_tablegan,
            max_fpr=0.1,
        )

        tpr_at_fpr_dict["TableGAN"].append(tpr_at_fpr_tablegan)
        print(
            f"TableGAN TPR at FPR==10% for {gen_name}_{data_id}: {tpr_at_fpr_tablegan}"
        )

        standard.create_directory(output_dir / "tablegan" / f"{gen_name}_{data_id}")
        np.savetxt(
            output_dir / "tablegan" / f"{gen_name}_{data_id}" / "prediction.csv",
            pred_proba_tablegan,
            delimiter=",",
        )

        draw.plot_pred(
            df_test=df_test,
            y_pred_proba=pred_proba_tablegan,
            cont_col=config.metadata["continuous"],
            n=50,
            save_path=output_dir
            / "tablegan"
            / f"{gen_name}_{data_id}"
            / "plot_pred.jpg",
            mode="eval",
            y_test=y_test,
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

        tpr_at_fpr_soft_voting = stats.get_tpr_at_fpr(
            true_membership=y_test,
            predictions=pred_proba_soft_voting,
            max_fpr=0.1,
        )

        tpr_at_fpr_dict["Soft Voting"].append(tpr_at_fpr_soft_voting)
        print(
            f"Soft voting TPR at FPR==10% for {gen_name}_{data_id}: {tpr_at_fpr_soft_voting}"
        )

        standard.create_directory(output_dir / "soft_voting" / f"{gen_name}_{data_id}")
        np.savetxt(
            output_dir / "soft_voting" / f"{gen_name}_{data_id}" / "prediction.csv",
            pred_proba_soft_voting,
            delimiter=",",
        )

        draw.plot_pred(
            df_test=df_test,
            y_pred_proba=pred_proba_soft_voting,
            cont_col=config.metadata["continuous"],
            n=50,
            save_path=output_dir
            / "soft_voting"
            / f"{gen_name}_{data_id}"
            / "plot_pred.jpg",
            mode="eval",
            y_test=y_test,
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

        tpr_at_fpr_stacking = stats.get_tpr_at_fpr(
            true_membership=y_test,
            predictions=pred_proba_stacking,
            max_fpr=0.1,
        )

        tpr_at_fpr_dict["Stacking"].append(tpr_at_fpr_stacking)
        print(
            f"Stacking TPR at FPR==10% for {gen_name}_{data_id}: {tpr_at_fpr_stacking}"
        )

        standard.create_directory(output_dir / "stacking" / f"{gen_name}_{data_id}")
        np.savetxt(
            output_dir / "stacking" / f"{gen_name}_{data_id}" / "prediction.csv",
            pred_proba_stacking,
            delimiter=",",
        )

        draw.plot_pred(
            df_test=df_test,
            y_pred_proba=pred_proba_stacking,
            cont_col=config.metadata["continuous"],
            n=50,
            save_path=output_dir
            / "stacking"
            / f"{gen_name}_{data_id}"
            / "plot_pred.jpg",
            mode="eval",
            y_test=y_test,
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

        tpr_at_fpr_stacking_plus = stats.get_tpr_at_fpr(
            true_membership=y_test,
            predictions=pred_proba_stacking_plus,
            max_fpr=0.1,
        )

        tpr_at_fpr_dict["Stacking+"].append(pred_proba_stacking_plus)
        print(
            f"Stacking+ TPR at FPR==10% for {gen_name}_{data_id}: {tpr_at_fpr_stacking_plus}"
        )

        standard.create_directory(
            output_dir / "stacking_plus" / f"{gen_name}_{data_id}"
        )
        np.savetxt(
            output_dir / "stacking_plus" / f"{gen_name}_{data_id}" / "prediction.csv",
            pred_proba_stacking_plus,
            delimiter=",",
        )

        draw.plot_pred(
            df_test=df_test,
            y_pred_proba=pred_proba_stacking_plus,
            cont_col=config.metadata["continuous"],
            n=50,
            save_path=output_dir
            / "stacking_plus"
            / f"{gen_name}_{data_id}"
            / "plot_pred.jpg",
            mode="eval",
            y_test=y_test,
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

        tpr_at_fpr_blending = stats.get_tpr_at_fpr(
            true_membership=y_test,
            predictions=pred_proba_blending,
            max_fpr=0.1,
        )

        tpr_at_fpr_dict["Blending"].append(tpr_at_fpr_blending)
        print(f"Blending TPR at FPR==10%: {tpr_at_fpr_blending}")

        standard.create_directory(output_dir / "blending" / f"{gen_name}_{data_id}")
        np.savetxt(
            output_dir / "blending" / f"{gen_name}_{data_id}" / "prediction.csv",
            pred_proba_blending,
            delimiter=",",
        )

        draw.plot_pred(
            df_test=df_test,
            y_pred_proba=pred_proba_blending,
            cont_col=config.metadata["continuous"],
            n=50,
            save_path=output_dir
            / "blending"
            / f"{gen_name}_{data_id}"
            / "plot_pred.jpg",
            mode="eval",
            y_test=y_test,
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

        tpr_at_fpr_blending_plus = stats.get_tpr_at_fpr(
            true_membership=y_test,
            predictions=pred_proba_blending_plus,
            max_fpr=0.1,
        )

        tpr_at_fpr_dict["Blending+"].append(tpr_at_fpr_blending_plus)
        print(
            f"Blending+ TPR at FPR==10% for {gen_name}_{data_id}: {tpr_at_fpr_blending_plus}"
        )

        standard.create_directory(
            output_dir / "blending_plus" / f"{gen_name}_{data_id}"
        )
        np.savetxt(
            output_dir / "blending_plus" / f"{gen_name}_{data_id}" / "prediction.csv",
            pred_proba_blending_plus,
            delimiter=",",
        )

        draw.plot_pred(
            df_test=df_test,
            y_pred_proba=pred_proba_blending_plus,
            cont_col=config.metadata["continuous"],
            n=50,
            save_path=output_dir
            / "blending_plus"
            / f"{gen_name}_{data_id}"
            / "plot_pred.jpg",
            mode="eval",
            y_test=y_test,
            seed=config.seed,
        )

    # Compare all the MIA models
    draw.compare_results(
        method_names=[
            "LOGAN",
            "TableGAN",
            "Soft Voting",
            "Stacking",
            "Stacking+",
            "Blending",
            "Blending+",
        ],
        metric_name="TPR at 10% FPR",
        metrics=tpr_at_fpr_dict,
        save_path=output_dir / "plot_comparison.jpg",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIAs model evaluation")

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

    args = parser.parse_args()
    main(
        meta_classifier_stacking_path=args.meta_classifier_stacking_path,
        meta_classifier_stacking_plus_path=args.meta_classifier_stacking_plus_path,
        meta_classifier_blending_path=args.meta_classifier_blending_path,
        meta_classifier_blending_plus_path=args.meta_classifier_blending_plus_path,
    )
