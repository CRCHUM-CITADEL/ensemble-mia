# Standard library
import argparse
from pathlib import Path

# 3rd party
import numpy as np
import pandas as pd

# Local
from attack import blending_plus_plus
from utils import draw, standard, stats
import config


def main(
    attack_model: list,
    meta_train_path: str,
    meta_train_label_path: str,
    train_pred_proba_rmia_path: str,
    meta_test_path: str,
    meta_test_label_path: str,
    test_pred_proba_rmia_path: str,
    synth_path: str,
    real_ref_path: str,
    meta_classifier_type: str,
    output_path: str,
) -> None:
    # Load data
    df_meta_train = pd.read_csv(Path(meta_train_path))
    y_meta_train_label = pd.read_csv(Path(meta_train_label_path))["is_train"].to_numpy()
    df_train_pred_proba_rmia = pd.read_csv(Path(train_pred_proba_rmia_path))
    df_meta_test = pd.read_csv(Path(meta_test_path))
    y_meta_test = pd.read_csv(Path(meta_test_label_path))["is_train"].to_numpy()
    df_test_pred_proba_rmia = pd.read_csv(Path(test_pred_proba_rmia_path))
    df_synth = pd.read_csv(Path(synth_path))
    df_real_ref = pd.read_csv(Path(real_ref_path))

    # Decimal place convertion for synthetic data
    df_synth = standard.trans_type(df=df_synth, col_type=config.col_type, decimal=1)

    # Type convertion
    for col in config.metadata["categorical"]:
        df_meta_train[col] = df_meta_train[col].astype("object")
        df_meta_test[col] = df_meta_test[col].astype("object")
        df_synth[col] = df_synth[col].astype("object")
        df_real_ref[col] = df_real_ref[col].astype("object")

    output_path = Path(output_path)

    ##################################################
    # Train and eval
    ##################################################

    # Blending++
    if "Blending++" in attack_model:
        output_blending_plus_plus = blending_plus_plus.fit_pred(
            df_synth=df_synth,
            df_ref=df_real_ref,
            df_test=df_meta_test,
            test_pred_proba_rmia=df_test_pred_proba_rmia,
            cat_cols=config.metadata["categorical"],
            iteration=1,
            meta_classifier=None,
            meta_classifier_type=meta_classifier_type,
            df_val=df_meta_train,
            y_val=y_meta_train_label,
            val_pred_proba_rmia=df_train_pred_proba_rmia,
        )

        pred_proba_blending_plus_plus = output_blending_plus_plus["pred_proba"][0]
        meta_classifier_blending_plus_plus = output_blending_plus_plus[
            "meta_classifier"
        ][0]

        tpr_at_fpr_blending_plus_plus = stats.get_tpr_at_fpr(
            true_membership=y_meta_test,
            predictions=pred_proba_blending_plus_plus,
            max_fpr=0.1,
        )

        print(f"Blending++ TPR at FPR==10%: {tpr_at_fpr_blending_plus_plus}")

        standard.create_directory(output_path / "blending_plus_plus")
        np.savetxt(
            output_path / "blending_plus_plus" / "prediction.csv",
            pred_proba_blending_plus_plus,
            delimiter=",",
        )
        standard.save_pickle(
            obj=meta_classifier_blending_plus_plus,
            folderpath=output_path / "blending_plus_plus",
            filename="meta_classifier",
            date=False,
        )

        draw.plot_pred(
            df_test=df_meta_test,
            y_pred_proba=pred_proba_blending_plus_plus,
            cont_col=config.metadata["continuous"],
            n=50,
            save_path=output_path / "blending_plus_plus" / "plot_pred.jpg",
            mode="eval",
            y_test=y_meta_test,
            seed=config.seed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIAs model training")

    parser.add_argument(
        "--attack_model",
        nargs="+",
        default=["Blending++"],
        type=str,
        help="Name of the attack model.",
    )

    parser.add_argument(
        "--meta_train_path",
        default=None,
        type=str,
        help="Full path of the train data",
    )

    parser.add_argument(
        "--meta_train_label_path",
        default=None,
        type=str,
        help="Full path of the train label",
    )

    parser.add_argument(
        "--train_pred_proba_rmia_path",
        default=None,
        type=str,
        help="Full path of the RMIA prediction for train data",
    )

    parser.add_argument(
        "--meta_test_path",
        default=None,
        type=str,
        help="Full path of the test data",
    )

    parser.add_argument(
        "--meta_test_label_path",
        default=None,
        type=str,
        help="Full path of the test label",
    )

    parser.add_argument(
        "--test_pred_proba_rmia_path",
        default=None,
        type=str,
        help="Full path of the RMIA prediction on test data",
    )

    parser.add_argument(
        "--synth_path",
        default=None,
        type=str,
        help="Full path of the synthetic data",
    )

    parser.add_argument(
        "--real_ref_path",
        default=None,
        type=str,
        help="Full path of the real reference data",
    )

    parser.add_argument(
        "--meta_classifier_type",
        default=None,
        type=str,
        help="type of the meta classifier, lr or xgb",
    )

    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="Output path",
    )

    args = parser.parse_args()
    main(
        attack_model=args.attack_model,
        meta_train_path=args.meta_train_path,
        meta_train_label_path=args.meta_train_label_path,
        train_pred_proba_rmia_path=args.train_pred_proba_rmia_path,
        meta_test_path=args.meta_test_path,
        meta_test_label_path=args.meta_test_label_path,
        test_pred_proba_rmia_path=args.test_pred_proba_rmia_path,
        synth_path=args.synth_path,
        real_ref_path=args.real_ref_path,
        meta_classifier_type=args.meta_classifier_type,
        output_path=args.output_path,
    )
