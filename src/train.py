# Standard library
import argparse
from pathlib import Path

# 3rd party
import numpy as np
import pandas as pd

# Local
from clover.utils.standard import save_pickle
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
from data.process import generate_val_test
from utils import draw, standard, stats


def main(
    real_train_path: str,
    real_val_path: str,
    real_test_path: str,
    synth_train_path: str,
    synth_test_path: str,
    synth_2nd_path: str,
    output_path: str,
) -> None:
    # Load data
    df_real_train = pd.read_csv(Path(real_train_path))
    df_real_val = pd.read_csv(Path(real_val_path))
    df_real_test = pd.read_csv(Path(real_test_path))
    df_synth_train = pd.read_csv(Path(synth_train_path))
    df_synth_test = pd.read_csv(Path(synth_test_path))
    df_synth_2nd = pd.read_csv(Path(synth_2nd_path))

    # Type convertion

    for col in config.metadata["categorical"]:
        df_real_train[col] = df_real_train[col].astype("object")
        df_real_val[col] = df_real_val[col].astype("object")
        df_real_test[col] = df_real_test[col].astype("object")
        df_synth_train[col] = df_synth_train[col].astype("object")
        df_synth_test[col] = df_synth_test[col].astype("object")
        df_synth_2nd[col] = df_synth_2nd[col].astype("object")

    output_path = Path(output_path)

    ##################################################
    # Prepare data
    ##################################################

    # Validation and test data
    df_val, y_val, df_test, y_test = generate_val_test(
        df_real_train=df_real_train,
        df_real_control_val=df_real_val,
        df_real_control_test=df_real_test,
        seed=config.seed,
    )

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

    print(f"LOGAN TPR at FPR==10%: {tpr_at_fpr_logan}")

    standard.create_directory(output_path / "logan")
    np.savetxt(
        output_path / "logan" / "prediction.csv", pred_proba_logan, delimiter=","
    )

    draw.plot_pred(
        df_test=df_test,
        y_pred_proba=pred_proba_logan,
        cont_col=config.metadata["continuous"],
        n=50,
        save_path=output_path / "logan" / "plot_pred.jpg",
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

    print(f"TableGAN TPR at FPR==10%: {tpr_at_fpr_tablegan}")

    standard.create_directory(output_path / "tablegan")
    np.savetxt(
        output_path / "tablegan" / "prediction.csv", pred_proba_tablegan, delimiter=","
    )

    draw.plot_pred(
        df_test=df_test,
        y_pred_proba=pred_proba_tablegan,
        cont_col=config.metadata["continuous"],
        n=50,
        save_path=output_path / "tablegan" / "plot_pred.jpg",
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

    print(f"Soft voting TPR at FPR==10%: {tpr_at_fpr_soft_voting}")

    standard.create_directory(output_path / "soft_voting")
    np.savetxt(
        output_path / "soft_voting" / "prediction.csv",
        pred_proba_soft_voting,
        delimiter=",",
    )

    draw.plot_pred(
        df_test=df_test,
        y_pred_proba=pred_proba_soft_voting,
        cont_col=config.metadata["continuous"],
        n=50,
        save_path=output_path / "soft_voting" / "plot_pred.jpg",
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
        meta_classifier=None,
        df_val=df_val,
        y_val=y_val,
    )

    pred_proba_stacking = output_stacking["pred_proba"][0]
    meta_classifier_stacking = output_stacking["meta_classifier"][0]

    tpr_at_fpr_stacking = stats.get_tpr_at_fpr(
        true_membership=y_test,
        predictions=pred_proba_stacking,
        max_fpr=0.1,
    )

    print(f"Stacking TPR at FPR==10%: {tpr_at_fpr_stacking}")

    standard.create_directory(output_path / "stacking")
    np.savetxt(
        output_path / "stacking" / "prediction.csv", pred_proba_stacking, delimiter=","
    )
    save_pickle(
        obj=meta_classifier_stacking,
        folderpath=output_path / "stacking",
        filename="meta_classifier",
        date=False,
    )

    draw.plot_pred(
        df_test=df_test,
        y_pred_proba=pred_proba_stacking,
        cont_col=config.metadata["continuous"],
        n=50,
        save_path=output_path / "stacking" / "plot_pred.jpg",
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
        meta_classifier=None,
        df_val=df_val,
        y_val=y_val,
    )

    pred_proba_stacking_plus = output_stacking_plus["pred_proba"][0]
    meta_classifier_stacking_plus = output_stacking_plus["meta_classifier"][0]

    tpr_at_fpr_stacking_plus = stats.get_tpr_at_fpr(
        true_membership=y_test,
        predictions=pred_proba_stacking_plus,
        max_fpr=0.1,
    )

    print(f"Stacking+ TPR at FPR==10%: {tpr_at_fpr_stacking_plus}")

    standard.create_directory(output_path / "stacking_plus")
    np.savetxt(
        output_path / "stacking_plus" / "prediction.csv",
        pred_proba_stacking_plus,
        delimiter=",",
    )
    save_pickle(
        obj=meta_classifier_stacking_plus,
        folderpath=output_path / "stacking_plus",
        filename="meta_classifier",
        date=False,
    )

    draw.plot_pred(
        df_test=df_test,
        y_pred_proba=pred_proba_stacking_plus,
        cont_col=config.metadata["continuous"],
        n=50,
        save_path=output_path / "stacking_plus" / "plot_pred.jpg",
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
        meta_classifier=None,
        df_val=df_val,
        y_val=y_val,
        bounds=config.bounds,
    )

    pred_proba_blending = output_blending["pred_proba"][0]
    meta_classifier_blending = output_blending["meta_classifier"][0]

    tpr_at_fpr_blending = stats.get_tpr_at_fpr(
        true_membership=y_test,
        predictions=pred_proba_blending,
        max_fpr=0.1,
    )

    print(f"Blending TPR at FPR==10%: {tpr_at_fpr_blending}")

    standard.create_directory(output_path / "blending")
    np.savetxt(
        output_path / "blending" / "prediction.csv", pred_proba_blending, delimiter=","
    )
    save_pickle(
        obj=meta_classifier_blending,
        folderpath=output_path / "blending" / "plot_pred.jpg",
        filename="meta_classifier",
        date=False,
    )

    draw.plot_pred(
        df_test=df_test,
        y_pred_proba=pred_proba_blending,
        cont_col=config.metadata["continuous"],
        n=50,
        save_path=output_path / "blending" / "plot_pred.jpg",
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
        meta_classifier=None,
        df_val=df_val,
        y_val=y_val,
        bounds=config.bounds,
    )

    pred_proba_blending_plus = output_blending_plus["pred_proba"][0]
    meta_classifier_blending_plus = output_blending_plus["meta_classifier"][0]

    tpr_at_fpr_blending_plus = stats.get_tpr_at_fpr(
        true_membership=y_test,
        predictions=pred_proba_blending_plus,
        max_fpr=0.1,
    )

    print(f"Blending+ TPR at FPR==10%: {tpr_at_fpr_blending_plus}")

    standard.create_directory(output_path / "blending_plus")
    np.savetxt(
        output_path / "blending_plus" / "prediction.csv",
        pred_proba_blending_plus,
        delimiter=",",
    )
    save_pickle(
        obj=meta_classifier_blending_plus,
        folderpath=output_path / "blending_plus",
        filename="meta_classifier",
        date=False,
    )

    draw.plot_pred(
        df_test=df_test,
        y_pred_proba=pred_proba_blending_plus,
        cont_col=config.metadata["continuous"],
        n=50,
        save_path=output_path / "blending_plus" / "plot_pred.jpg",
        mode="eval",
        y_test=y_test,
        seed=config.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIAs model training")
    parser.add_argument(
        "--real_train_path",
        default=None,
        type=str,
        help="Full path of the real train data",
    )

    parser.add_argument(
        "--real_val_path",
        default=None,
        type=str,
        help="Full path of the real validation data",
    )

    parser.add_argument(
        "--real_test_path",
        default=None,
        type=str,
        help="Full path of the real test data",
    )

    parser.add_argument(
        "--synth_train_path",
        default=None,
        type=str,
        help="Full path of the synthetic train data",
    )

    parser.add_argument(
        "--synth_test_path",
        default=None,
        type=str,
        help="Full path of the synthetic test data",
    )

    parser.add_argument(
        "--synth_2nd_path",
        default=None,
        type=str,
        help="Full path of the 2nd generation synthetic data",
    )

    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="Output path",
    )

    args = parser.parse_args()
    main(
        real_train_path=args.real_train_path,
        real_val_path=args.real_val_path,
        real_test_path=args.real_test_path,
        synth_train_path=args.synth_train_path,
        synth_test_path=args.synth_test_path,
        synth_2nd_path=args.synth_2nd_path,
        output_path=args.output_path,
    )
