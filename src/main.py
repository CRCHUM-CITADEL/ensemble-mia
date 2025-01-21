# Standard library
import argparse
import os
from pathlib import Path
from typing import Union

# Local
from clover.metrics.utility.report import UtilityReport
import data.real_data as real_data
import data.synthetic_data as synthetic_data

from attack import (
    GAN_Leaks,
    Monte_Carlo,
    LOGAN,
    TableGAN,
    Detector,
    Ensemble,
    soft_voting,
    stacking,
    stacking_plus,
    blending,
    blending_plus,
)
import config
from utils import draw


def create_directory(path: Union[Path, str]) -> None:
    """
    Create directory if it does not exist

    :param path: the directory to be created
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def main(generator, iteration) -> None:
    ###############################
    # Directories creation
    ###############################
    root_dir = config.OUTPUT_PATH / generator
    data_dir = root_dir / "data"
    gen_dir = root_dir / "generator"
    for dir_ in [
        data_dir / "real",
        data_dir / "1st_generation",
        data_dir / "2nd_generation",
        gen_dir / "1st_generation",
        gen_dir / "2nd_generation",
        root_dir / "plot",
        root_dir / "report",
    ]:
        create_directory(dir_)

    ###############################
    # Data preparation
    ###############################

    # Split the real data
    (
        df_real_train,
        df_real_control_detector,
        df_real_control_val,
        df_real_control_test,
    ) = real_data.split_real_data(
        data_path=config.DATASET_FILEPATH,
        seed=config.seed,
        var_to_predict=config.metadata["variable_to_predict"],
        save_folder=config.OUTPUT_PATH / generator / "data" / "real",
    )

    # Generate 1st generation synthetic data
    synth_train_size = (
        len(df_real_train) * 2
    )  # The size is 2x of df_real_train, because it will be split into 2 for the 2-steps training of TableGAN
    synth_test_size = len(df_real_train)  # Used to train TableGAN
    df_synth_train, df_synth_test = synthetic_data.generate_synth_data(
        df_real=df_real_train,
        metadata=config.metadata,
        generator=generator,
        synth_size=[synth_train_size, synth_test_size],
        output_path=config.OUTPUT_PATH,
        save_folder="1st_generation",
        seed=config.seed,
    )

    # Compute and save the utility metrics
    df_real_dict = {"train": df_real_train, "test": df_real_control_test}
    df_synth_dict = {
        "train": df_synth_train.sample(
            n=len(df_real_train),
            replace=False,
            ignore_index=True,
            random_state=config.seed,
        ),
        "test": df_synth_test.sample(
            n=len(df_real_control_test),
            replace=False,
            ignore_index=True,
            random_state=config.seed,
        ),
    }

    report = UtilityReport(
        dataset_name="Dataset",
        df_real=df_real_dict,
        df_synthetic=df_synth_dict,
        metadata=config.metadata,
        figsize=(8, 6),
        random_state=config.seed,
        report_filepath=None,
        metrics=None,
        cross_learning=False,
        num_repeat=1,  # for the metrics relying on predictions to account for randomness
        num_kfolds=3,  # the number of folds to tune the hyperparameters for the metrics relying on predictors
        num_optuna_trials=20,  # the number of trials of the optimization process for tuning hyperparameters for the metrics relying on predictors
        use_gpu=True,  # run the learning tasks on the GPU
        alpha=0.05,  # for the pairwise chi-square metric
    )
    report.compute()
    print(
        report.summary()
        .groupby(["name", "objective", "min", "max"])
        .apply(
            lambda x: x.drop(["name", "objective", "min", "max"], axis=1).reset_index(
                drop=True
            )
        )
    )
    report.save(
        savepath=config.OUTPUT_PATH / generator / "report", filename="utility_report"
    )

    # Generate 2nd generation synthetic data
    df_synth_2nd = synthetic_data.generate_synth_data(
        df_real=df_synth_train,
        metadata=config.metadata,
        generator=generator,
        synth_size=[synth_test_size],
        output_path=config.OUTPUT_PATH,
        save_folder="2nd_generation",
        seed=config.seed,
    )[0]

    # Build train, validation and test sets
    #   - The size of the train set for all the models is the same
    #   - Validation set is only used for ensemble model
    #   - We use the same test set to evaluate each model

    cat_cols = [
        col for col in df_real_train.columns if col not in config.metadata["continuous"]
    ]
    cont_cols = [col for col in df_real_train.columns if col not in cat_cols]

    # LOGAN
    df_train_logan, y_train_logan = LOGAN.prepare_data(
        df_synth_train=df_synth_train,
        df_synth_2nd=df_synth_2nd,
        size=len(df_real_train),
        cat_cols=cat_cols,
        seed=config.seed,
    )

    # TableGAN
    (
        df_train_tablegan_discriminator,
        y_train_tablegan_discriminator,
        df_train_tablegan_classifier,
        y_train_tablegan_classifier,
    ) = TableGAN.prepare_data(
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_synth_2nd=df_synth_2nd,
        size=len(df_synth_test),
        cat_cols=cat_cols,
        seed=config.seed,
    )

    # Detector
    df_train_detector, y_train_detector = Detector.prepare_data(
        df_real_control_detector=df_real_control_detector,
        df_synth_train=df_synth_train,
        size=len(df_real_control_detector),
        cat_cols=cat_cols,
        seed=config.seed,
    )

    # Validation (for ensemble) and test sets
    df_val, y_val, df_test, y_test = Ensemble.prepare_data(
        df_real_train=df_real_train,
        df_real_control_val=df_real_control_val,
        df_real_control_test=df_real_control_test,
        cat_cols=cat_cols,
        seed=config.seed,
    )

    ###############################
    # Attack models
    ###############################

    # GANLeaks
    precision_top1_ganleaks, precision_top50_ganleaks = GAN_Leaks.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_test=df_test,
        y_test=y_test,
        metadata=config.metadata,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # MC Membership
    precision_top1_mcmembership, precision_top50_mcmembership = Monte_Carlo.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_test=df_test,
        y_test=y_test,
        metadata=config.metadata,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # LOGAN
    _, precision_top1_logan, precision_top50_logan = LOGAN.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_train_logan=df_train_logan,
        y_train_logan=y_train_logan,
        df_test=df_test,
        y_test=y_test,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        iteration=iteration,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # TableGAN
    _, precision_top1_tablegan, precision_top50_tablegan = TableGAN.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_train_tablegan_discriminator=df_train_tablegan_discriminator,
        y_train_tablegan_discriminator=y_train_tablegan_discriminator,
        df_train_tablegan_classifier=df_train_tablegan_classifier,
        y_train_tablegan_classifier=y_train_tablegan_classifier,
        df_test=df_test,
        y_test=y_test,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        iteration=iteration,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # Detector
    _, precision_top1_detector, precision_top50_detector = Detector.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_train_detector=df_train_detector,
        y_train_detector=y_train_detector,
        df_test=df_test,
        y_test=y_test,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        iteration=iteration,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # Soft voting
    _, precision_top1_voting, precision_top50_voting = soft_voting.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_train_logan=df_train_logan,
        y_train_logan=y_train_logan,
        df_train_tablegan_discriminator=df_train_tablegan_discriminator,
        y_train_tablegan_discriminator=y_train_tablegan_discriminator,
        df_train_tablegan_classifier=df_train_tablegan_classifier,
        y_train_tablegan_classifier=y_train_tablegan_classifier,
        df_train_detector=df_train_detector,
        y_train_detector=y_train_detector,
        df_test=df_test,
        y_test=y_test,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        iteration=iteration,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # Stacking
    _, precision_top1_stacking, precision_top50_stacking = stacking.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_train_logan=df_train_logan,
        y_train_logan=y_train_logan,
        df_train_tablegan_discriminator=df_train_tablegan_discriminator,
        y_train_tablegan_discriminator=y_train_tablegan_discriminator,
        df_train_tablegan_classifier=df_train_tablegan_classifier,
        y_train_tablegan_classifier=y_train_tablegan_classifier,
        df_train_detector=df_train_detector,
        y_train_detector=y_train_detector,
        df_val=df_val,
        y_val=y_val,
        df_test=df_test,
        y_test=y_test,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        iteration=iteration,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # Stacking+
    (
        _,
        precision_top1_stacking_plus,
        precision_top50_stacking_plus,
    ) = stacking_plus.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_train_logan=df_train_logan,
        y_train_logan=y_train_logan,
        df_train_tablegan_discriminator=df_train_tablegan_discriminator,
        y_train_tablegan_discriminator=y_train_tablegan_discriminator,
        df_train_tablegan_classifier=df_train_tablegan_classifier,
        y_train_tablegan_classifier=y_train_tablegan_classifier,
        df_train_detector=df_train_detector,
        y_train_detector=y_train_detector,
        df_val=df_val,
        y_val=y_val,
        df_test=df_test,
        y_test=y_test,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        iteration=iteration,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # Blending
    _, precision_top1_blending, precision_top50_blending = blending.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_train_logan=df_train_logan,
        y_train_logan=y_train_logan,
        df_train_tablegan_discriminator=df_train_tablegan_discriminator,
        y_train_tablegan_discriminator=y_train_tablegan_discriminator,
        df_train_tablegan_classifier=df_train_tablegan_classifier,
        y_train_tablegan_classifier=y_train_tablegan_classifier,
        df_train_detector=df_train_detector,
        y_train_detector=y_train_detector,
        df_val=df_val,
        y_val=y_val,
        df_test=df_test,
        y_test=y_test,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        iteration=iteration,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # Blending+
    (
        _,
        precision_top1_blending_plus,
        precision_top50_blending_plus,
    ) = blending_plus.model(
        df_real_train=df_real_train,
        df_synth_train=df_synth_train,
        df_synth_test=df_synth_test,
        df_train_logan=df_train_logan,
        y_train_logan=y_train_logan,
        df_train_tablegan_discriminator=df_train_tablegan_discriminator,
        y_train_tablegan_discriminator=y_train_tablegan_discriminator,
        df_train_tablegan_classifier=df_train_tablegan_classifier,
        y_train_tablegan_classifier=y_train_tablegan_classifier,
        df_train_detector=df_train_detector,
        y_train_detector=y_train_detector,
        df_val=df_val,
        y_val=y_val,
        df_test=df_test,
        y_test=y_test,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        iteration=iteration,
        save_path=config.OUTPUT_PATH / generator / "plot",
        seed=config.seed,
    )

    # Show results
    names = [
        "LOGAN",
        "TableGAN",
        "Detector",
        "Voting",
        "Stacking",
        "Stacking+",
        "Blending",
        "Blending+",
    ]
    precision_top1_results = [
        precision_top1_logan,
        precision_top1_tablegan,
        precision_top1_detector,
        precision_top1_voting,
        precision_top1_stacking,
        precision_top1_stacking_plus,
        precision_top1_blending,
        precision_top1_blending_plus,
    ]
    precision_top50_results = [
        precision_top50_logan,
        precision_top50_tablegan,
        precision_top50_detector,
        precision_top50_voting,
        precision_top50_stacking,
        precision_top50_stacking_plus,
        precision_top50_blending,
        precision_top50_blending_plus,
    ]

    ganleaks_results = precision_top1_ganleaks, precision_top50_ganleaks
    mcmembership_results = precision_top1_mcmembership, precision_top50_mcmembership

    draw.compare_results(
        names=names,
        precision_top1_results=precision_top1_results,
        precision_top50_results=precision_top50_results,
        ganleaks_results=ganleaks_results,
        mcmembership_results=mcmembership_results,
        save_path=config.OUTPUT_PATH / generator / "plot" / "results_comparison.jpg",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run membership inference attack")
    parser.add_argument(
        "--generator",
        default="findiff",
        type=str,
        help="Synthetic data generator",
    )

    parser.add_argument(
        "--iteration",
        default=10,
        type=int,
        help="Times of repetition for each attack",
    )

    args = parser.parse_args()
    main(generator=args.generator, iteration=args.iteration)
