# Standard library
import pickle
from typing import List

import warnings

warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # Suppress warnings for DataSynthesizer

# 3rd party packages
import numpy as np
import pandas as pd
from pathlib import Path
from optuna import samplers

# Local
from generators.synthpop_generator import SynthpopGenerator
from generators.smote import SmoteGenerator
from generators.dataSynthesizer import DataSynthesizerGenerator
from generators.mst_generator import MSTGenerator
from generators.ctgan_generator import CTGANGenerator
from generators.tvae_generator import TVAEGenerator
from generators.ctabgan_generator import CTABGANGenerator
from generators.tabddpm_generator import TabDDPMGenerator
from optimization.discrete_pso_search import DiscreteParticleSwarmOptimizationSearch
from optimization.optuna_search import OptunaSearch
from optimization.objective_function import (
    distinguishability_hinge_loss,
    ratio_match_loss,
)


def compound_loss(
    df: dict[str, pd.DataFrame],
    df_to_compare: dict[str, pd.DataFrame],
    metadata: dict,
    use_gpu: bool = False,
) -> float:
    """
    Compound loss that incorporates both utility and privacy

    :param df: the real dataset, split into **train** and **test** sets
    :param df_to_compare: the synthetic dataset, split into **train** and **test** sets
    :param metadata: a dict containing the metadata with the following keys:
      **continuous**, **categorical** and **variable_to_predict**
    :param use_gpu: flag to use GPU computation power to accelerate the learning
    :return: the cost
    """

    utility_loss = distinguishability_hinge_loss(
        df=df, df_to_compare=df_to_compare, metadata=metadata, use_gpu=use_gpu
    )

    privacy_loss = ratio_match_loss(
        df=df, df_to_compare=df_to_compare, metadata=metadata
    )

    loss = utility_loss + privacy_loss

    return loss


def generate_synth_data(
    df_real: pd.DataFrame,
    metadata: dict,
    generator: str,
    synth_size: List[int],
    output_path: Path,
    save_folder: "str",
    seed: int,
) -> List[pd.DataFrame]:
    """Generate synthetic data

    :param df_real: the real data
    :param metadata: a dict containing the metadata with the following keys:
          **continuous**, **categorical** and **variable_to_predict**
    :param generator: the name of the generator to be applied
    :param synth_size: the size(s) of the synthetic data to be generated
    :param output_path: the path in which to save the output
    :param save_folder: the name of the sub-folder to save the result
    :param seed: for reproduction

    :return: the synthetic data
    """
    if seed is not None:
        np.random.seed(seed)

    if generator == "synthpop":
        # Optimize the visiting order with PSO
        optim_order = DiscreteParticleSwarmOptimizationSearch(
            df=df_real,
            metadata=metadata,
            hyperparams={"variables_order": list(df_real.columns)},
            generator=SynthpopGenerator,  # the generator
            objective_function=compound_loss,
            cv_num_folds=0,  # the number of folds for cross-validation (0 or 1 to deactivate)
            use_gpu=True,  # flag to use the gpu if there are available
            direction="min",  # the direction of optimization ("min" or "max")
            num_iter=5,  # the number of iterations to repeat the search
            population_size=3,  # the size of the swarm
        )

        optim_order.fit()
        optimal_vars_order = optim_order.best_params["variables_order"]

        print("Best parameters:")
        print(optim_order.best_params)

        # Save best parameters
        with open(
            output_path
            / generator
            / "generator"
            / save_folder
            / f"{generator}_best_params.pkl",
            "wb",
        ) as file:
            pickle.dump(optim_order.best_params, file)

        # Configure generator with best hyperparameters
        gen = SynthpopGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator,
            variables_order=optimal_vars_order,  # optimal order
            min_samples_leaf=5,  # TODO: also optimize min_samples_leaf
            max_depth=None,
        )

    elif generator == "smote":

        def params_to_explore_optuna(trial):
            params = {
                "k_neighbors": trial.suggest_int(
                    name="k_neighbors", low=3, high=10, step=1, log=False
                ),
            }
            return params

        sampler = samplers.RandomSampler()

        optim_params = OptunaSearch(
            df=df_real,
            metadata=metadata,
            hyperparams=params_to_explore_optuna,
            generator=SmoteGenerator,  # the generator
            objective_function=compound_loss,
            cv_num_folds=1,  # the number of folds for cross-validation (0 or 1 to deactivate)
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=5,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        optim_params.fit()
        optimal_parameters = optim_params.best_params

        print("Best parameters:")
        print(optimal_parameters)

        # Save best parameters
        with open(
            output_path
            / generator
            / "generator"
            / save_folder
            / f"{generator}_best_params.pkl",
            "wb",
        ) as file:
            pickle.dump(optimal_parameters, file)

        gen = SmoteGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator,
            **optimal_parameters,
        )

    elif generator == "datasynthesizer":

        def params_to_explore_optuna(trial):
            params = {
                "degree": trial.suggest_int(
                    name="degree", low=1, high=5, step=1, log=False
                ),
            }
            return params

        sampler = samplers.RandomSampler()

        optim_params = OptunaSearch(
            df=df_real,
            metadata=metadata,
            hyperparams=params_to_explore_optuna,
            generator=DataSynthesizerGenerator,  # the generator
            objective_function=compound_loss,
            cv_num_folds=1,  # the number of folds for cross-validation (0 or 1 to deactivate)
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=3,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        optim_params.fit()
        optimal_parameters = optim_params.best_params

        print("Best parameters:")
        print(optimal_parameters)

        # Save best parameters
        with open(
            output_path
            / generator
            / "generator"
            / save_folder
            / f"{generator}_best_params.pkl",
            "wb",
        ) as file:
            pickle.dump(optimal_parameters, file)

        gen = DataSynthesizerGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator
            candidate_keys=None,  # the identifiers
            epsilon=0,  # for the differential privacy, set to 0 (default=0) to turn DP off
            **optimal_parameters,
        )

    elif generator == "mst":
        # Set the generator to non-private mode
        gen = MSTGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator
            epsilon=1e5,  # the privacy budget of the differential privacy
            delta=0.9999,  # the failure probability of the differential privacy, 1 means non privacy, set to 0.9999 to prevent returning error
        )

    elif generator == "ctgan":

        def params_to_explore_optuna(trial):
            params = {
                "discriminator_steps": trial.suggest_int(
                    name="discriminator_steps", low=2, high=10, step=2, log=False
                ),
                "epochs": trial.suggest_int(
                    name="epochs", low=100, high=500, step=100, log=False
                ),
                "batch_size": trial.suggest_int(
                    name="batch_size", low=50, high=100, step=50, log=False
                ),
            }
            return params

        sampler = samplers.TPESampler(n_startup_trials=10)

        optim_params = OptunaSearch(
            df=df_real,
            metadata=metadata,
            hyperparams=params_to_explore_optuna,
            generator=CTGANGenerator,  # the generator
            objective_function=compound_loss,
            cv_num_folds=1,  # the number of folds for cross-validation (0 or 1 to deactivate)
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=20,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        optim_params.fit()
        optimal_parameters = optim_params.best_params

        print("Best parameters:")
        print(optimal_parameters)

        # Save best parameters
        with open(
            output_path
            / generator
            / "generator"
            / save_folder
            / f"{generator}_best_params.pkl",
            "wb",
        ) as file:
            pickle.dump(optimal_parameters, file)

        gen = CTGANGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator
            epsilon=None,  # Turn DP off, which is the default setting
            delta=None,
            max_grad_norm=1,
            verbose=0,
            **optimal_parameters,
        )

    elif generator == "tvae":

        def params_to_explore_optuna(trial):
            params = {
                "epochs": trial.suggest_int(
                    name="epochs", low=100, high=500, step=100, log=False
                ),
                "batch_size": trial.suggest_int(
                    name="batch_size", low=50, high=200, step=50, log=False
                ),
            }
            return params

        sampler = samplers.TPESampler(n_startup_trials=10)

        optim_params = OptunaSearch(
            df=df_real,
            metadata=metadata,
            hyperparams=params_to_explore_optuna,
            generator=TVAEGenerator,  # the generator
            objective_function=compound_loss,
            cv_num_folds=1,  # the number of folds for cross-validation (0 or 1 to deactivate)
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=20,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        optim_params.fit()
        optimal_parameters = optim_params.best_params

        print("Best parameters:")
        print(optimal_parameters)

        # Save best parameters
        with open(
            output_path
            / generator
            / "generator"
            / save_folder
            / f"{generator}_best_params.pkl",
            "wb",
        ) as file:
            pickle.dump(optimal_parameters, file)

        gen = TVAEGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator
            compress_dims=(249, 249),  # the size of the hidden layers in the encoder
            decompress_dims=(249, 249),  # the size of the hidden layers in the decoder
            epsilon=None,  # Turn DP off
            delta=None,
            max_grad_norm=1,
            max_physical_batch_size=126,
            **optimal_parameters,
        )

    elif generator == "ctabgan":

        def params_to_explore_optuna(trial):
            params = {
                "epochs": trial.suggest_int(
                    name="epochs", low=100, high=500, step=100, log=False
                ),
                "batch_size": trial.suggest_int(
                    name="batch_size", low=50, high=200, step=50, log=False
                ),
            }
            return params

        sampler = samplers.TPESampler(n_startup_trials=10)

        optim_params = OptunaSearch(
            df=df_real,
            metadata=metadata,
            hyperparams=params_to_explore_optuna,
            generator=CTABGANGenerator,  # the generator
            objective_function=compound_loss,
            cv_num_folds=1,  # the number of folds for cross-validation (0 or 1 to deactivate)
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=20,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        optim_params.fit()
        optimal_parameters = optim_params.best_params

        print("Best parameters:")
        print(optimal_parameters)

        # Save best parameters
        with open(
            output_path
            / generator
            / "generator"
            / save_folder
            / f"{generator}_best_params.pkl",
            "wb",
        ) as file:
            pickle.dump(optimal_parameters, file)

        gen = CTABGANGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator
            mixed_columns=None,  # dictionary of "mixed" column names with corresponding categorical modes
            log_columns=None,  # list of skewed exponential numerical columns
            integer_columns=None,  # list of numeric columns without floating numbers
            class_dim=(
                256,
                256,
                256,
                256,
            ),  # size of each desired linear layer for the auxiliary classifier
            random_dim=100,  # dimension of the noise vector fed to the generator
            num_channels=64,  # number of channels in the convolutional layers of both the generator and the discriminator
            l2scale=1e-5,  # rate of weight decay used in the optimizer of the generator, discriminator and auxiliary classifier
            **optimal_parameters,
        )

    elif generator == "tabDDPM":

        def params_to_explore_optuna(trial):
            params = {
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
                ),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                "num_timesteps": trial.suggest_int(
                    name="num_timesteps", low=50, high=150, step=10, log=False
                ),
                "num_iter": trial.suggest_int(
                    name="num_iter", low=100, high=2000, step=100, log=False
                ),
            }
            return params

        sampler = samplers.TPESampler(n_startup_trials=10)

        optim_params = OptunaSearch(
            df=df_real,
            metadata=metadata,
            hyperparams=params_to_explore_optuna,
            generator=TabDDPMGenerator,  # the generator
            objective_function=compound_loss,
            cv_num_folds=1,  # the number of folds for cross-validation (0 or 1 to deactivate)
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=20,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        optim_params.fit()
        optimal_parameters = optim_params.best_params

        print("Best parameters:")
        print(optimal_parameters)

        # Save best parameters
        with open(
            output_path
            / generator
            / "generator"
            / save_folder
            / f"{generator}_best_params.pkl",
            "wb",
        ) as file:
            pickle.dump(optimal_parameters, file)

        gen = TabDDPMGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator
            layers=None,  # the width of the MLP layers
            **optimal_parameters,
        )

    else:
        raise ValueError("Generator does not exist")

    # Fit the generator
    gen.preprocess()
    gen.fit(
        save_path=output_path / generator / "generator" / save_folder
    )  # the path should exist

    # Generate synthetic data
    synth_data = []
    for size in synth_size:
        df_synth = gen.sample(
            save_path=output_path
            / generator
            / "data"
            / save_folder,  # the path should exist
            num_samples=size,
        )

        synth_data.append(df_synth)

    return synth_data
