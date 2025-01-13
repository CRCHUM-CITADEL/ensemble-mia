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
from clover.generators.synthpop_generator import SynthpopGenerator
from clover.generators.smote import SmoteGenerator
from clover.generators.dataSynthesizer import DataSynthesizerGenerator
from clover.generators.mst_generator import MSTGenerator
from clover.generators.ctgan_generator import CTGANGenerator
from clover.generators.tvae_generator import TVAEGenerator
from clover.generators.ctabgan_generator import CTABGANGenerator
from clover.generators.findiff_generator import FinDiffGenerator
from clover.optimization.discrete_pso_search import (
    DiscreteParticleSwarmOptimizationSearch,
)
from clover.optimization.optuna_search import OptunaSearch
from clover.optimization.objective_function import (
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
            random_state=None,
            use_gpu=True,  # flag to use the gpu if there are available
            direction="min",  # the direction of optimization ("min" or "max")
            num_iter=5,  # the number of iterations to repeat the search
            population_size=3,  # the size of the swarm
        )

        optim_order.fit()
        optimal_vars_order = optim_order.best_params["variables_order"]

        print("Best visiting order:")
        print(optim_order.best_params)

        # Save best parameters
        with open(
            output_path
            / generator
            / "generator"
            / save_folder
            / f"{generator}_best_visiting_order.pkl",
            "wb",
        ) as file:
            pickle.dump(optim_order.best_params, file)

        # Optimize other hyperparameters
        def params_to_explore_optuna(trial):
            params = {
                "min_samples_leaf": trial.suggest_int(
                    name="min_samples_leaf", low=3, high=7, step=1, log=False
                )
            }
            return params

        sampler = samplers.RandomSampler()

        optim_params = OptunaSearch(
            df=df_real[optimal_vars_order],  # Optimal visiting order from PSO
            metadata=metadata,
            hyperparams=params_to_explore_optuna,
            generator=SynthpopGenerator,
            objective_function=compound_loss,
            cv_num_folds=1,  # the number of folds for cross-validation (0 or 1 to deactivate)
            random_state=None,
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

        # Configure generator with best hyperparameters
        gen = SynthpopGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator,
            variables_order=optimal_vars_order,  # optimal order
            epsilon=None,
            min_samples_leaf=optimal_parameters["min_samples_leaf"],
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
            random_state=None,
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
            epsilon=None,
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
            random_state=None,
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=3,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
            epsilon=None,
            preprocess_metadata=None,
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
            preprocess_metadata=None,
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
                "batch_size": trial.suggest_categorical(
                    "batch_size", [64, 128, 256, 512]
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
            random_state=None,
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=20,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
            max_grad_norm=None,
            preprocess_metadata=None,
            verbose=0,
            **optimal_parameters,
        )

    elif generator == "tvae":

        def params_to_explore_optuna(trial):
            params = {
                "epochs": trial.suggest_int(
                    name="epochs", low=100, high=500, step=100, log=False
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [64, 128, 256, 512]
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
            random_state=None,
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=20,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
            max_physical_batch_size=125,
            epsilon=None,  # Turn DP off
            delta=None,
            max_grad_norm=None,
            preprocess_metadata=None,
            **optimal_parameters,
        )

    elif generator == "ctabgan":

        def params_to_explore_optuna(trial):
            params = {
                "epochs": trial.suggest_int(
                    name="epochs", low=100, high=500, step=100, log=False
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [64, 128, 256, 512]
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
            random_state=None,
            use_gpu=True,  # flag to use the gpu if there are available
            sampler=sampler,
            direction="minimize",  # the direction of optimization ("minimize" or "maximize")
            num_iter=20,  # the number of iterations to repeat the search
            verbose=0,  # whether to print the INFO logs (1) or not (0)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
            epsilon=None,
            delta=None,
            max_grad_norm=None,
            preprocess_metadata=None,
            **optimal_parameters,
        )

    elif generator == "findiff":

        def params_to_explore_optuna(trial):
            params = {
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", [1e-3, 1e-4]
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size",
                    [64, 128, 256],
                ),
                "epochs": trial.suggest_int(
                    name="epochs", low=100, high=500, step=100, log=False
                ),
            }
            return params

        sampler = samplers.TPESampler(n_startup_trials=10)

        optim_params = OptunaSearch(
            df=df_real,
            metadata=metadata,
            hyperparams=params_to_explore_optuna,
            generator=FinDiffGenerator,  # the generator
            objective_function=compound_loss,
            cv_num_folds=1,  # the number of folds for cross-validation (0 or 1 to deactivate)
            random_state=None,
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

        gen = FinDiffGenerator(
            df=df_real,
            metadata=metadata,
            random_state=seed,  # for reproducibility, can be set to None
            generator_filepath=None,  # to load an existing generator
            diffusion_steps=500,  # the diffusion timesteps for the forward diffusion process
            mpl_layers=[1024, 1024, 1024, 1024],  # the width of the MLP layers
            activation="lrelu",  # the activation fuction
            dim_t=64,  # dimensionality of the intermediate layer for connecting the embeddings
            cat_emb_dim=2,  # dimension of categorical embeddings
            diff_beta_start_end=[1e-4, 0.02],  # diffusion start and end betas
            scheduler="linear",  # diffusion scheduler
            epsilon=None,
            delta=None,
            max_grad_norm=None,
            preprocess_metadata=None,
            **optimal_parameters,
        )

    else:
        raise ValueError("Generator does not exist")

    # Fit the generator
    gen.preprocess()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
