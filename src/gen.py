# Standard library
import argparse
import json
import shutil
import warnings
from pathlib import Path

# 3rd party
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Local
import config
from utils import standard

from externals.MIDSTModels.midst_models.single_table_TabDDPM.complex_pipeline import (
    clava_clustering,
    clava_synthesizing,
    clava_training,
    load_configs,
)
from externals.MIDSTModels.midst_models.single_table_TabDDPM.pipeline_modules import (
    load_multi_table,
)
from externals.MIDSTModels.midst_models.single_table_TabSyn.scripts.process_dataset import (
    process_data,
)
from externals.MIDSTModels.midst_models.single_table_TabSyn.src import load_config
from externals.MIDSTModels.midst_models.single_table_TabSyn.src.data import (
    TabularDataset,
    preprocess,
)
from externals.MIDSTModels.midst_models.single_table_TabSyn.src.tabsyn.pipeline import (
    TabSyn,
)


def main(generator: str, dataset: str, ref_data_path: str) -> None:
    # Paths
    root_dir = config.DATA_PATH / f"{generator}_black_box" / dataset

    if generator == "tabddpm":
        config_file_path = (
            Path("externals")
            / "MIDSTModels"
            / "midst_models"
            / "single_table_TabDDPM"
            / "configs"
        )
    else:  # tabsyn
        data_config_file_path = (
            Path("externals")
            / "MIDSTModels"
            / "midst_models"
            / "single_table_TabSyn"
            / "data_info"
        )

        model_config_file_path = (
            Path("externals")
            / "MIDSTModels"
            / "midst_models"
            / "single_table_TabSyn"
            / "src"
            / "configs"
        )

    # Load data id for different datasets
    if dataset == "train":
        data_id = config.train_id
    elif dataset == "dev":
        data_id = config.dev_id
    else:  # final
        data_id = config.final_id

    for id_ in data_id:
        ##################################################
        # Prepare data
        ##################################################
        print("-----------------------------")
        print(f"Training for {generator}_{id_}")
        print("-----------------------------")

        input_dir = root_dir / f"{generator}_{id_}"
        output_dir = input_dir / "2nd_gen"

        # Create the new folder if it doesn't exist
        standard.create_directory(output_dir)

        df_synth = pd.read_csv(input_dir / "trans_synthetic.csv")

        # Type convertion
        col_type = {
            "float": ["amount", "balance"],
            "int": [
                "trans_date",
                "account",
                "trans_type",
                "operation",
                "k_symbol",
                "bank",
            ],
        }

        df_synth = standard.trans_type(df=df_synth, col_type=col_type, decimal=1)

        # Split data into train and test
        df_synth_train, df_synth_test = train_test_split(
            df_synth,
            test_size=len(df_synth) // 3,
            random_state=config.seed,
            stratify=df_synth["trans_type"],
        )

        # Save data
        df_synth_train.to_csv(input_dir / "synth_train.csv", index=False)
        df_synth_test.to_csv(input_dir / "synth_test.csv", index=False)

        if generator == "tabddpm":
            ##################################################
            # Train model
            ##################################################

            # Save the training data
            df_synth_train.to_csv(output_dir / "train.csv", index=False)

            # Copy the original config file to the new folder
            shutil.copy(config_file_path / "trans.json", output_dir)
            shutil.copy(config_file_path / "dataset_meta.json", output_dir)
            shutil.copy(config_file_path / "trans_domain.json", output_dir)

            # Modify the config file to give the correct training data and saving directory
            with open(output_dir / "trans.json", "r") as file:
                trans_config = json.load(file)

            trans_config["general"]["data_dir"] = str(output_dir)
            trans_config["general"]["workspace_dir"] = str(output_dir)
            trans_config["general"]["test_data_dir"] = str(input_dir)

            # Save the changed to the new json file
            with open(output_dir / "trans.json", "w") as file:
                json.dump(trans_config, file, indent=4)

            # Set up the config
            configs, save_dir = load_configs(output_dir / "trans.json")

            # Load tables
            tables, relation_order, dataset_meta = load_multi_table(
                configs["general"]["data_dir"]
            )

            # Clustering on the multi-table dataset
            tables, all_group_lengths_prob_dicts = clava_clustering(
                tables, relation_order, save_dir, configs
            )

            # Train models
            models = clava_training(tables, relation_order, save_dir, configs)

            ##################################################
            # Generate synthetic data
            ##################################################

            # Generate synthetic data from scratch
            (
                cleaned_tables,
                synthesizing_time_spent,
                matching_time_spent,
            ) = clava_synthesizing(
                tables,
                relation_order,
                save_dir,
                all_group_lengths_prob_dicts,
                models,
                configs,
                sample_scale=1,
            )

            df_synth_2nd = standard.trans_type(
                df=cleaned_tables["trans"], col_type=col_type, decimal=1
            )
            df_synth_2nd.to_csv(input_dir / "synth_2nd.csv", index=False)

        else:  # tabsyn
            ##################################################
            # Preprocess training data
            ##################################################

            data_name = "trans"

            info_dir = output_dir / "data_info"
            raw_data_dir = output_dir / "raw_data"
            processed_data_dir = output_dir / "processed_data"
            synth_data_dir = output_dir / "synthetic_data"
            model_path = output_dir / "models " / "tabsyn"

            # Create the new folder if it doesn't exist
            for dir_ in [
                info_dir,
                raw_data_dir,
                processed_data_dir,
                synth_data_dir,
                model_path,
            ]:
                standard.create_directory(dir_)

            # Save the training data to data folder
            df_synth_train.to_csv(raw_data_dir / "train.csv", index=False)

            # Copy the original config file to the new folder
            shutil.copy(data_config_file_path / "trans.json", info_dir)

            # Modify the config file
            with open(info_dir / "trans.json", "r") as file:
                trans_config = json.load(file)

            trans_config["name"] = data_name
            trans_config["data_path"] = str(raw_data_dir / "train.csv")
            trans_config["test_path"] = ""

            # Rewrite the config file
            with open(info_dir / "trans.json", "w") as file:
                json.dump(trans_config, file, indent=4)

            # Process training data
            process_data(data_name, info_dir, output_dir)

            # Load model config
            raw_config = load_config(model_config_file_path / f"{data_name}.toml")

            # Preprocess data
            X_num, X_cat, categories, d_numerical = preprocess(
                processed_data_dir / data_name,
                ref_dataset_path=ref_data_path,
                transforms=raw_config["transforms"],
                task_type=raw_config["task_type"],
            )

            # Separate train and test data
            X_train_num, X_test_num = X_num
            X_train_cat, X_test_cat = X_cat

            # Convert to float tensor
            X_train_num, X_test_num = (
                torch.tensor(X_train_num).float(),
                torch.tensor(X_test_num).float(),
            )
            X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(
                X_test_cat
            )

            # Create dataset object with train data and return tokens of a single row at a time
            train_data = TabularDataset(X_train_num.float(), X_train_cat)

            # Move test data to gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_test_num = X_test_num.float().to(device)
            X_test_cat = X_test_cat.to(device)

            # Create train dataloader
            train_loader = DataLoader(
                train_data,
                batch_size=raw_config["train"]["vae"]["batch_size"],
                shuffle=True,
                num_workers=raw_config["train"]["vae"]["num_dataset_workers"],
            )
            ##################################################
            # Train model
            ##################################################

            # Instantiate model
            tabsyn = TabSyn(
                train_loader,
                X_test_num,
                X_test_cat,
                num_numerical_features=d_numerical,
                num_classes=categories,
                device=device,
            )

            # Instantiate and train VAE model
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tabsyn.instantiate_vae(
                    **raw_config["model_params"],
                    optim_params=raw_config["train"]["optim"]["vae"],
                )

            standard.create_directory(model_path / data_name / "vae")

            tabsyn.train_vae(
                **raw_config["loss_params"],
                num_epochs=raw_config["train"]["vae"]["num_epochs"],
                save_path=model_path / data_name / "vae",
            )

            # Embed all inputs in the latent space
            tabsyn.save_vae_embeddings(
                X_train_num, X_train_cat, vae_ckpt_dir=model_path / data_name / "vae"
            )

            # Load latent space embeddings
            train_z, token_dim = tabsyn.load_latent_embeddings(
                model_path / data_name / "vae"
            )  # train_z dim: B x in_dim

            # Normalize embeddings
            mean = train_z.mean(0)
            latent_train_data = (train_z - mean) / 2

            # Create data loader
            latent_train_loader = DataLoader(
                latent_train_data,
                batch_size=raw_config["train"]["diffusion"]["batch_size"],
                shuffle=True,
                num_workers=raw_config["train"]["diffusion"]["num_dataset_workers"],
            )

            # Instantiate and train diffusion model
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tabsyn.instantiate_diffusion(
                    in_dim=train_z.shape[1],
                    hid_dim=train_z.shape[1],
                    optim_params=raw_config["train"]["optim"]["diffusion"],
                )

            tabsyn.train_diffusion(
                latent_train_loader,
                num_epochs=raw_config["train"]["diffusion"]["num_epochs"],
                ckpt_path=model_path / data_name,
            )
            ##################################################
            # Generate synthetic data
            ##################################################

            # Load data info file
            with open(processed_data_dir / data_name / "info.json", "r") as file:
                data_info = json.load(file)

            data_info["token_dim"] = token_dim

            # Get inverse tokenizers
            _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
                processed_data_dir / data_name,
                ref_dataset_path=ref_data_path,
                transforms=raw_config["transforms"],
                task_type=raw_config["task_type"],
                inverse=True,
            )

            standard.create_directory(synth_data_dir / data_name)

            # Sample data
            num_samples = len(df_synth_train)
            in_dim = train_z.shape[1]
            mean_input_emb = train_z.mean(0)
            tabsyn.sample(
                num_samples,
                in_dim,
                mean_input_emb,
                info=data_info,
                num_inverse=num_inverse,
                cat_inverse=cat_inverse,
                save_path=synth_data_dir / data_name / "tabsyn.csv",
            )

            df_synth_2nd = pd.read_csv(synth_data_dir / data_name / "tabsyn.csv")
            df_synth_2nd = standard.trans_type(
                df=df_synth_2nd, col_type=col_type, decimal=1
            )
            df_synth_2nd.to_csv(input_dir / "synth_2nd.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate second generation synthetic data"
    )

    parser.add_argument(
        "--generator",
        default=None,
        type=str,
        help="Name of the generator: tabddpm or tabsyn",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Name of dataset: train, dev or final",
    )

    parser.add_argument(
        "--ref_data_path",
        default=None,
        type=str,
        help="Output folder to store the results from preprocessing all the real data ",
    )

    args = parser.parse_args()
    main(
        generator=args.generator, dataset=args.dataset, ref_data_path=args.ref_data_path
    )
