# Standard library
import argparse
import json
import shutil
from pathlib import Path

# 3rd party
import pandas as pd
from sklearn.model_selection import train_test_split

# Local
import config
from utils import standard

from external.MIDSTModels.midst_models.single_table_TabDDPM.complex_pipeline import (
    clava_clustering,
    clava_synthesizing,
    clava_training,
    load_configs,
)
from external.MIDSTModels.midst_models.single_table_TabDDPM.pipeline_modules import (
    load_multi_table,
)


def main(
    generator: str,
    dataset: str,
) -> None:
    # Paths
    root_dir = config.DATA_PATH / f"{generator}_black_box" / dataset

    if generator == "tabddpm":
        gen_name = "TabDDPM"
    else:
        gen_name = "TabSyn"

    config_file_path = (
        Path("external")
        / "MIDSTModels"
        / "midst_models"
        / f"single_table_{gen_name}"
        / "configs"
    )

    # Load data for each model
    if dataset == "train":
        data_id = config.train_id
    elif dataset == "dev":
        data_id = config.dev_id
    else:
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

        # create the new folder if it doesn't exist
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

        ##################################################
        # Train model
        ##################################################

        # save the training data
        df_synth_train.to_csv(output_dir / "train.csv", index=False)

        # copy the original config file to the new folder
        shutil.copy(config_file_path / "trans.json", output_dir)
        shutil.copy(config_file_path / "dataset_meta.json", output_dir)
        shutil.copy(config_file_path / "trans_domain.json", output_dir)

        # Modify the config file to give the correct training data and saving directory
        with open(output_dir / "trans.json", "r") as file:
            trans_config = json.load(file)

        trans_config["general"]["data_dir"] = str(output_dir)
        trans_config["general"]["workspace_dir"] = str(output_dir)
        trans_config["general"]["test_data_dir"] = str(input_dir)

        # save the changed to the new json file
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

    args = parser.parse_args()
    main(
        generator=args.generator,
        dataset=args.dataset,
    )
