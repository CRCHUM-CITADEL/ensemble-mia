import os
import sys

sys.path.append("..")
sys.path.append(".")

# 3rd party
import pandas as pd
import numpy as np
import copy
import json

# Local
from src.externals.MIDSTModels.midst_models.single_table_TabDDPM.complex_pipeline import (
    clava_clustering,
    clava_training,
    clava_fine_tuning,
    clava_synthesizing,
    load_configs,
)
from src.externals.MIDSTModels.midst_models.single_table_TabDDPM.pipeline_modules import (
    load_multi_table,
)


def config_tabddpm(
    data_dir="/data8/projets/dev_synthetic_data/data/MIDST/tabddpm_black_box/train/tabddpm_1_attack/temp_dir",
    json_path=None,
    final_json_path=None,
    diffusion_layers=[32, 64, 64, 64, 64, 32],
    diffusion_iterations=10000,
    classifier_layers=[16, 32, 64, 128, 64, 32, 16],
    classifier_dim_t=16,
    classifier_iterations=1000,
):
    """
    :param data_dir: The folder in which can be found the files dataset_meta.json, trans_domain.json and
    trans.json
    :return: configs, save_dir objects necessary for the tabddpm code to run and save
    """
    # modify the config file to give the correct training data and saving directory
    if json_path is not None:
        temp_json_file_path = json_path
    else:
        temp_json_file_path = os.path.join(data_dir, "trans.json")
    with open(temp_json_file_path, "r") as file:
        data = json.load(file)
    data["general"]["data_dir"] = data_dir
    data["general"]["exp_name"] = "tmp"
    data["general"]["workspace_dir"] = os.path.join(data_dir, "tmp_workspace")

    # modify the model parameters for smaller sets
    data["diffusion"]["d_layers"] = diffusion_layers
    data["diffusion"]["iterations"] = diffusion_iterations
    data["classifier"]["d_layers"] = classifier_layers
    data["classifier"]["dim_t"] = classifier_dim_t
    data["classifier"]["iterations"] = classifier_iterations

    # save the changed to the new json file
    if final_json_path is not None:
        final_json_file_path = final_json_path
    else:
        final_json_file_path = temp_json_file_path
    with open(final_json_file_path, "w") as file:
        json.dump(data, file, indent=4)

    print("Changes made successfully in path ", final_json_file_path)

    # Set up the config
    configs, save_dir = load_configs(final_json_file_path)

    return configs, save_dir


def train_tabddpm(
    train_set,
    configs,
    save_dir,
):
    material = {
        "tables": {},
        "relation_order": {},
        "save_dir": save_dir,
        "all_group_lengths_prob_dicts": {},
        "models": {},
        "configs": configs,
        "synth_data": {},
    }

    # Load tables
    tables, relation_order, dataset_meta = load_multi_table(
        configs["general"]["data_dir"], train_df=train_set
    )
    material["relation_order"] = relation_order

    # Clustering on the multi-table dataset
    tables, all_group_lengths_prob_dicts = clava_clustering(
        tables, relation_order, save_dir, configs
    )
    material["tables"] = tables
    material["all_group_lengths_prob_dicts"] = all_group_lengths_prob_dicts

    # Train models
    models = clava_training(tables, relation_order, save_dir, configs)
    material["models"] = models

    # Determine the sample scale
    # We want the final synthetic data = len(provided_synth_data) = 20,000
    sample_scale = 20000 / len(tables["trans"]["df"])

    # Generate synthetic data from scratch
    cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
        tables,
        relation_order,
        save_dir,
        all_group_lengths_prob_dicts,
        models,
        configs,
        sample_scale=sample_scale,
    )

    material["synth_data"] = cleaned_tables["trans"]

    return material


def fine_tune_tabddpm(
    trained_models,
    new_train_set,
    configs,
    save_dir,
    new_diffusion_iterations=100,
    new_classifier_iterations=10,
    n_synth=20000,
):
    material = {
        "tables": {},
        "relation_order": {},
        "save_dir": save_dir,
        "all_group_lengths_prob_dicts": {},
        "models": {},
        "configs": configs,
        "synth_data": {},
    }

    # Load tables
    new_tables, relation_order, dataset_meta = load_multi_table(
        configs["general"]["data_dir"], train_df=new_train_set
    )
    material["relation_order"] = relation_order

    # Clustering on the multi-table dataset
    new_tables, all_group_lengths_prob_dicts = clava_clustering(
        new_tables, relation_order, save_dir, configs, force_tables=True
    )
    material["tables"] = new_tables
    material["all_group_lengths_prob_dicts"] = all_group_lengths_prob_dicts

    # Train models
    copied_models = copy.deepcopy(trained_models)
    new_models = clava_fine_tuning(
        copied_models,
        new_tables,
        relation_order,
        save_dir,
        configs,
        new_diffusion_iterations,
        new_classifier_iterations,
    )
    material["new_models"] = new_models

    # Determine the sample scale
    # We want the final synthetic data = len(provided_synth_data) = 20,000
    sample_scale = n_synth / len(new_tables["trans"]["df"])

    # Generate synthetic data from scratch
    cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
        new_tables,
        relation_order,
        save_dir,
        all_group_lengths_prob_dicts,
        new_models,
        configs,
        sample_scale=sample_scale,
    )

    material["synth_data"] = cleaned_tables["trans"]

    return material
