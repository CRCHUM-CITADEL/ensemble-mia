import sys
sys.path.append("/data8/projets/dev_synthetic_data/code/lherbault/github_ensemble_mia")

# Local
from distinguishability_attack_fn import *

import os
import shutil
import pickle
import argparse
import numpy as np
import random

# 3rd party
import pandas as pd

json_file_path = '/data8/projets/dev_synthetic_data/code/lherbault/github_ensemble_mia/configs/trans.json'
trans_domain_file_path = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box/train/tabddpm_1/trans_domain.json'
dataset_meta_file_path = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box/train/tabddpm_1/dataset_meta.json'

train_pop_file_path = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box/population/all/real_all.csv'


n_reps = 12 # number of repetitions for each challenge point in the fine-tuning set
n_models = 4 # number of shadow models to train, must be even

metadata = {
    "continuous": ["trans_date", "amount", "balance", "bank"],
    "categorical": ["trans_type", "operation", "k_symbol", "account"],
    "variable_to_predict": "trans_type",
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Attack parser.")
    parser.add_argument("challenge_points_repo", type=str, help="Path to the repo of all challenge csvs.")

    args = parser.parse_args()

    data_dir = args.challenge_points_repo
    train_original = pd.read_csv(train_pop_file_path)

    # Initialize an empty list to store dataframes
    dfs = []
    dfs_1 = []

    # Create the unique lists of challenge points (trans_id)
    unique_ids = set()
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):  # Ensure it's a folder
            csv_file = os.path.join(folder_path, "challenge_with_id.csv")
            label_file = os.path.join(folder_path, "challenge_label.csv")
            if os.path.exists(csv_file) and os.path.exists(label_file):
                df = pd.read_csv(csv_file)
                label = pd.read_csv(label_file)
                dfs.append(df)
                dfs_1.append(df[label['is_train']==1])

    # 1. Merge all dataframes into one master dataframe, removing duplicates
    master_challenge_df = pd.concat(dfs, ignore_index=True).drop_duplicates().dropna(subset=['trans_id'])
    master_challenge_df_1 = pd.concat(dfs_1, ignore_index=True).drop_duplicates().dropna(subset=['trans_id'])

    # 2. Extract unique trans_id values
    unique_ids = master_challenge_df['trans_id'].unique().tolist()
    unique_ids_1 = master_challenge_df_1['trans_id'].unique().tolist()

    # Train the initial model without any challenge point
    train_pop = train_original[~train_original['trans_id'].isin(unique_ids)]
    print('Length of potential train population excluding all ids in challenge sets: ', len(train_pop))

    # create the initial training set
    train_out = train_pop.sample(n=20000-len(master_challenge_df_1))
    train = pd.concat([master_challenge_df_1, train_out])
    train = train.sample(frac=1).reset_index(drop=True)

    # Create the necessary folders and config files
    new_folder = os.path.join(data_dir, "metaclassifier_target_model_rmia")
    # create the new folder if it doesn't exist
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    # store files
    train.to_csv(os.path.join(new_folder, 'initial_train_set.csv'))
    master_challenge_df.to_csv(os.path.join(new_folder, 'master_challenge.csv'))
    master_challenge_labels = np.where(master_challenge_df['trans_id'].isin(unique_ids_1), 1, 0)
    master_challenge_labels = pd.DataFrame(master_challenge_labels, columns=['is_train'])
    master_challenge_labels.to_csv(os.path.join(new_folder, 'master_challenge_labels.csv'))
    print(f"Length of master_challenge_df: {len(master_challenge_df)}")
    print(f"Length of master_challenge_labels: {len(master_challenge_labels)}")
    # set config
    shutil.copyfile(trans_domain_file_path, os.path.join(new_folder, 'trans_domain.json'))
    shutil.copyfile(dataset_meta_file_path, os.path.join(new_folder, 'dataset_meta.json'))
    configs, save_dir = config_tabddpm(
        data_dir=new_folder,
        json_path=json_file_path,
        final_json_path=os.path.join(data_dir, 'trans.json'),
        diffusion_layers=[
            512,
            1024,
            1024,
            1024,
            1024,
            512
        ],
        diffusion_iterations=200000,
        classifier_layers=[
            128,
            256,
            512,
            1024,
            512,
            256,
            128
        ],
        classifier_dim_t=128,
        classifier_iterations=20000,
    )

    # train the initial model
    initial_model = train_tabddpm(train, configs, save_dir)

    # Pickle dump the results
    with open(os.path.join(new_folder, 'rmia_target_model.pkl'), 'wb') as file:
        pickle.dump(initial_model, file)

    initial_model['synth_data'].to_csv(os.path.join(new_folder, 'synthetic_data.csv'))


