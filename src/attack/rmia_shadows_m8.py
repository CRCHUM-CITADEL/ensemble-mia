import sys
sys.path.append("/data8/projets/dev_synthetic_data/code/lherbault/github_ensemble_mia")

# Local
from tabddpm_attack_fn import *

import os
import shutil
import pickle
import argparse
import random

# 3rd party
import pandas as pd

json_file_path = '/data8/projets/dev_synthetic_data/code/lherbault/github_ensemble_mia/configs/trans.json'
trans_domain_file_path = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box/train/tabddpm_1/trans_domain.json'
dataset_meta_file_path = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box/train/tabddpm_1/dataset_meta.json'

train_pop_file_path = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box/population/all/real_all.csv'


n_reps = 12 # number of repetitions for each challenge point in the fine-tuning set
n_models = 8 # number of shadow models to train, must be even

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

    # Create the unique lists of challenge points (trans_id)
    unique_ids = set()
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):  # Ensure it's a folder
            csv_file = os.path.join(folder_path, "challenge_with_id.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                dfs.append(df)

    # 1. Merge all dataframes into one master dataframe, removing duplicates
    master_challenge_df = pd.concat(dfs, ignore_index=True).drop_duplicates()

    # 2. Extract unique trans_id values
    unique_ids = master_challenge_df['trans_id'].unique().tolist()

    # create the random lists, each with half the size of unique_ids
    random.shuffle(unique_ids)  # Shuffle to randomize order
    half_models = n_models // 2
    half_size = len(unique_ids) // 2
    lists = [[] for _ in range(n_models)]

    # Assign each unique_id to half of the random lists
    for uid in unique_ids:
        selected_lists = random.sample(range(n_models), half_models)  # Select 2 random list indices
        for idx in selected_lists:
            lists[idx].append(uid)

    attack_data = {'selected_sets': lists,
                   'trained_results': []}

    i=0

    for i, ref_list in enumerate(lists):
        i += 1
        print('Reference model number ', i)

        # Create the necessary folders and config files
        folder_name = 'shadow_model_rmia_' + str(i)
        new_folder = os.path.join(data_dir, folder_name)
        # create the new folder if it doesn't exist
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
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

        selected_challenges = master_challenge_df[master_challenge_df['trans_id'].isin(ref_list)]
        print('Number of selected challenges to train the shadow model: ', len(selected_challenges))
        # Repeat each row n_reps times
        selected_challenges = pd.concat([selected_challenges] * n_reps, ignore_index=True)
        # Shuffle the dataset
        selected_challenges = selected_challenges.sample(frac=1, random_state=42).reset_index(drop=True)

        train_result = train_tabddpm(selected_challenges, configs, save_dir)

        attack_data['trained_results'].append(train_result)

    # Pickle dump the results
    with open(os.path.join(data_dir, 'rmia_shadows_m8.pkl'), 'wb') as file:
        pickle.dump(attack_data, file)





