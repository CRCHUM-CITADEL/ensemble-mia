import sys
sys.path.append("/data8/projets/dev_synthetic_data/code/lherbault/github_ensemble_mia")

# Local
from distinguishability_attack_fn import *

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

    # Create the unique lists of challenge points (trans_id)
    unique_ids = set()
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):  # Ensure it's a folder
            csv_file = os.path.join(folder_path, "challenge_with_id.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                dfs.append(df)

    # for file in [os.path.join(data_dir, "train_meta.csv"), os.path.join(data_dir, "test_meta.csv")]: # To delete
    #     if os.path.exists(file):  # To delete
    #         df = pd.read_csv(file)  # To delete
    #         dfs.append(df)  # To delete

    # 1. Merge all dataframes into one master dataframe, removing duplicates
    master_challenge_df = pd.concat(dfs, ignore_index=True).drop_duplicates()

    # 2. Extract unique trans_id values
    unique_ids = master_challenge_df['trans_id'].unique().tolist()

    # Train the initial model without any challenge point
    train_pop = train_original[~train_original['trans_id'].isin(unique_ids)]
    print('Length of potential train population excluding all ids in challenge sets: ', len(train_pop))

    # Create the necessary folders and config files
    new_folder = os.path.join(data_dir, "initial_model_rmia_2")
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

    # create the initial training set
    train = train_pop.sample(n=60000)
    train.to_csv(os.path.join(new_folder, 'initial_train_set.csv'))

    # train the initial model
    initial_model = train_tabddpm(train, configs, save_dir)
    # Pickle dump the results
    with open(os.path.join(data_dir, 'rmia_initial_model_2.pkl'), 'wb') as file:
        pickle.dump(initial_model, file)

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

    attack_data = {'fine_tuning_sets': lists,
                   'fine_tuned_results': []}

    i=0

    for ref_list in lists:
        i+=1
        print('Reference model number ', i)
        selected_challenges = master_challenge_df[master_challenge_df['trans_id'].isin(ref_list)]
        # Repeat each row n_reps times
        selected_challenges = pd.concat([selected_challenges] * n_reps, ignore_index=True)
        # Shuffle the dataset
        selected_challenges = selected_challenges.sample(frac=1, random_state=42).reset_index(drop=True)

        train_result = fine_tune_tabddpm(trained_models=initial_model['models'],
                                         new_train_set=selected_challenges,
                                         configs=configs,
                                         save_dir=save_dir,
                                         new_diffusion_iterations=200000,
                                         new_classifier_iterations=20000,
                                         n_synth=20000,
                                         )

        attack_data['fine_tuned_results'].append(train_result)

    # Pickle dump the results
    with open(os.path.join(data_dir, 'rmia_shadows_2.pkl'), 'wb') as file:
        pickle.dump(attack_data, file)





