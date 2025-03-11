import os

import numpy as np
import pandas as pd

import src.utils.external.gower.gower_dist as gower

# Compute the gower distance matrix and membership signals wrt the provided synthetic dataset

k = 5  # number of neighbors to consider in the membership signal computation
base_path = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box'

metadata = {
    "continuous": ["trans_date", "amount", "balance", "bank"],
    "categorical": ["trans_type", "operation", "k_symbol", "account"],
    "variable_to_predict": "trans_type",
}

all_columns = ['trans_date', 'trans_type', 'operation', 'amount', 'balance', 'k_symbol', 'bank', 'account']
# Define categorical features (boolean array instead of column names)
cat_features = [True if col in metadata["categorical"] else False for col in all_columns]

challenge_file = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box/train/metaclassifier_target_model_rmia/master_challenge.csv'
ds_file = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box/train/metaclassifier_target_model_rmia/synthetic_data.csv'
challenge_df = pd.read_csv(challenge_file)
ds_df = pd.read_csv(ds_file)

# Convert numerical columns to float (otherwise error in the numpy divide)
challenge_df[metadata['continuous']] = challenge_df[metadata['continuous']].astype(float)
ds_df[metadata['continuous']] = ds_df[metadata['continuous']].astype(float)

# Compute the Gower distance
pairwise_gower = gower.gower_matrix(
    data_x=challenge_df[all_columns], data_y=ds_df[all_columns], cat_features=cat_features
)
if np.any((pairwise_gower < 0) | (pairwise_gower > 1)):
    print('Distances are falling outside of range [0, 1].')

# Save array
folder_path = '/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box/train/metaclassifier_target_model_rmia'
np.savetxt(os.path.join(folder_path, 'gower_matrix.csv'), pairwise_gower, delimiter=",")

# Keep only the k smallest distances for each challenge point (first column is 0 for the within real/synth)
dist = np.sort(pairwise_gower, axis=1)[:, 0: k]
mean_dist = np.mean(dist, axis=1)

# Save array
file_name = 'mean_gower_distance_top_' + str(k) + '.csv'
np.savetxt(os.path.join(folder_path, file_name), mean_dist, delimiter=",")