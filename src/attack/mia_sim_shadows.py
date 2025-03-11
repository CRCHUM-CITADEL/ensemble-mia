import os
import pickle

import numpy as np
import pandas as pd

import src.utils.external.gower.gower_dist as gower

# For all challenge sets, compute the gower distance matrix and membership signals wrt the provided synthetic dataset

k = 1  # number of neighbors to consider in the membership signal computation
base_path = "/data8/projets/dev_synthetic_data/data/MIDST_open/tabddpm_black_box"
repo_list = ["train", "dev", "final"]

metadata = {
    "continuous": ["trans_date", "amount", "balance", "bank"],
    "categorical": ["trans_type", "operation", "k_symbol", "account"],
    "variable_to_predict": "trans_type",
}

all_columns = [
    "trans_date",
    "trans_type",
    "operation",
    "amount",
    "balance",
    "k_symbol",
    "bank",
    "account",
]
# Define categorical features (boolean array instead of column names)
cat_features = [
    True if col in metadata["categorical"] else False for col in all_columns
]

if __name__ == "__main__":
    for repo in repo_list:
        repo_path = os.path.join(base_path, repo)
        with open(os.path.join(repo_path, "rmia_shadows_2.pkl"), "rb") as file:
            attack_data = pickle.load(file)
            for ref_dict in attack_data["fine_tuned_results"]:
                # Convert numerical columns to float (otherwise error in the numpy divide)
                ref_dict["synth_data"][metadata["continuous"]] = ref_dict["synth_data"][
                    metadata["continuous"]
                ].astype(float)

        for folder in os.listdir(repo_path):
            print(folder)
            folder_path = os.path.join(repo_path, folder)
            if (
                os.path.isdir(folder_path) and "tabddpm" in folder
            ):  # Ensure it's a folder
                challenge_file = os.path.join(folder_path, "challenge_with_id.csv")
                challenge_df = pd.read_csv(challenge_file)
                gower_matrix_shadow = []
                mean_dist_shadow = []
                # Convert numerical columns to float (otherwise error in the numpy divide)
                challenge_df[metadata["continuous"]] = challenge_df[
                    metadata["continuous"]
                ].astype(float)

                for ref_dict in attack_data["fine_tuned_results"]:
                    ds_df = ref_dict["synth_data"]
                    ds_df = ds_df.sample(n=20000, random_state=42)

                    # Compute the Gower distance
                    pairwise_gower = gower.gower_matrix(
                        data_x=challenge_df[all_columns],
                        data_y=ds_df[all_columns],
                        cat_features=cat_features,
                    )
                    if np.any((pairwise_gower < 0) | (pairwise_gower > 1)):
                        print("Distances are falling outside of range [0, 1].")

                    gower_matrix_shadow.append(pairwise_gower)

                    # Keep only the k smallest distances for each challenge point (first column is 0 for the within real/synth)
                    #dist = np.sort(pairwise_gower, axis=1)[:, 0:k]
                    #mean_dist = np.mean(dist, axis=1)
                    mean_dist = np.min(pairwise_gower, axis=1)

                    mean_dist_shadow.append(mean_dist)

                # Save array
                file_name = 'gower_matrix_shadow_2_top_' + str(k) + '.pkl'
                with open(
                    os.path.join(folder_path, file_name), "wb"
                ) as file:
                    pickle.dump(
                        {
                            "gower_matrix": gower_matrix_shadow,
                            "mean_dist": mean_dist_shadow,
                        },
                        file,
                    )
