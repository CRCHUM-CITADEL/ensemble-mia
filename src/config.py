from pathlib import Path

# Path, folder and files
DATA_PATH = Path("/data8/projets/dev_synthetic_data/data/MIDST")
OUTPUT_PATH = Path("/data8/projets/dev_synthetic_data/output/MIDST")

attack_type = "tabddpm_black_box"

# Synthetic data, the files names are the same in train, dev and final
synth_train_file = "synth_train.csv"
synth_test_file = "synth_test.csv"
synth_2nd_file = "synth_2nd.csv"
test_file = "challenge_with_id.csv"
test_label = "challenge_label.csv"  # Only in train

# Train
train_id = list(range(1, 31))

# Dev
dev_id = list(range(51, 61)) + list(range(91, 101))

# Final
final_id = list(range(61, 71)) + list(range(101, 111))


# Metadata for real data
metadata = {
    "continuous": ["trans_date", "amount", "balance", "account"],
    "categorical": ["trans_type", "operation", "k_symbol", "bank"],
    "variable_to_predict": "trans_type",
}

bounds = {
    "trans_type": {"categories": ["0", "1", "2"]},
    "operation": {"categories": ["0", "1", "2", "3", "4", "5"]},
    "k_symbol": {"categories": ["0", "1", "2", "3", "4", "5", "6", "7", "8"]},
    "bank": {
        "categories": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
        ]
    },
}

# Random state for reproducing the results
seed = 42
