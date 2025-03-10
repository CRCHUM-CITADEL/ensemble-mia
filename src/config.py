from pathlib import Path

# Path, folder and files
DATA_PATH = Path(
    "../input"
)  # Replace this with the path of the data, i.e., Path("input")
RMIA_PRED_PATH = Path("../input")  # Path where the prediction from RMIA is stored
OUTPUT_PATH = Path(
    "../output"
)  # Replace with the path to store the results (figures and predictions)

# Synthetic data, the files names are the same in train, dev and final (should not be modified)
synth_file = "trans_synthetic.csv"
test_file = "challenge_with_id.csv"
test_label = "challenge_label.csv"  # Only in train
rmia_file = "rmia_scores_k_5.csv"

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
