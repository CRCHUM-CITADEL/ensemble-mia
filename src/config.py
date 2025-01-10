from pathlib import Path

# Path configurations
PROJECT_PATH = Path("..")
DATA_PATH = PROJECT_PATH / "data"
DATASET_FILEPATH = DATA_PATH / "adult_train.csv"  # Adult
OUTPUT_PATH = PROJECT_PATH / "output"

# Random state for reproducing the results
seed = 42

# Metadata for real data
metadata = {
    "continuous": [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ],
    "categorical": [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "salary",
    ],
    "variable_to_predict": "salary",
}
