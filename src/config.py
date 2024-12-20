# 3rd party packages
from pathlib import Path

# Path configurations
PROJECT_PATH = Path("..")
DATA_PATH = PROJECT_PATH / "../data/"  # Adult
DATASET_FILEPATH = DATA_PATH / "adult_train.csv"  # Adult
# DATA_PATH = (
#     PROJECT_PATH / "../../../../../ChasseM_MIMICIII/code/yqi/git_ChasseM_MIMICIII/"
# )  # MIMIC-III
# DATASET_FILEPATH = DATA_PATH / "output" / "harutyunyan2017multitask_nona.csv"  # MIMIC-III
OUTPUT_PATH = PROJECT_PATH / "output"

# Random state for reproducing the results
seed = 42

# Meta data for real data
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
