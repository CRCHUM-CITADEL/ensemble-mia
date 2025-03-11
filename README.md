# Ensemble membership inference attack model

![Training Pipeline.jpg](Training%20Pipeline.jpg)

## Overview

**Description:** This GitHub repository includes Python scripts designed to perform membership inference attacks 
using ensemble technique.

The meta-classifier takes the following inputs:
* All continuous features of the data (i.e., train or test data)
* The minimum Gower distance between each data point and the synthetic dataset
* The mean Gower distance of each data point to the 5, 10, 20, 30, 40 and 50 nearest records in the synthetic dataset
* The nearest neighbor distance ratio (NNDR) of each data point for its neighbors in the synthetic dataset
* The number of neighbors in the synthetic data for each data point 
* The membership prediction from [DOMIAS](https://arxiv.org/abs/2302.12580) model
* The membership prediction from [RMIA](https://arxiv.org/abs/2502.14921) model

The meta-classifier outputs the probability that each data point is member of the dataset used to generate 
the synthetic data.

Note: the RMIA attack requires the separate training of shadow models. 
More details are available in the RMIA notebook and scripts.

## Usage

1.Edit the `config.py` file (this is optional and the only configurations need to be modified are 
**DATA_PATH**, **RMIA_PRED_PATH** and **OUTPUT_PATH**).

2.**Collect** the population data and save it as **.csv** file, i.e., in the intput folder 
(refer to `1.real_data_processing.ipynb` for details). This step is required for the DOMIAS model.
Note: we have already collected the population data and saved it in the intput folder.

3.**Prepare** the training data to train the meta-classifier: 
generate the training set to train the meta-classifier.

The intput folder should have the following structure and the files should have the
same naming convention:

```
└── input
    ├── tabddpm_black_box
    │   │── meta_classifier
    │   │   └── train_meta.csv
    │   │   └── train_meta_label.csv
    │   │   └── synth.csv
    │   │   └── rmia_train_meta_pred.csv
    │   │
    │   │── train
    │   │   └── tabddpm_#
    │   │       └── train_with_id.csv
    │   │       └── trans_synthetic.csv
    │   │       └── challenge_with_id.csv
    │   │       └── challenge_label.csv
    │   │       └── rmia_scores_k_5.csv
    │   ├── dev
    │   │   └── tabddpm_#
    │   │       └── trans_synthetic.csv
    │   │       └── challenge_with_id.csv
    │   │       └── rmia_scores_k_5.csv
    │   │
    │   └── final
    │   │   └── tabddpm_#
    │   │       └── trans_synthetic.csv
    │   │       └── challenge_with_id.csv
    │   │       └── rmia_scores_k_5.csv
    │   │
    └── population
        └── population_all_with_challenge_no_id.csv
```

Note: We have generated the training set to train the meta-classifier and computed RMIA scores and generated 
predictions and placed the **.csv** files in the **input** folder. You can skip this step and use the provided data, 
if you do not want to use the generated training set and RMIA scores.

4.**Train** the meta-classifier: train the meta classifier of the ensemble models and save the trained meta classifier.
Note: We have added a trained XGBoost meta-classifiers in the **output** folder. 
You can skip this step and use the provided meta-classifier, if you do not want to train your own.

```
python train.py \
--attack_model Blending++ \
--meta_train_path ../input/tabddpm_black_box/meta_classifier/train_meta.csv \
--meta_train_label_path ../input/tabddpm_black_box/meta_classifier/train_meta_label.csv \
--train_pred_proba_rmia_path ../input/tabddpm_black_box/meta_classifier/rmia_train_meta_pred.csv \
--meta_test_path ../input/tabddpm_black_box/meta_classifier/train_meta.csv \
--meta_test_label_path ../input/tabddpm_black_box/meta_classifier/train_meta_label.csv \
--test_pred_proba_rmia_path ../input/tabddpm_black_box/meta_classifier/rmia_train_meta_pred.csv \
--synth_path ../input/tabddpm_black_box/meta_classifier/synth.csv \
--real_ref_path ../input/population/population_all_with_challenge_no_id.csv \
--meta_classifier_type xgb \
--output_path ../output/train/tabddpm_black_box
```

5.**Evaluate** the ensemble model: evaluate the ensemble model with all the datasets in **train**.

```
python eval.py \
--attack_model Blending++ \
--attack_type tabddpm_black_box \
--real_ref_path ../input/population/population_all_with_challenge_no_id.csv \
--meta_classifier_blending_plus_plus_path ../output/train/tabddpm_black_box/blending_plus_plus/meta_classifier.pkl
```

6.**Predict** with the ensemble model: make predictions using the ensemble model on all the datasets 
in **dev** and **final** (pass dev or final as argument for `--dataset`).

```
python infer.py \
--attack_model Blending++ \
--attack_type tabddpm_black_box \
--real_ref_path ../input/population/population_all_with_challenge_no_id.csv \
--meta_classifier_blending_plus_plus_path ../output/train/tabddpm_black_box/blending_plus_plus/meta_classifier.pkl \
--dataset final \
--is_plot False
```

7.**Generate** the .zip file: create the .zip file for submission 
(refer to refer notebook `3.prepare_submmission.ipynb` for details)

## Requirements

Install required python packages with:

```bash
pip install -r requirements.txt
```
