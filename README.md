# Ensemble membership inference attack model

## Overview

- **Description:** This GitHub repository includes Python scripts designed to perform membership inference attacks using ensemble technique.

## Usage

1.Edit the `config.py` file (The only configurations need to be modified are **DATA_PATH** and **OUTPUT_PATH**).

2.**Training** stage: train the meta classifier of the ensemble models with the real data in **train** 
(along with the 1st and 2nd generation synthetic data generated based on the real data) 
and save the trained meta classifiers. See `1.real_data_processing.ipynb` for how to collect and split real data 
and `2.tabsyn_synth_gen_population.ipynb` and `3.tabddpm_synth_gen_population.ipynb` for instructions on how to 
generate synthetic data with TabDDPM and TabSyn. Note: We have added the trained XGBoost meta classifiers in the **output** folder.
You can skip this step and use the provided meta classifiers, if you do not want to train your own.

```
python train.py \
--attack_model Stacking Stacking+ Blending Blending+ Blending++ \
--real_train_path input/tabddpm_black_box/population/real_train_no_id.csv \
--real_val_path input/tabddpm_black_box/population/real_val_no_id.csv \
--real_test_path input/tabddpm_black_box/population/real_test_no_id.csv \
--real_ref_path input/tabddpm_black_box/population/population_no_id.csv \
--synth_train_path input/tabddpm_black_box/population/synth/1st_gen/synth_train.csv \
--synth_test_path input/tabddpm_black_box/population/synth/1st_gen/synth_test.csv \
--synth_2nd_path input/tabddpm_black_box/population/synth/2nd_gen/synth_2nd.csv \
--meta_classifier_type xgb \
--output_path /output/tabddpm_black_box \
```

3.**Generation** stage: generate a 2nd generation synthetic data in **train**, **dev** and **final**. 
The user can use `gen.py` to split the 1st generation synthetic data into training and test set and 
generate 2nd generation synthetic data for each dataset. 

```
python gen.py \
--generator tabddpm \
--dataset dev \
--ref_data_path None \
```

The synthetic data generated in **train**, **dev** and **final** should be placed in the same folder 
as the challenge data. The data folder should have the following structure and files should have the
same naming convention:

```
└── input
    ├── tabddpm_black_box
    │   │── train 
    │   │   └── tabddpm_#
    │   │       └── synth_train.csv
    │   │       └── synth_test.csv
    │   │       └── synth_2nd.csv
    │   │       └── challenge_with_id.csv
    │   │       └── challenge_label.csv    
    │   ├── dev
    │   │   └── tabddpm_#
    │   │       └── synth_train.csv
    │   │       └── synth_test.csv
    │   │       └── synth_2nd.csv    
    │   │       └── challenge_with_id.csv
    │   └── final
    │   │   └── tabddpm_#
    │   │       └── synth_train.csv
    │   │       └── synth_test.csv
    │   │       └── synth_2nd.csv
    │   │       └── challenge_with_id.csv
    └── tabsyn_black_box
        ... 
```
Note: We have split the 1st generation synthetic data and generated the 2nd generation synthetic data
and put them in the **input** folder. You can skip this step and use the provided data, 
if you do not want to generate your own synthetic data.

4.**Evaluation** stage: evaluate the MIA models with all the datasets in **train**.

```
python eval.py \
--attack_model LOGAN TableGAN DOMIAS "Soft Voting" Stacking Stacking+ Blending Blending+ Blending++ \
--attack_type tabddpm_black_box \
--real_ref_path input/tabddpm_black_box/population/population_no_id.csv \
--meta_classifier_stacking_path output/tabddpm_black_box/stacking/meta_classifier.pkl \
--meta_classifier_stacking_plus_path output/tabddpm_black_box/stacking_plus/meta_classifier.pkl \
--meta_classifier_blending_path output/tabddpm_black_box/blending/meta_classifier.pkl \
--meta_classifier_blending_plus_path output/tabddpm_black_box/blending_plus/meta_classifier.pkl \
--meta_classifier_blending_plus_plus_path output/tabddpm_black_box/blending_plus_plus/meta_classifier.pkl \
```

5.**Inference** stage: make predictions with the MIA models on all the datasets in **dev** and **final**.

```
python infer.py \
--attack_model LOGAN TableGAN DOMIAS "Soft Voting" Stacking Stacking+ Blending Blending+ Blending++ \
--attack_type tabddpm_black_box \
--real_ref_path input/tabddpm_black_box/population/population_no_id.csv \
--meta_classifier_stacking_path output/tabddpm_black_box/stacking/meta_classifier.pkl \
--meta_classifier_stacking_plus_path output/tabddpm_black_box/stacking_plus/meta_classifier.pkl \
--meta_classifier_blending_path output/tabddpm_black_box/blending/meta_classifier.pkl \
--meta_classifier_blending_plus_path output/tabddpm_black_box/blending_plus/meta_classifier.pkl \
--meta_classifier_blending_plus_plus_path output/tabddpm_black_box/blending_plus_plus/meta_classifier.pkl \
--dataset dev \
--is_plot True \
```

## Requirements

Install synthetic generation package clover with:

```bash
poetry install
````