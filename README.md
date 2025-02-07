# Ensemble membership inference attack model

## Overview

- **Description:** This GitHub repository includes Python scripts designed to perform membership inference attacks using ensemble technique.

## Usage

1.Edit the `config.py` file to configure the paths, metadata etc.

2.**Training** stage: train the meta classifier of the ensemble models with the real data in **train** folder 
(along with the 1st and 2nd generation synthetic data generated based on the real data) 
and save the trained meta classifiers. See `1.real_data_processing.ipynb` for how to collect and split real data 
and `2.tabsyn_synth_gen_population.ipynb`for instructions on how to generate synthetic data with TabSyn.

```
python train.py \
--real_train_path data/real_train.csv \
--real_val_path data/real_val.csv \
--real_test_path data/real_test.csv \
--synth_train_path data/synth_train.csv \
--synth_test_path data/synth_test.csv \
--synth_2nd_path data/synth_2nd.csv \
--output_path output \
```

3.**Evaluation** stage: evaluate the MIA models with all the datasets in **train** folder.

```
python eval.py \
--meta_classifier_stacking_path stacking/meta_classifier.pkl \
--meta_classifier_stacking_plus_path stacking_plus/meta_classifier.pkl \
--meta_classifier_blending_path blending/meta_classifier.pkl \
--meta_classifier_blending_plus_path blending/meta_classifier.pkl \
```

4.**Inference** stage: make predictions with the MIA models on all the datasets in **dev** and **final** folders.

```
python infer.py \
--meta_classifier_stacking_path stacking/meta_classifier.pkl \
--meta_classifier_stacking_plus_path stacking_plus/meta_classifier.pkl \
--meta_classifier_blending_path blending/meta_classifier.pkl \
--meta_classifier_blending_plus_path blending/meta_classifier.pkl \
--dataset dev \
--is_plot False \
```

## Requirements

1.Install synthetic generation package clover with:

```bash
poetry install
````

2.Generate a 2nd generation synthetic data for **training**, **evaluation** and **inference** stages. 
The user can use `gen.py` to split the 1st generation synthetic data into training and test set and 
generate 2nd generation synthetic data for each dataset. The synthetic data generated for 
**evaluation** and **inference** stages should be placed in the same folder 
as the challenge data. The input data folder should have the following structure:

```
└── input_folder
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