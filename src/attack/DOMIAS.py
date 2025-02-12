import numpy as np
import pandas as pd
import os
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler


MODEL_NAME= 'tabddpm' #'tabsyn'
SUBMISSION_NAME= 'SUBMISSION_DOMIAS'
DATA_SPLIT= 'train'
AUX_DIRS= ['/home/hadrien/Documents/Phd/ensemble-mia/data/tabddpm_black_box/train','/home/hadrien/Documents/Phd/ensemble-mia/data/tabsyn_black_box/train']
SYNTH_DIR= f'/home/hadrien/Documents/Phd/ensemble-mia/data/{MODEL_NAME}_black_box/{DATA_SPLIT}'
CHALLENGE_DIR= f'/home/hadrien/Documents/Phd/ensemble-mia/data/{MODEL_NAME}_black_box/{DATA_SPLIT}'
EVAL_ATTACK= False
SAVE_FOLDER= f'/home/hadrien/Documents/Phd/ensemble-mia/{SUBMISSION_NAME}/{MODEL_NAME}_black_box/{DATA_SPLIT}'

def deduplicate_transactions(df):
    return df.drop_duplicates(subset=['trans_id','account_id'])

def load_aux_dataframes(train_dir):
    # Load all train_with_id CSV files from train directory
    train_dfs = []
    for train_dir in AUX_DIRS:
        for sub_folder in os.listdir(train_dir):
            file_path = os.path.join(train_dir, sub_folder, 'train_with_id.csv')
            df = pd.read_csv(file_path)
            train_dfs.append(df)
    
    return deduplicate_transactions(pd.concat(train_dfs, ignore_index=True))

#DOMIAS RATIO = PG(x)/Paux(x)ir0lacStCyr

def domias_attack(df_challenge, df_synth, df_aux,density_aux=None, drop_columns=['trans_id','account_id']):
    if density_aux is None:
        density_aux= gaussian_kde(df_aux.drop(columns=drop_columns).values.transpose(1,0))
    density_synth= gaussian_kde(df_synth.values.transpose(1,0))
    PG= density_synth(df_challenge.drop(columns=drop_columns).values.transpose(1,0))
    Paux= density_aux(df_challenge.drop(columns=drop_columns).values.transpose(1,0))
    df_challenge['PG']= PG
    df_challenge['Paux']= Paux
    df_challenge['domias_ratio']= PG/Paux   
    return df_challenge

def compute_TPR_FPR(df_challenge, df_label, threshold=.1):
    """return TPR at threshold FPR"""
    fpr, tpr, thresholds = roc_curve(df_label, df_challenge['domias_ratio'])
    return tpr[np.argmin(np.abs(fpr - threshold))]

def scale_domias_ratio(df_challenge):
    scaler= MinMaxScaler()
    df_challenge['domias_ratio']= scaler.fit_transform(df_challenge[['domias_ratio']])
    return df_challenge


def main():
    df_aux= load_aux_dataframes(AUX_DIRS)
    density_aux= gaussian_kde(df_aux.drop(columns=['trans_id','account_id']).values.transpose(1,0))
    TPR_list= []
    for dir in os.listdir(SYNTH_DIR):
        i= int(dir.split('_')[1])
        df_synth_i= pd.read_csv(f'{SYNTH_DIR}/{dir}/trans_synthetic.csv')
        df_challenge_i= pd.read_csv(f'{CHALLENGE_DIR}/{dir}/challenge_with_id.csv')
        df_challenge_i= domias_attack(df_challenge_i, df_synth_i, df_aux, density_aux)
        df_challenge_i= scale_domias_ratio(df_challenge_i)
        if EVAL_ATTACK:
            df_label_i= pd.read_csv(f'{CHALLENGE_DIR}/{dir}/challenge_label.csv')
            TPR_i= compute_TPR_FPR(df_challenge_i, df_label_i, threshold=.1)
            print(f'TPR at threshold 0.1 for {MODEL_NAME}_{i}: {TPR_i}')
            TPR_list.append(TPR_i)
        else:
            #check if target folder exists
            if not os.path.exists(os.path.join(SAVE_FOLDER,f'{MODEL_NAME}_{i}')):
                os.makedirs(os.path.join(SAVE_FOLDER,f'{MODEL_NAME}_{i}'))
            df_challenge_i['domias_ratio'].to_csv(os.path.join(SAVE_FOLDER,f'{MODEL_NAME}_{i}/prediction.csv'), index=False, header=False)
    if EVAL_ATTACK:
        print('mean TPR:',np.mean(TPR_list))
        print('std TPR:',np.std(TPR_list))
        print('min TPR:',np.min(TPR_list))
        print('max TPR:',np.max(TPR_list))

if __name__ == "__main__":
    main()

"""
import csv
import os
import random
import zipfile

TABDDPM_DATA_DIR='/home/hadrien/Documents/Phd/ensemble-mia/SUBMISSION_DOMIAS/tabddpm_black_box'
TABSYN_DATA_DIR='/home/hadrien/Documents/Phd/ensemble-mia/SUBMISSION_DOMIAS/tabsyn_black_box'

import os
import pandas as pd
import numpy as np

def check_predictions(base_dir, models=['tabddpm', 'tabsyn'], splits=['dev', 'final']):
    for model in models:
        for split in splits:
            folder_path = os.path.join(base_dir, f'{model}_black_box', split)
            if not os.path.exists(folder_path):
                print(f"Skipping {folder_path} - does not exist")
                continue
                
            for subdir in os.listdir(folder_path):
                file_path = os.path.join(folder_path, subdir, 'prediction.csv')
                
                if os.path.exists(file_path):
                    # Read the CSV file without header
                    predictions = pd.read_csv(file_path, header=None)
                    
                    # Check if any value is greater than 1
                    if (predictions > 1).any().any():
                        print(f"Found values > 1 in: {file_path}")
                        print(f"Max value: {predictions.max().max()}")

if __name__ == "__main__":
    base_dir = '/home/hadrien/Documents/Phd/ensemble-mia/SUBMISSION_DOMIAS'
    check_predictions(base_dir)

with zipfile.ZipFile(f"black_box_single_table_submission.zip", 'w') as zipf:
    for phase in ["dev", "final"]:
        for base_dir in [TABDDPM_DATA_DIR, TABSYN_DATA_DIR]:
            root = os.path.join(base_dir, phase)
            model_folders = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
            for model_folder in sorted(model_folders, key=lambda d: int(d.split('_')[1])):
                path = os.path.join(root, model_folder)
                if not os.path.isdir(path): continue

                file = os.path.join(path, "prediction.csv")
                if os.path.exists(file):
                    # Use `arcname` to remove the base directory and phase directory from the zip path
                    arcname = os.path.relpath(file, os.path.dirname(base_dir))
                    zipf.write(file, arcname=arcname)
                else:
                    raise FileNotFoundError(f"`prediction.csv` not found in {path}.")
"""