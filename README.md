# Ensemble memembership inference attack model

## Overview

- **Description:** This GitHub repository includes Python scripts designed to perform membership inference attacks using ensemble technique.

## Usage

1.Place the real data in the **data** folder.

2.Specify the path of the data and metadata in the `config.py` file.

3.Run the command below to generate and evaluate synthetic data via utility metrics and individual and ensemble MIAs. The results will be stored in the **output** folder.
The available generators are: **synthpop**, **smote**, **datasynthesizer**, **mst**, **ctgan**, **tvae**, **ctabgan** and **findiff**.

```
python main.py \
    --generator ctgan \
    --iteration 10 \
```

## Requirements

Install synthetic generation package clover with:
```bash
poetry install