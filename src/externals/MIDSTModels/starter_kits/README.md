# Starter Kits
This folder contains starter kits that provide an overview of the MIDST competition and outline how to package a submission to submit to CodaBench. There is a starter kit for each of the four competitions:

| Starter Kit File              | CodaBench Competition       |
|-------------------------------|-----------------------------|
| blackbox_multi_table.ipynb    | [Black Box Multi Table](https://www.codabench.org/competitions/4671/)                            |
| blackbox_single_table.ipynb   | [Black Box Single Table](https://www.codabench.org/competitions/4670/)                             |
| whitebox_multi_table.ipynb    | [White Box Multi Table](https://www.codabench.org/competitions/4673/)                             |
| whitebox_single_table.ipynb   | [White Box Single Table](https://www.codabench.org/competitions/4672/)                             |


## Environment Installation
In order to run the starter kits, a minimal environment is provided. Feel free to use your own environment if it already has the required dependencies. The environment can be created as follows: 

```bash
python3 -m venv /path/to/env
source /path/to/env/bin/activate
pip install -r requirements.txt
jupyter notebook
```
