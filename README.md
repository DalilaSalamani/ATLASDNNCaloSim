
This repository contains the set of scripts used to train the VAE model used for the CaloChallenge [Dataset1 pions](https://zenodo.org/records/8099322#:~:text=Download-,dataset_1_pions_1.hdf5,-md5%3A6a5f52722064a1bcd8a0bc002f16515d).

- preprocess.py: defines the data loading and preprocessing functions.
- gpu_limiter.py: defines a logic responsible for GPU memory management.
- constants.py: defines the set of common variables.
- model.py: defines the VAE model class and a handler to construct the model.
- train.py: performs model training.

## Getting Started

Create a virtual environment.
```
python3 -m venv VENV
``` 

Activate the virtual environment

```
source VENV/bin/activate
``` 

Install the list of required packages and their dependencies.

```
pip install -r requirements.txt
``` 

`setup.py` script creates necessary folders used to save model checkpoints, generate showers and validation plots.

```
python3 setup.py
``` 

## Training

To run the training:
```
python3 train.py --file-name dataset/dataset_1_pions_1.hdf5
```


## Generation 

To generate showers using the ML model and also generate the HDF5 file used for evaluation:
```
python3 generate.py --version 2 --epoch 13
``` 

## Evaluation

The evaluation is based on the [CaloChallenge code](https://github.com/CaloChallenge/homepage). Go to the evaluation folder and run:
```
python3 evaluate.py -i ../generation/VAE_dataset_1_pions_1.hdf5 -r ../dataset/dataset_1_pions_1.hdf5 -m all -d 1-pions --output_dir evaluation_plots
``` 
