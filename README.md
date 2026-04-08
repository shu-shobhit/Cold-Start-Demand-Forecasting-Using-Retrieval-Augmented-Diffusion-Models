# RATD
This repository contains the experiments in the Neurips 2024 paper "[Retrieval-Augmented Diffusion Models for Time Series Forecasting](https://arxiv.org/abs/2410.18712)" by Jingwei Liu, Ling Yang, Hongyan Li and Shenda Hong.

## Requirement

Please install the packages in requirements.txt

## Notice

This version is not the final version of our code, we will update the full version ASAP.

## Data
The electricity dataset can be found at ./data/ts2vec

## Experiments 
### Retrieval
We use the TCN as our encoder, the code can be found at ./TCN-master. 
```shell
python retrieval.py --type encode
```
To save the references, you can run
```shell
python retrieval.py --type retrieval
```
### Training and forecasting for the electricity dataset
```shell
python exe_forecasting.py --datatype electricity
```

## Acknowledgements

Codes are based on [CSDI](https://github.com/ermongroup/CSDI)



