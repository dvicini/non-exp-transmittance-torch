# Non-exponential Transmittance PyTorch Operator

This repository contains a simple PyTorch CUDA operator that implements the transmittance model proposed in our paper:

Delio Vicini, Wenzel Jakob, Anton Kaplanyan, [A Non-Exponential Transmittance Model for Volumetric Scene Representations](https://dvicini.github.io/non-exponential-representation/), ACM Transactions on Graphics (Proceedings of SIGGRAPH), 40(4), August 2021. 

## Installation
To install the operator, please navigate to the `op` folder and run 
```
python setup.py install
```

The operator is then available in a Python package `nonexp`. 


## Usage
See `example.py` for an example on how to invoke the operator and the paper for the mathematical definition & motivation.


## Citation
When using this code for research, please cite it using the following bibtex reference:

```
@article{Vicini2021NonExponential,
    author    = {Vicini, Delio and Jakob, Wenzel and Kaplanyan, Anton},
    title     = {A Non-Exponential Transmittance Model for Volumetric Scene Representations},
    journal   = {Transactions on Graphics (Proceedings of SIGGRAPH),
    volume    = {40},
    number    = {4},
    year      = {2021},
    month     = aug,
    pages     = {136:1--136:16},
    doi       = {10.1145/3450626.3459815},
}
```
