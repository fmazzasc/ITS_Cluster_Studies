# **PID ITS2**

## General overview

Here are the macros to perform particle identification with data collected by ALICE's ITS2. 
Cluster size information is combined with other variables to perform PID using machine learning algorithms.

## Requirements

Make sure you have the following prerequisites installed before running the software:

* [**Python**](https://www.python.org/downloads/)

* [**Polars**](https://pola-rs.github.io/polars/): Data manipulation library specialized in big datasets. 

* [**Optuna**](https://optuna.readthedocs.io/en/stable/index.html) Library for hyperparameter optimisation of machine learning models.

* [**Sci-kit Learn**](https://scikit-learn.org/stable/documentation.html)

* [**xgboost**](https://xgboost.readthedocs.io/en/latest/): For BDT models.

* [**hipe4ml**](https://hipe4ml.readthedocs.io/en/latest/)

* [**matplotlib**](https://matplotlib.org/stable/contents.html)

* [**ROOT**](https://root.cern/documentation.html)

## Usage

Files to be executed are colected in the scripts folder.

* performPID.py perfoms particle identification. All configurations can be set in files placed in the configs folder. To run the macro, adjust the configuration files based on your need, then move to the scripts folder and simply run the script from terminal with "python3 performPID.py"

