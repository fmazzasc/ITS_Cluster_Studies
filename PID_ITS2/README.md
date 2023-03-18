# **PID ITS Giorgio Alberto**

## **Introduction**
General overview for the macros used to study ITS2 PID capabilities. Many macros used to compare distribution between different species etc were deleted. You can rebuild basically any of them using as input .root files created by PID_ITS.py and PID_ITS2.py.

---

## **How to use**
The only script to run is PID_ITS2.py. Others are provided both because containing necessary classes and functions for the former or as reference for the work done. 

Different options (whether to save plots or not, to use reweighting and/or data augmentation methods) can be set from the configuration file (`PID_ITS_Giorgio_Alberto/configs/config2.yml`). 

---

## **Requirements**

* [hipe4ml](https://github.com/hipe4ml/hipe4ml "hipe4ml home") 

**NOTE**: a version of hipe4ml is included in the folder. It contains a small change (currently a pull request is active) that lets hipe4ml perform optuna hyperparameters optimization with reweighting of the dataset (as optuna is already able to do).

---

## **What are the macros**

### **CompareGraph.py**
Almost the same as the one in utils. Can be used for TH2s, although PlotAdjustments is probably more adequate.
To run it, adjust the `PID_ITS_Giorgio_Alberto/configs/config_plots.yml` file accordingly, then run
```
python3 CompareGraph.py ../configs/config_plots.yml
```
**NOTE**: This macro was created with a previous version of `StyleFormatter.py`, included from `utils/StyleFormatter_old.py`. 

### **PID_ITS.py**
Macro used to preprocess experimental data and train and apply ML models to those samples. This is the core idea to identify particles with ITS2. 
- This macro uses a yaml configuration file at `PID_ITS_Giorgio_Alberto/configs/config.yml`
- This is an outdated version. Most functions have been reimplemented as classes. Results for hybrid dataset and TPC dataset were created with this version.

### **PID_ITS2.py**
Macro used to preprocess experimental data and train and apply ML models to those samples. This is the core idea to identify particles with ITS2. 
* This macro uses a yaml configuration file at `PID_ITS_Giorgio_Alberto/configs/config2.yml`
* This is the newest version. It should be a lot easier to read and largely relies on classes from UsefulClasses.py. Results for MC samples were created with this version.

### **PlotAdjustments.py**
Manages multiple settings to make plots look better. Uses yaml configuration files like `PID_ITS_Giorgio_Alberto/configs/config_adj2.yml`
**NOTE**: This macro was created with the current version of `StyleFormatter.py`, included from `utils/StyleFormatter.py`. 

### **ROOT_Graph.py**
Initially included to adjust plot options. Probably better to just use macros in utils folder.

### **UsefulClasses.py**
Macro with essential classes used in PID_ITS2.py. They implement the basic processes needed for the ML training and application.

### **UsefulFunctions.py**
Macro with essential functions used in PID_ITS.py. Most of them have been implemented again in some classes from UsefulClasses.py.
