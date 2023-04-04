#
#   Script to compare cluster sizes for samples in different readout frames
#

import os
import sys
import uproot 
import pandas as pd 
import matplotlib.pyplot as plt

from ROOT import TH1D, TFile

sys.path.append('..')
from utils.data_reader import readFile
from utils.compare_graph import multiHistCanvas

# GLOBAL VARIABLES
part_names = ['P', 'K', 'Pi']
compare_graph_configFile = '/Users/giogi/Desktop/ALICE/ITS_Cluster_Studies/PID_ITS2/configs/config_compareGraph.yml'

#

def split_for_column(dataset, column):
    """
    For a given dataset, return a list of subsets with unique values of a column

    Parameters
    ----------
        dataset (pd.DataFrame): dataset to split
        column (str): column to consider

    Returns
    -------
        unique_dfs (list[pd.DataFrame]): list of datasets with unique values for the given column
    """

    unique_values = dataset[column].unique().tolist()
    unique_dfs = []

    for unique in unique_values:    unique_dfs.append(dataset.query(f"{column} == {unique}", inplace=False))
    return unique_dfs

def compare_rof(inputData, outputFilePath, column):
    """
    Create a ROOT file with canvas with comparison between particle distribution of given column for a specific
    readout frame and their total distribution
    """

    rofDatasets = split_for_column(inputData, 'rofBC')
    outputFile = TFile(outputFilePath, 'recreate')

    for name in part_names:

        partDataset = inputData.query(f"label == '{name}'", inplace=False)
        partHist = TH1D(f'{name}', f'{name}', 250, 0, 25)
        for x in partDataset[column]:   partHist.Fill(x)

        for rofDataset in rofDatasets:

            rofPartDataset = rofDataset.query(f"label == '{name}'", inplace=False)
            rofValue = rofPartDataset['rofBC'].unique()[0]
            rofHist = TH1D(f'{name}_rofBC_{rofValue}', f'{name}#; rofBC = {rofValue}', 250, 0, 25)
            for x in rofPartDataset[column]:   rofHist.Fill(x)

            multiHistCanvas(compare_graph_configFile, outputFile, f'{name}_rof_{rofValue}_{column}', 
                            True, partHist, rofHist)
    
    outputFile.Close()

def compare_rof_Pi(inputData, outputFilePath, column):
    """
    Create a ROOT file with canvas with comparison between particle distribution of given column for a specific
    readout frame and their total distribution, as well as the pi distribution
    """

    rofDatasets = split_for_column(inputData, 'rofBC')
    outputFile = TFile(f'{os.path.splitext(outputFilePath)[0]}_Pi{os.path.splitext(outputFilePath)[1]}', 'recreate')

    # pi distribution
    piDataset = inputData.query(f"label == 'Pi'", inplace=False)
    piHist = TH1D('Pi', 'Pi', 250, 0, 25)
    for x in piDataset[column]:   piHist.Fill(x)


    for name in part_names:

        partDataset = inputData.query(f"label == '{name}'", inplace=False)
        partHist = TH1D(f'{name}', f'{name}', 250, 0, 25)
        for x in partDataset[column]:   partHist.Fill(x)

        for rofDataset in rofDatasets:

            rofPartDataset = rofDataset.query(f"label == '{name}'", inplace=False)
            rofValue = rofPartDataset['rofBC'].unique()[0]
            rofHist = TH1D(f'{name}_rofBC_{rofValue}', f'{name}#; rofBC = {rofValue}', 250, 0, 25)
            for x in rofPartDataset[column]:   rofHist.Fill(x)

            multiHistCanvas(compare_graph_configFile, outputFile, f'{name}_rof_{rofValue}_{column}', 
                            True, partHist, rofHist, piHist)
    
    outputFile.Close()


if __name__ == "__main__":

    inputFile = '/Users/giogi/Desktop/ALICE/ITS_Cluster_Studies/PID_ITS2/data/preprocessed/TPC/TrainSet.parquet.gzip'
    inputData = readFile(inputFile, 'ITStreeML')

    outputFilePath = '../output/rofComparison/rofComparison.root'
    compare_rof(inputData, outputFilePath, 'clSizeCosLam')
    compare_rof_Pi(inputData, outputFilePath, 'clSizeCosLam')
    
