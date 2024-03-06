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
compare_graph_configFile2 = '/Users/giogi/Desktop/ALICE/ITS_Cluster_Studies/PID_ITS2/configs/config_compareGraph2.yml'

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

def compare_rof(inputData, configFile, outputFilePath, column, additional_selections=None):
    """
    Create a ROOT file with canvas with comparison between particle distribution of given column for a specific
    readout frame and their total distribution

    Parameters
    ----------
        inputData (pd.DataFrame): input data
        configFile (str): YAML configuration file used by multiHistCanvas
        outputFilePath (str): file to save canvases to 
        column (str): name of the column to generate the histogram from
        additional_selection (str): additional tags to use on the dataset before generating histograms
    """

    if additional_selections is not None:   additional_selections = f'and {additional_selections}'
    else:                                   additional_selections = ''

    temp_rofDatasets = split_for_column(inputData, 'rofBC')
    rofDatasets = [item for item in temp_rofDatasets if item['rofBC'].unique()[0] != 2178]   # discard problematic rof values
    outputFile = TFile(outputFilePath, 'recreate')

    for name in part_names:

        partDataset = inputData.query(f"label == '{name}' {additional_selections}", inplace=False)
        partHist = TH1D(f'{name}', f'{name}', 250, 0, 25)
        for x in partDataset[column]:   partHist.Fill(x)

        for rofDataset in rofDatasets:

            rofPartDataset = rofDataset.query(f"label == '{name}' {additional_selections}", inplace=False)
            rofValue = rofPartDataset['rofBC'].unique()[0]
            rofHist = TH1D(f'{name}_rofBC_{rofValue}', f'{name}#; rofBC = {rofValue}', 250, 0, 25)
            for x in rofPartDataset[column]:   rofHist.Fill(x)

            multiHistCanvas(configFile, outputFile, f'{name}_rof_{rofValue}_{column}', 
                            True, partHist, rofHist)
    
    outputFile.Close()

def compare_rof_Pi(inputData, configFile, outputFilePath, column, additional_selections=None):
    """
    Create a ROOT file with canvas with comparison between particle distribution of given column for a specific
    readout frame and their total distribution, as well as the pi distribution

    Parameters
    ----------
        inputData (pd.DataFrame): input data
        configFile (str): YAML configuration file used by multiHistCanvas
        outputFilePath (str): file to save canvases to 
        column (str): name of the column to generate the histogram from
        additional_selection (str): additional tags to use on the dataset before generating histograms
    """

    if additional_selections is not None:   additional_selections = f'and {additional_selections}'
    else:                                   additional_selections = ''

    temp_rofDatasets = split_for_column(inputData, 'rofBC')
    rofDatasets = [item for item in temp_rofDatasets if item['rofBC'].unique()[0] != 2178]   # discard problematic rof values
    outputFile = TFile(f'{os.path.splitext(outputFilePath)[0]}_Pi{os.path.splitext(outputFilePath)[1]}', 'recreate')

    # pi distribution
    piDataset = inputData.query(f"label == 'Pi' {additional_selections}", inplace=False)
    piHist = TH1D('Pi', 'Pi', 250, 0, 25)
    for x in piDataset[column]:   piHist.Fill(x)


    for name in part_names:

        partDataset = inputData.query(f"label == '{name}' ", inplace=False)
        partHist = TH1D(f'{name}', f'{name}', 250, 0, 25)
        for x in partDataset[column]:   partHist.Fill(x)

        for rofDataset in rofDatasets:

            rofPartDataset = rofDataset.query(f"label == '{name}' {additional_selections}", inplace=False)
            rofValue = rofPartDataset['rofBC'].unique()[0]
            rofHist = TH1D(f'{name}_rofBC_{rofValue}', f'{name}#; rofBC = {rofValue}', 250, 0, 25)
            for x in rofPartDataset[column]:   rofHist.Fill(x)

            multiHistCanvas(configFile, outputFile, f'{name}_rof_{rofValue}_{column}', 
                            True, partHist, rofHist, piHist)
    
    outputFile.Close()


if __name__ == "__main__":

    inputFile = '/Users/giogi/Desktop/ALICE/ITS_Cluster_Studies/PID_ITS2/data/preprocessed/TPC/TrainSet.parquet.gzip'
    inputData = readFile(inputFile, 'ITStreeML')

    outputFilePath = '../output/rofComparison/rofComparison.root'
    compare_rof(inputData, compare_graph_configFile, outputFilePath, 'clSizeCosLam')
    compare_rof_Pi(inputData, compare_graph_configFile, outputFilePath, 'clSizeCosLam')

    additional_selections = '0.3 <= p < 0.4'
    outputFilePath2 = '../output/rofComparison/rofComparison_0304.root'
    compare_rof(inputData, compare_graph_configFile2, outputFilePath2, 'clSizeCosLam', additional_selections=additional_selections)
    compare_rof_Pi(inputData, compare_graph_configFile2, outputFilePath2, 'clSizeCosLam', additional_selections=additional_selections)
    
