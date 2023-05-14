#
#   Script to investigate eventual anomalies in the strange patterns observed
#

import sys
import pandas as pd
from ROOT import TH1D, TFile

sys.path.append('..')
from utils.data_reader import readFile
from utils.plotter import Plotter


def rangeHists(inputData, outputFilePath, column, p_ranges):
    """
    Create histograms of given variable (column) of a dataset in a limited range of momentum.

    Parameters
    ----------
        inputData (pd.DataFrame): input data
        outputFilePath (str): file where the histograms will be saved to
        column (str): column of the dataframe to plot
        p_ranges (list[list[float, float]]): list of momentum ranges associated to each histogram
    """

    outputFile = TFile(outputFilePath, 'recreate') 

    for prange in p_ranges:

        data = inputData.query(f'{prange[0]} <= p < {prange[1]}', inplace=False)
        hist = TH1D(f'{column}_{prange[0]}_{prange[1]}', f'{column}#; {prange[0]} #leq p < {prange[1]}; {column}; Counts', 250, 0, 25)
        for x in data[column]:  hist.Fill(x)

        outputFile.cd()
        hist.Write()

    outputFile.Close()

def peak_inspection(inputData, outputFilePath):
    """
    Create histograms for some variables that may suggest interpretation for the presence of the peak

    Parameters
    ----------
        inputData (pd.DataFrame): input data
        outputFilePath (str): file where the histograms will be saved to    
    """

    outputFile = TFile(outputFilePath, 'recreate')
    peakPlotter = Plotter(inputData, outputFile)
    
    xVarsToPlot = [#'Delta', 
                    'tpcITSchi2', 'delta_p', 'nClusTPC', 'nSigmaP', 'nSigmaK', 'nSigmaPi', 'ClSizeL0', 'ClSizeL1', 'ClSizeL2', 'ClSizeL3', 'ClSizeL4', 'ClSizeL5', 'ClSizeL6', 'tgL', 'meanPattID', 'clSizeCosLam']
    plot_specifics = [#['#Delta', 300, -1.5, 1.5], 
                     ['tpcITSchi2', 100, 0, 10], ['delta_p', 100, -0.5, .5], ['nClusTPC', 100, 0, 200], ['nSigmaP', 100, -5, 5], ['nSigmaK', 100, -5, 5], ['nSigmaPi', 150, -5, 5],
                     ['Cl. Size L_{0}', 25, 0, 25], ['Cl. Size L_{1}', 25, 0, 25], ['Cl. Size L_{2}', 25, 0, 25], ['Cl. Size L_{3}', 25, 0, 25], ['Cl. Size L_{4}', 25, 0, 25], ['Cl. Size L_{5}', 25, 0, 25], ['Cl. Size L_{6}', 25, 0, 25],
                     ["<tan#lambda>", 80, -4, 4], ["<Pattern ID>", 100, 0, 100], ['<Cl. size> <cos#lambda>', 250, 0, 25]]
    peakPlotter.plot1D(xVarsToPlot, plot_specifics)

    outputFile.Close()

def clSizeHist(inputData, histName, outputFile):
    """
    Create histograms for cluster size in a file
    """

    hist = TH1D(histName, histName, 250, 0, 25)
    hist.SetTitle(f'{histName}; <Cl. size> <cos#lambda>; Counts')
    for x in inputData['clSizeCosLam']: hist.Fill(x)
    outputFile.cd()
    hist.Write()

"""
if __name__ == '__main__':

    inputDataPath = '../data/preprocessed/TPC/TrainSet.parquet.gzip'
    inputData = readFile(inputDataPath, 'ITStreeML')
    pData = inputData.query("label == 'P'", inplace=False)

    outputFilePath = '../output/p_peak_inspection/clSizeCosLam_pranges.root'
    p_ranges = [[0., 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8],
                [0.8, 0.9], [0.9, 1.], [1.0, 1.1], [1.1, 1.2], [1.2, 1.3], [1.3, 1.4], [1.4, 1.5]]
    
    rangeHists(pData, outputFilePath, 'clSizeCosLam', p_ranges)

    # create comparison hist with p and pi full datasets
    piData = inputData.query("label == 'Pi'", inplace=False)
    outputFilePathP = '../output/p_peak_inspection/inspection_p.root'
    outputFilePathPi = '../output/p_peak_inspection/inspection_pi.root'
    peak_inspection(pData, outputFilePathP)
    peak_inspection(piData, outputFilePathPi)

    # create cluster size histograms in particular momentum ranges
    pData0405 = pData.query('0.4 <= p < 0.5', inplace=False)
    piData0405 = piData.query('0.4 <= p < 0.5', inplace=False)
    outputFileClSize = TFile('../output/p_peak_inspection/inspection_clSize0405.root', 'recreate')
    clSizeHist(pData0405, 'clSizeCosLam_0405_P', outputFileClSize)
    clSizeHist(piData0405, 'clSizeCosLam_0405_Pi', outputFileClSize)
    outputFileClSize.Close()

    # select specific peak
    peakData = inputData.query('0.3 <= p < 0.4 and 0 <= clSizeCosLam < 3.3', inplace=False)
    outputFilePath = '../output/p_peak_inspection/inspection_0405.root'
    peak_inspection(peakData, outputFilePath)

    peakData = inputData.query('0.3 <= p < 0.4 and 0 <= clSizeCosLam < 3.3', inplace=False)
    outputFilePath = '../output/p_peak_inspection/inspection_0304.root'
    peak_inspection(peakData, outputFilePath)

    peakData = inputData.query('0.3 <= p < 0.4 and 0 <= clSizeCosLam < 3.3 and nSigmaPi <= 1', inplace=False)
    outputFilePath = '../output/p_peak_inspection/inspection_maybePions_0304.root'
    peak_inspection(peakData, outputFilePath)
"""

if __name__ == "__main__":
    
    inputDataPath = '../data/preprocessed/TPC/ApplicationDf_beta_pflat.parquet.gzip'
    inputData = readFile(inputDataPath, 'ITStreeML')
    
    highBetaPi = inputData.query('label == "Pi" and beta_pred > beta', inplace=False)
    peak_inspection(highBetaPi, '../output/peak_inspection/highBetaPi.root')

    pi = inputData.query('label == "Pi"', inplace=False)
    peak_inspection(pi, '../output/peak_inspection/pi.root')