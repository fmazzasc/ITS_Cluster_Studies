import sys
import uproot 
import pandas as pd 

from ROOT import TH1, TFile

sys.path.append('..')
from utils.data_reader import readFile

#fimpPath = '../data/input/ITSTPCClusterTree505582_apass5_140323.root'
#data = uproot.open(fimpPath)['ITStreeML'].arrays(library='pd')

inputFile = '../data/preprocessed/TPC/ApplicationDf_beta_pflat.parquet.gzip'
inputData = readFile(inputFile)
print(inputData.describe())
print(inputData.columns)