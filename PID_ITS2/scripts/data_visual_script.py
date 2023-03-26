import sys
import uproot 
import pandas as pd 
import matplotlib.pyplot as plt

from ROOT import TH1, TFile

sys.path.append('..')
from utils.data_reader import readFile

#fimpPath = '../data/input/ITSTPCClusterTree505582_apass5_140323.root'
#data = uproot.open(fimpPath)['ITStreeML'].arrays(library='pd')

inputFile = '../data/input/particles_pid_520143.parquet'
inputData = readFile(inputFile)
print(inputData.columns)
hist = inputData['rofBC'].hist(bins=150)
plt.show()