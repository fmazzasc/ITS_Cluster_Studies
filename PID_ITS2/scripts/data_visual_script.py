import sys
import uproot 
import pandas as pd 
import matplotlib.pyplot as plt

from ROOT import TH1, TFile

sys.path.append('..')
from utils.data_reader import readFile

#fimpPath = '../data/input/ITSTPCClusterTree505582_apass5_140323.root'
#data = uproot.open(fimpPath)['ITStreeML'].arrays(library='pd')

inputFile = '/Users/giogi/Desktop/ALICE/ITS_Cluster_Studies/PID_ITS2/data/input/ITSTPCClusterTree_LHC22m_apass3_523308.root'
inputData = readFile(inputFile, 'ITStreeML')
print(inputData.columns)
hist = inputData['rofBC'].hist(bins=150)
plt.show()