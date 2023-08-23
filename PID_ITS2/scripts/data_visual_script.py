import sys
import uproot 
import pandas as pd 
import matplotlib.pyplot as plt

from ROOT import TH1D, TFile, TCanvas, gStyle
from ROOT import kAzure

sys.path.append('..')
from utils.data_reader import readFile

gStyle.SetOptStat(0)

#fimpPath = '../data/input/ITSTPCClusterTree505582_apass5_140323.root'
#data = uproot.open(fimpPath)['ITStreeML'].arrays(library='pd')

inputFile = '../data/input/ITSTPCClusterTree_LHC22m_apass3_523308.root'
inputData = readFile(inputFile, 'ITStreeML')
outputFile = TFile('../output/analysis/rofHIst.root', 'recreate')
#print(inputData.columns)
#hist = inputData['rofBC'].hist(bins=150)
#plt.show()


## VISUALIZE READOUT FRAMES
histRoF = TH1D('rofHist', 'Readout Frame visualization; Readout Frame; Counts (a.u.)', 3500, 0, 3500)
for x in inputData['rofBC']:  histRoF.Fill(x)

canvas = TCanvas('canvas', '', 900, 900)
histRoF.SetFillColorAlpha(kAzure-4, 0.5)
#histRoF.SetLineWidth(0)
histRoF.Draw()
canvas.SaveAs('../report/img5/rofHist.png')

outputFile.cd()
histRoF.Write()
outputFile.Close()