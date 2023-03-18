import uproot
import pandas as pd 

from ROOT import TH1, TFile

fimpPath = '../data/input/ITSTPCClusterTree505582_apass5_140323.root'

data = uproot.open(fimpPath)['ITStreeML'].arrays(library='pd')
print(data.describe())