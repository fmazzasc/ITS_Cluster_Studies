'''
python script to produce training variables distributions
run: python PlotTrainingVars.py
'''
from ROOT import TFile, TH1F, TCanvas, TMath, TLegend, kBlack, kAzure, kOrange, kSpring, kOpenCircle, kFullCross, kFullSquare, TLatex # pylint: disable=import-error,no-name-in-module
import sys
import numpy as np
import pandas as pd
import uproot
from alive_progress import alive_bar
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.AnalysisUtils import ComputeRatioDiffBins, ScaleGraph, ComputeRatioGraph #pylint: disable=wrong-import-position,import-error

def angle_corr(df): #pylint: disable=too-many-statements, too-many-branches

    hSnPhi = []
    hTanLam = []
    df_sel = ApplySelections(df, ClSizeMin=50, ClSizeMax=10000, pMin=0, pMax=0.2, verbose=False)

    for i in range(7):
        hSnPhi.append(TH1F(f'hSnPhiL{i}', '', 200, -1, 1))
        hTanLam.append(TH1F(f'hTanLamL{i}', '', 200, -1, 1))

        dfSnPhi_sel = df_sel[i][f'SnPhiL{i}']
        dfTanLam_sel = df_sel[i][f'TanLamL{i}']

        for snphi in (dfSnPhi_sel):
            hSnPhi[i].Fill(snphi)
        for tanlam in (dfTanLam_sel):
            hTanLam[i].Fill(tanlam)

    del df_sel, dfSnPhi_sel, dfTanLam_sel

    return hSnPhi, hTanLam


def ApplySelections(df, ClSizeMin=0, ClSizeMax=1000000, SnPhiMin=-1, SnPhiMax=1, TanLamMin=-1, TanLamMax=1, pMin=0, pMax=10000, verbose=False):
    '''
    Prepare selections
    '''
    df_sel = []
    for layer in range(7):
        df_sel.append(df)
        df_sel[layer] = df_sel[layer].query(f'{ClSizeMin} < ClSizeL{layer} < {ClSizeMax}')
        df_sel[layer] = df_sel[layer].query(f'{pMin} < p < {pMax}')
        df_sel[layer] = df_sel[layer].query(f'{SnPhiMin} < SnPhiL{layer} < {SnPhiMax}')
        df_sel[layer] = df_sel[layer].query(f'{TanLamMin} < TanLamL{layer} < {TanLamMax}')

    return df_sel

def CheckClusterCorrelations(df):
    '''
    Check cluster correlations
    '''
    #_____________________________________________
    # Sice correlations
    hAll, hSel = [], []


def main():
    '''
    Main function
    '''
    #_____________________________________________
    # Get dataframe
    data = '/data/fmazzasc/its_data/TreeITSClusters505658.root'
    df = uproot.open(data)['ITStreeML'].arrays(library='pd')

    #_____________________________________________
    # Angle correletions
    hSnPhi, hTanLam = [], []
    outlabel = 'Cl50_10000_p0_0.2'
    hSnPhi, hTanLam = angle_corr(df)

    #_____________________________________________
    # Saving
    outFile = TFile(f'ClusITSAngles_{outlabel}.root', 'recreate')
    for i, (hSn, hTg) in enumerate(zip(hSnPhi, hTanLam)):
        hSn.SetName(f'hSnPhiL{i}')
        hTg.SetName(f'hTanLamL{i}')
        hSn.Write()
        hTg.Write()
    outFile.Close()

    input('Press enter to exit')
    sys.exit()

main()
