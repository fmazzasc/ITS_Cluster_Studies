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

#
# General varibales
#__________________________________
piMass = 0.13957000
kMass = 0.4937
pMass = 0.93827200

labels = ['#pi', 'K', 'p']
colors = [kAzure+2, kOrange-2, kSpring+2]
markers = [kOpenCircle, kFullCross, kFullSquare]

#
# Tag conditions
#__________________________________
eTag = '-2 < nSigmaE < 1 and nSigmaK > 4 and nSigmaPi > 2 and p < 0.2'
piTag = 'nSigmaPiAbs < 1 and nSigmaKAbs > 3 and p <= 0.7'
kTag = 'nSigmaKAbs < 1 and nSigmaPiAbs > 3 and p <= 0.7'
pTag = 'nSigmaPAbs < 1 and nSigmaKAbs > 3 and nSigmaPiAbs > 3 and p <= 0.7'
#______________________________________________________________________________


def SetHistStyle(histo, color, marker, xtitle='', ytitle='', style=1):
    '''
    Method to set histogram style
    '''
    histo.SetTitle(f'{xtitle}')
    histo.SetLineColor(color)
    histo.SetLineStyle(style)
    histo.SetLineWidth(2)
    histo.SetMarkerColor(color)
    histo.SetMarkerStyle(20)
    histo.SetMarkerSize(1)
    histo.SetMarkerStyle(marker)
    histo.SetStats(0)
    histo.SetTitle('')
    histo.GetXaxis().SetTitle(f'{xtitle}')
    histo.GetXaxis().SetTitleSize(0.06)
    histo.GetXaxis().SetTitleOffset(0.8)
    histo.GetXaxis().SetLabelSize(0.04)
    histo.GetYaxis().SetTitleSize(0.05)
    histo.GetYaxis().SetTitleOffset(1.1)
    histo.GetYaxis().SetLabelSize(0.04)
    histo.GetYaxis().SetTitle(f'{ytitle}')

def data_prep(data='', betamin=0.6, betamax=0.7, outlabel='', verbose=False): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for data preparation
    '''
    #
    # Load data
    #_____________________________________________
    df = uproot.open(data)['ITStreeML'].arrays(library='pd')

    #
    # Adding columns in dataframe
    #_____________________________________________
    for i in range(7):
        df[f'ClSizeL{i}'] = np.where(df[f'ClSizeL{i}'] < 0, float("nan"), df[f'ClSizeL{i}'])
    df['L'] = np.arctan(df['tgL'])
    df['nSigmaPiAbs'] = abs(df['nSigmaPi'])
    df['nSigmaKAbs'] = abs(df['nSigmaK'])
    df['nSigmaPAbs'] = abs(df['nSigmaP'])
    df.eval('mean_patt_ID = (ClPattIDL0 + ClPattIDL1 + ClPattIDL2 + ClPattIDL3 + ClPattIDL4 + ClPattIDL5 + ClPattIDL6)/7', inplace=True)
    presel = "ClSizeL0 >= 0 and ClSizeL1 >= 0 and ClSizeL2 >= 0 and ClSizeL3 >= 0 and ClSizeL4 >= 0 and ClSizeL5 >= 0 and ClSizeL6 >= 0 and SnPhiL0 >= -1 and SnPhiL1 >= -1 and SnPhiL2 >= -1 and SnPhiL3 >= -1 and SnPhiL4 >= -1 and SnPhiL5 >= -1 and SnPhiL6 >= -1 and  0.05 < p"
    df_sel = df.query(presel)

    #
    # Particle species df preparation
    #_____________________________________________
    # pi
    df_pi = df_sel.query(f'{piTag}', inplace=False)
    df_pi = pd.concat([df_pi, df_sel.query('nSigmaPiAbs < 1 and p > 0.7')])
    df_pi.eval(f'beta = p/sqrt({piMass}**2 + p**2)',  inplace=True)
    df_pi['label'] = 0
    # k
    df_K = df_sel.query(f'{kTag}', inplace=False)
    df_K = pd.concat([df_K, df_sel.query('nSigmaKAbs < 1 and p > 0.7')])
    df_K.eval(f'beta = p/sqrt({kMass}**2 + p**2)',  inplace=True)
    df_K['label'] = 1
    # p
    df_P = df_sel.query(f'{pTag}', inplace=False)
    df_P = pd.concat([df_P, df_sel.query('nSigmaPAbs < 1 and p > 0.7')])
    df_P.eval(f'beta = p/sqrt({pMass}**2 + p**2)',  inplace=True)
    df_P['label'] = 2

    #
    # Plotting training variables
    #_____________________________________________
    print('Plotting variable distribution for pions and protons')
    c1 = TCanvas("c1", "", 1200, 1200)
    c1.Divide(5, 4)

    Vars = ["ClSizeL0", "ClSizeL1", "ClSizeL2",
            "ClSizeL3", "ClSizeL4", "ClSizeL5",
            "ClSizeL6", "meanClsize",
            "SnPhiL0", "SnPhiL1", "SnPhiL2",
            "SnPhiL3", "SnPhiL4", "SnPhiL5", "SnPhiL6",  "tgL",
            "p", "mean_patt_ID"]
    outFile = TFile(f'TrainVarsPlot{outlabel}.root', 'recreate')

    leg = TLegend(0.4,0.4,0.8,0.8)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.1)
    latKS = TLatex()
    latKS.SetNDC()
    latKS.SetTextSize(0.06)
    latKS.SetTextFont(42)
    latKS.SetTextColor(kBlack)

    df_pi_sel = df_pi.query(f'{betamin} < beta < {betamax}')
    df_p_sel = df_P.query(f'{betamin} < beta < {betamax}')

    with alive_bar(len(Vars), title="Plotting variables") as bar:
        for i, _ in enumerate(Vars):
            pad = c1.cd(i+1)
            pad.SetLogy()
            if 0 <= i <= 7: # ClSize & meanClsize
                minvar = 0.5
                maxvar = 10.5
                bins = 10
                latX = 0.2
                latY = 0.2
            elif (i > 7 and i <= 15): # SnPhi & tgL
                minvar = -1.
                maxvar = 1.
                bins = 2000
                latX = 0.15
                latY = 0.85
            elif i == 16:  # p
                minvar = 0.
                maxvar = 2.
                bins = 2000
                latX = 0.7
                latY = 0.8
            else:   # mean_patt_ID
                minvar = min(min(df_pi[Vars[i]]), min(df_P[Vars[i]]))
                maxvar = max(max(df_pi[Vars[i]]), max(df_P[Vars[i]]))
                bins = int((maxvar - minvar)*100)
                latX = 0.7
                latY = 0.8
            hVarpi = TH1F(f'h{Vars[i]}_pi', f';{Vars[i]}; counts', bins, minvar, maxvar)
            hVarp = TH1F(f'h{Vars[i]}_p', f';{Vars[i]}; counts', bins, minvar, maxvar)
            SetHistStyle(hVarpi, colors[0], markers[0], xtitle=f'{Vars[i]}', ytitle='counts')
            SetHistStyle(hVarp, colors[2], markers[2], xtitle=f'{Vars[i]}', ytitle='counts')
            for pi in df_pi_sel[f'{Vars[i]}']:
                hVarpi.Fill(pi)
            for p in df_p_sel[f'{Vars[i]}']:
                hVarp.Fill(p)
            KStest = hVarpi.KolmogorovTest(hVarp) # KS test
            if verbose:
                print(f'Kolmogorov-Smirnov test for pions and protons - variable: {Vars[i]} --> KS-pion: {hVarpi.KolmogorovTest(hVarp)} - KS-proton: {hVarp.KolmogorovTest(hVarpi)}\n')
            hVarpi.DrawNormalized('e')
            hVarp.DrawNormalized('esame')
            latKS.DrawLatex(latX, latY, f'KS={round(KStest, 4)}')
            if i+1 == len(Vars):
                leg.SetHeader(f'{betamin} < #beta < {betamax}')
                leg.AddEntry(hVarpi,"#pi","l")
                leg.AddEntry(hVarp,"p","l")
                c1.cd(len(Vars)+1)
                leg.Draw()
            c1.Update()
            hVarpi.Write()
            hVarp.Write()
            bar()
    c1.Write()
    outFile.Close()
    for format in ('png', 'pdf'):
        c1.SaveAs(f'TrainVarsPlot{outlabel}.{format}')
    print('Training variables plot completed.\n')

def main():
    data = '/data/fmazzasc/its_data/TreePID_505658.root'
    outlabel = ''
    data_prep(data, betamin=0.6, betamax=0.7, outlabel=outlabel)
    input('Press enter to exit')
    sys.exit()

main()