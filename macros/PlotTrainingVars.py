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

def data_prep(data = '', betamin = 0.6, betamax = 0.7, SnPhiMin = 0., SnPhiMax = 0.2, outlabel = '', verbose = False): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for data preparation
    '''
    # Load data
    df = uproot.open(data)['ITStreeML'].arrays(library='pd')
    if verbose:
        print(f'Loading data from {data}')
        print(f'Data loaded: {df.keys()}')

    # Adding columns in dataframe
    for i in range(7):
        df[f'ClSizeL{i}'] = np.where(df[f'ClSizeL{i}'] < 0, float("nan"), df[f'ClSizeL{i}'])
    df['L'] = np.arctan(df['tgL'])
    df['nSigmaPiAbs'] = abs(df['nSigmaPi'])
    df['nSigmaKAbs'] = abs(df['nSigmaK'])
    df['nSigmaPAbs'] = abs(df['nSigmaP'])
    df.eval('mean_patt_ID = (PattIDL0 + PattIDL1 + PattIDL2 + PattIDL3 + PattIDL4 + PattIDL5 + PattIDL6)/7', inplace=True)
    presel = "ClSizeL0 >= 0 and ClSizeL1 >= 0 and ClSizeL2 >= 0 and ClSizeL3 >= 0 and ClSizeL4 >= 0 and ClSizeL5 >= 0 and ClSizeL6 >= 0 and SnPhiL0 >= -1 and SnPhiL1 >= -1 and SnPhiL2 >= -1 and SnPhiL3 >= -1 and SnPhiL4 >= -1 and SnPhiL5 >= -1 and SnPhiL6 >= -1 and  0.05 < p"
    if verbose:
        print(f'Preselection: {presel}')
    df_sel = df.query(presel)

    # Particle species df preparation
    #_____________________________________________
    # pi
    if verbose:
        print('Preparing pi df')
        print(f'piTag: {piTag}')
    df_pi = df_sel.query(f'{piTag}', inplace=False)
    df_pi = pd.concat([df_pi, df_sel.query('nSigmaPiAbs < 1 and p > 0.7')])
    df_pi.eval(f'beta = p/sqrt({piMass}**2 + p**2)',  inplace=True)
    df_pi['label'] = 0
    # k
    if verbose:
        print('Preparing k df')
        print(f'kTag: {kTag}')
    df_K = df_sel.query(f'{kTag}', inplace=False)
    df_K = pd.concat([df_K, df_sel.query('nSigmaKAbs < 1 and p > 0.7')])
    df_K.eval(f'beta = p/sqrt({kMass}**2 + p**2)',  inplace=True)
    df_K['label'] = 1
    # p
    if verbose:
        print('Preparing p df')
        print(f'pTag: {pTag}')
    df_P = df_sel.query(f'{pTag}', inplace=False)
    df_P = pd.concat([df_P, df_sel.query('nSigmaPAbs < 1 and p > 0.7')])
    df_P.eval(f'beta = p/sqrt({pMass}**2 + p**2)',  inplace=True)
    df_P['label'] = 2

    #
    # Plotting training variables
    #_____________________________________________
    if verbose:
        print('Plotting variable distribution for pions and protons')
    c1 = TCanvas("c1", "", 1800, 1200)
    c1.Divide(7, 5)
    outFile = TFile(f'TrainVarsPlot{outlabel}.root', 'recreate')

    # Training variables
    Vars = ["ClSizeL0", "ClSizeL1", "ClSizeL2", "ClSizeL3", "ClSizeL4", "ClSizeL5", "ClSizeL6",
            "PattIDL0", "PattIDL1", "PattIDL2", "PattIDL3", "PattIDL4", "PattIDL5", "PattIDL6",
            "TanLamL0", "TanLamL1", "TanLamL2", "TanLamL3", "TanLamL4", "TanLamL5", "TanLamL6",
            "SnPhiL0", "SnPhiL1", "SnPhiL2", "SnPhiL3", "SnPhiL4", "SnPhiL5", "SnPhiL6",  
            "meanClsize", "tgL", "mean_patt_ID", "p"] # maintain order of plotting for consistency

    # Legend and labels
    leg = TLegend(0.4,0.4,0.8,0.8)
    leg.SetTextSize(0.1)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    latKS = TLatex()
    latKS.SetNDC()
    latKS.SetTextSize(0.06)
    latKS.SetTextFont(42)
    latKS.SetTextColor(kBlack)

    # Beta preselection
    if verbose:
        print(f'Preselection: {betamin} < beta < {betamax}')
    df_pi_sel = df_pi.query(f'{betamin} < beta < {betamax}')
    df_p_sel = df_P.query(f'{betamin} < beta < {betamax}')

    # Plotting
    #_____________________________________________
    with alive_bar(len(Vars), title="Plotting variables") as bar:
        for i, _ in enumerate(Vars):
            pad = c1.cd(i+1)
            pad.SetLogy()
            pad.SetRightMargin(0.005)
            pad.SetLeftMargin(0.1)
            pad.SetTopMargin(0.05)
            pad.SetBottomMargin(0.1)
            pad.SetTickx()

            # Binnning and histograms
            if 'ClSize' in Vars[i]: # ClSize
                minvar = 0.5
                maxvar = 10.5
                bins = 10
                latX = 0.2
                latY = 0.2
            elif ('SnPhi' in Vars[i]): # SnPhi
                minvar = -1.
                maxvar = 1.
                bins = 2000
                latX = 0.25
                latY = 0.85
            elif ('TanLam' in Vars[i]): # TanLam
                minvar = -1
                maxvar = 1
                bins = 100
                latX = 0.25
                latY = 0.85
            elif ('tgL' in Vars[i]): # tgL
                minvar = -1
                maxvar = 1
                bins = 100
                latX = 0.25
                latY = 0.85
            elif 'p' == Vars[i]: # p
                minvar = 0.
                maxvar = 2.
                bins = 2000
                latX = 0.7
                latY = 0.8
            else:   # patt_ID
                minvar = min(min(df_pi[Vars[i]]), min(df_P[Vars[i]]))
                maxvar = max(max(df_pi[Vars[i]]), max(df_P[Vars[i]]))
                bins = int((maxvar - minvar)*10)
                latX = 0.6
                latY = 0.8
            hVarpi = TH1F(f'h{Vars[i]}_pi', f';{Vars[i]}; counts', bins, minvar, maxvar)
            hVarp = TH1F(f'h{Vars[i]}_p', f';{Vars[i]}; counts', bins, minvar, maxvar)
            SetHistStyle(hVarpi, colors[0], markers[0], xtitle=f'{Vars[i]}', ytitle='counts')
            SetHistStyle(hVarp, colors[2], markers[2], xtitle=f'{Vars[i]}', ytitle='counts')


            # Apply selections and fill histograms
            if ('ClSizeL' in Vars[i]) or ('PattIDL' in Vars[i]) or ('TanLamL' in Vars[i]) or ('SnPhiL' in Vars[i]):
                j = int(Vars[i][-1])
                if verbose:
                    print(f'Applying SnPhi selection ({SnPhiMin} < SnPhi{j} < {SnPhiMax}) and fill histograms for {Vars[i]}')
                df_pi_sel_var = df_pi_sel.query(f'{SnPhiMin}< SnPhiL{j} < {SnPhiMax}', inplace=False)
                df_p_sel_var = df_p_sel.query(f'{SnPhiMin}< SnPhiL{j} < {SnPhiMax}', inplace=False)
            else:
                df_pi_sel_var = df_pi_sel
                df_p_sel_var = df_p_sel

            for pi in (df_pi_sel_var[f'{Vars[i]}']):
                hVarpi.Fill(pi)
            for p in df_p_sel_var[f'{Vars[i]}']:
                hVarp.Fill(p)

            # K-S test
            KStest = hVarpi.KolmogorovTest(hVarp)
            if verbose:
                print(f'Kolmogorov-Smirnov test for pions and protons - variable: {Vars[i]} --> KS-pion: {hVarpi.KolmogorovTest(hVarp)} - KS-proton: {hVarp.KolmogorovTest(hVarpi)}\n')
            
            # Drawing and saving histograms
            hVarpi.DrawNormalized('esame')
            hVarp.DrawNormalized('esame')
            latKS.DrawLatex(latX, latY, f'KS={round(KStest, 4)}')
            if i+1 == len(Vars):
                leg.SetHeader(f'{betamin} < #beta < {betamax}')
                leg.AddEntry(hVarpi,"#pi","lp")
                leg.AddEntry(hVarp,"p","lp")
                c1.cd(len(Vars)+1)
                leg.Draw()
            c1.Update()
            hVarpi.Write()
            hVarp.Write()
            bar()
    c1.Write()
    outFile.Close()

    # Saving png and pdf files
    for format in ('png', 'pdf'):
        if verbose:
            print(f'Saving plot in {format} format')
        c1.SaveAs(f'TrainVarsPlot{outlabel}.{format}')
    if verbose:
        print('Training variables plot completed.\n')

def main():
    data = '/data/fmazzasc/its_data/TreeITSClusters505658.root'
    outlabel = 'test'
    data_prep(data, betamin=0.7, betamax=0.8, outlabel=outlabel)
    input('Press enter to exit')
    sys.exit()

main()
