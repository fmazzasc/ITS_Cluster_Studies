'''
python script to produce training variables distributions
run: python PlotTrainingVars.py
'''
from ROOT import TFile, TH1F, TCanvas, TMath, TLegend, kRainBow, kBlack, kRed, kAzure, kOrange, kSpring, kOpenCircle, kFullCross, kFullSquare, TLatex # pylint: disable=import-error,no-name-in-module
import sys
import numpy as np
import pandas as pd
import uproot
from alive_progress import alive_bar
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.AnalysisUtils import ComputeRatioDiffBins

colors = [kBlack, kAzure+2, kRed+1, kOrange+1]
markers = [kOpenCircle, kFullSquare]


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

def MCProcess(procId):
    '''
    Method to get MC process
    '''
    if procId == 0:
        return 'Primary'
    elif procId == 4:
        return 'MultipleScattering'
    elif procId == 2:
        return 'Energyloss'
    elif procId == 3:
    kPPrimary = 0, kPMultipleScattering = 1, kPCoulombScattering = 45, kPEnergyLoss = 2,
    kPMagneticFieldL = 3, kPDecay = 4, kPPair = 5, kPCompton = 6,
    kPPhotoelectric = 7, kPBrem = 8, kPDeltaRay = 9, kPAnnihilation = 10,
    kPAnnihilationRest = 11, kPAnnihilationFlight = 12, kPHadronic = 13, kPEvaporation = 14,
    kPNuclearFission = 15, kPNuclearAbsorption = 16, kPPbarAnnihilation = 17, kPNbarAnnihilation = 18,
    kPNCapture = 19, kPHElastic = 20, kPHIElastic = 21, kPHCElastic = 22,
    kPHInhelastic = 23, kPPhotonInhelastic = 24, kPMuonNuclear = 25, kPElectronNuclear = 26,
    kPPositronNuclear = 27, kPPhotoNuclear = 46, kPTOFlimit = 28, kPPhotoFission = 29,
    kPRayleigh = 30, kPNull = 31, kPStop = 32, kPLightAbsorption = 33,
    kPLightDetection = 34, kPLightScattering = 35, kPLightWLShifting = 48, kStepMax = 36,
    kPCerenkov = 37, kPFeedBackPhoton = 38, kPLightReflection = 39, kPLightRefraction = 40,
    kPSynchrotron = 41, kPScintillation = 42, kPTransitionRadiation = 49, kPTransportation = 43,
    kPUserDefined = 47, kPNoProcess = 44 


def main():
    #----------------------------------------------------------------
    data = '/home/spolitan/Analyses/ITS_Cluster_Studies/macros/outFileMCid_thr0_1107_MCtree_nigthly.root'
    outlabel = 'PROVAMCORIGIN'
    query = ''
    outFile = TFile(f'MCOriginStudy{outlabel}.root', 'recreate')
    Vars = [] # if left empty consider all the vars
    #----------------------------------------------------------------

    df = uproot.open(data)['MCtree'].arrays(library='pd')
    df_sel = df

    if query != '':
        df_sel = df_sel.query(query, inplace=False)
    df_sel_proc = []

    for idProcess in df_sel['ProcessID'].unique():
        df_sel_proc.append(df_sel.query(f'ProcessID == {idProcess}'))

    if not Vars:
        Vars = df.keys()
    binning, mins, maxs = ([] for i in range(3))
    for var in Vars:
        if var == 'eta':
            binning.append(100)
            mins.append(-3.14)
            maxs.append(3.14)

        if var == 'phi':
            binning.append(100)
            mins.append(0)
            maxs.append(6.28)

        if var == 'CLsize':
            binning.append(100)
            mins.append(-0.5)
            maxs.append(99.5)
        else:
            binning.append(100)
            mins.append(0.)
            maxs.append(100)

    with alive_bar(len(Vars), title="Plotting variables") as bar:
        for i, (var, bins, minvar, maxvar) in enumerate(zip(Vars, binning, mins, maxs)):
            hVar = TH1F(f'h{var}', f';{var}; counts', bins, minvar, maxvar)
            hVarSel = TH1F(f'h{var}Sel', f';{var}; counts', bins, minvar, maxvar)
            hVarSel_dray = TH1F(f'h{var}Sel_dray', f';{var}; counts', bins, minvar, maxvar)
            hVarSel_proc = []
            for j, proc in enumerate(df_sel_proc):
                hVarSel_proc.append(TH1F(f'h{var}Sel_proc{proc}', f';{var}; counts', bins, minvar, maxvar))
    
            # plot
            c1 = TCanvas(f"c{var}", "", 1800, 1200)
            for i in (df[f'{var}']):
                hVar.Fill(i)
            for i, dfproc in enumerate(df_sel_proc):
                for j in (dfproc[f'{var}']):
                    hVarSel_proc[i].Fill(j)
            SetHistStyle(hVar, colors[0], markers[0], xtitle=f'{var}', ytitle='counts')
            SetHistStyle(hVarSel, colors[1], markers[1], xtitle=f'{var}', ytitle='counts')
            SetHistStyle(hVarSel_dray, colors[2], markers[1], xtitle=f'{var}', ytitle='counts')
            for i, h in enumerate(hVarSel_proc):
                SetHistStyle(h, kRainBow+i*5, kOpenCircle, xtitle=f'{var}', ytitle='counts')
            hVar.Draw('esame')
            hVarSel.Draw('esame')
            for i, h in enumerate(hVarSel_proc):
                h.Draw('esame')
                h.Write()
            c1.Write()
        
            # ratio
            c2 = TCanvas(f"c{var}_ratio", "", 800, 800)
            c2.cd()
            hRatio = []
            for i, h in enumerate(hVarSel_proc):
                hRatio.append(ComputeRatioDiffBins(h, hVarSel, 'B'))
                hRatio[i].SetDirectory(0)
                hRatio[i].SetName(f'h{var}Sel_proc{i}_ratio')
                hRatio[i].Draw('esame')
                hRatio[i].Write()
            c2.Write()
            bar()

    outFile.Write()
    input('Press enter to exit')
    sys.exit()

main()
