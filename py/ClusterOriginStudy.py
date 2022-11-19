'''
python script to produce training variables distributions
run: python PlotTrainingVars.py
'''
from ROOT import TFile, TH1F, TH2F, TCanvas, TMath, TLegend, kRainBow, kBlack, kRed, kAzure, kOrange, kSpring, kOpenCircle, kFullCross, kFullSquare, TLatex # pylint: disable=import-error,no-name-in-module
import sys
import numpy as np
import pandas as pd
import uproot
from alive_progress import alive_bar
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.AnalysisUtils import ComputeRatioDiffBins, MCProcess

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

def main():
    #----------------------------------------------------------------
    data = '/home/fmazzasc/alice/ITS_Cluster_Studies/macros/outFileMCid_fmazz.root'
    outlabel = 'FracDraysNewCut181122'
    query = ''
    outFile = TFile(f'MCOriginStudy{outlabel}.root', 'recreate')
    Vars = ['CLsize'] # if left empty consider all the vars
    enabledProcesses = ['d-rays'] #, 'Primary', 'HInhelastic', 'Hadronic', 'PositronNuclear', 'ElectronNuclear', 'Pair']
    doLayerStudy = False # if true, study L0 and L6 clusters
    doEkinStudy = True # if true, study Ekin distssribution and CL0, CL6 of d-rays and close
    doLayerFracStudy = True # if true, study the fraction of d-rays on each layer and close 
    #----------------------------------------------------------------

    # read data
    df = uproot.open(data)['MCtree'].arrays(library='pd')
    df_sel = df
    df_sel['E_mev'] = df_sel['E'] * 1000
 
    if query != '':
        df_sel = df_sel.query(query, inplace=False)

    df_sel_proc = []
    labels = []
    print(f'\033[1mEnabled processes: {enabledProcesses}\033[0m')
    for i, idProcess in enumerate(df_sel['ProcessID'].unique()):
        label = MCProcess(idProcess)
        if label not in enabledProcesses:
            continue
        labels.append(label)
        df_sel_proc.append(df_sel.query(f'ProcessID == {idProcess}'))

    if doEkinStudy:
        if enabledProcesses == ['d-rays']:
            print(f'\033[1mEkin study for d-rays\033[0m')
            hEkin = TH1F('hEkin', 'hEkin', 1000, 0, 1)
            hEtotal = TH1F('hEtotal', 'hEtotal', 1000, 0, 1)
            hCL0 = TH1F('hCL0', 'hCL0', 100, 0, 100)
            hCL6 = TH1F('hCL6', 'hCL6', 100, 0, 100)
            hEkin_Clsize_corr_L0 = TH2F('hEkin_Clsize_corr_L0', 'hEkin_Clsize_corr_L0; Ekin (MeV); Cluster size L0', 1000, 0, 1, 100, 0, 100)
            hEkin_Clsize_corr_L6 = TH2F('hEkin_Clsize_corr_L6', 'hEkin_Clsize_corr_L6; Ekin (MeV); Cluster size L6', 1000, 0, 1, 100, 0, 100)

            for i, (E_mev, clsize, layer) in enumerate(zip(df_sel_proc[0]['E_mev'], df_sel_proc[0]['CLsize'], df_sel_proc[0]['Layer'])):
                hEtotal.Fill(E_mev)
                Ekin = E_mev - 0.5
                hEkin.Fill(Ekin)
                if layer == 0:
                    hCL0.Fill(clsize)
                    hEkin_Clsize_corr_L0.Fill(Ekin, clsize)
                if layer == 6:
                    hCL6.Fill(clsize)
                    hEkin_Clsize_corr_L6.Fill(Ekin, clsize)

            SetHistStyle(hEtotal, kBlack, kOpenCircle, 'E_{tot} [MeV]', 'Events')
            SetHistStyle(hEkin, kRed, kOpenCircle, 'E_{tot} [MeV]', 'Events')
            SetHistStyle(hCL0, kAzure+2, kOpenCircle, 'E_{tot} [MeV]', 'Events')
            SetHistStyle(hCL6, kOrange+1, kOpenCircle, 'E_{tot} [MeV]', 'Events')
            hEtotal.Write()
            hEkin.Write()
            hCL0.Write()
            hCL6.Write()
            hEkin_Clsize_corr_L0.Write()
            hEkin_Clsize_corr_L6.Write()

            outFile.Write()
            input('Press enter to exit')
            sys.exit()
        else:
            print('Ekin study not implemented for this process. Exit.')
            sys.exit()

    if doLayerFracStudy:
        print(f'\033[1mLayer fraction study\033[0m')
        if enabledProcesses == ['d-rays']:
            hLayers = TH1F('hLayers', 'hLayers', 7, -0.5, 6.5)
            hLayersOver40 = TH1F('hLayersOver40', 'hLayersOver40', 7, -0.5, 6.5)
            for i, (layer, clsize) in enumerate(zip(df_sel_proc[0]['Layer'], df_sel_proc[0]['CLsize'])):
                hLayers.Fill(layer)
                if clsize > 40:
                    hLayersOver40.Fill(layer)
            SetHistStyle(hLayers, kBlack, kOpenCircle, 'Layer', '#delta-rays per layer')
            SetHistStyle(hLayersOver40, kRed, kOpenCircle, 'Layer', '#delta-rays > 40 per layer')
            hLayers.Write()
            hLayersOver40.Write()
            outFile.Write()
            input('Press enter to exit')
            sys.exit()
        else:
            print('Layer fraction study not implemented for this process. Exit.')
            sys.exit()

    print(f'\033[1mEnabled variables: {Vars}\033[0m')
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
            print(f'Plotting {var}')
            hVar = TH1F(f'h{var}', f';{var}; counts', bins, minvar, maxvar)
            if doLayerStudy and var == 'CLsize':
                hCL0 = TH1F(f'h{var}_L0', f';{var}; counts', bins, minvar, maxvar)
                hCL6 = TH1F(f'h{var}_L6', f';{var}; counts', bins, minvar, maxvar)
            hVarSel = TH1F(f'h{var}Sel', f';{var}; counts', bins, minvar, maxvar)
            hVarSel_dray = TH1F(f'h{var}Sel_dray', f';{var}; counts', bins, minvar, maxvar)
            hVarSel_proc = []
            for _, label in enumerate(labels):
                hVarSel_proc.append(TH1F(f'h{var}Sel_proc{label}', f';{var}; counts', bins, minvar, maxvar))
    
            # plot
            c1 = TCanvas(f"c{var}", "", 1800, 1200)
            for i in (df[f'{var}']):
                hVar.Fill(i)
            if doLayerStudy and var == 'CLsize':
                for i, (clsize, layer) in enumerate(zip(df[f'{var}'], df['Layer'])):
                    if layer == 0:
                        hCL0.Fill(clsize)
                    if layer == 6:
                        hCL6.Fill(clsize)
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
    if doLayerStudy:
        hCL0.Write()
        hCL6.Write()
    outFile.Write()
    input('Press enter to exit')
    sys.exit()

main()
