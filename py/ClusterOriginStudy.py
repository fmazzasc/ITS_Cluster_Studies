'''
python script to study the origin of the clusters in MC
run: python ClusterOriginStudy.py
'''
from ROOT import TFile, TH1F, TH2F, TCanvas, TMath, TLegend, kRainBow, kBlack, kRed, kPink, kAzure, kOrange, kSpring, kOpenCircle, kFullCross, kFullSquare, TLatex, kMagenta, kCyan # pylint: disable=import-error,no-name-in-module
import sys
import numpy as np
import pandas as pd
import uproot
from alive_progress import alive_bar
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.AnalysisUtils import ComputeRatioDiffBins, MCProcess

markers = [kOpenCircle, kFullSquare]

def main():
    #----------------------------------------------------------------
    data = '/home/spolitan/Analyses/ITS_Cluster_Studies/macros/outFileMCid_thr0_1207_MCtree_morning.root'
    outlabel = 'globalMC_wcut_1207'
    query = ''
    outFile = TFile(f'MCOriginStudy{outlabel}.root', 'recreate')
    Vars = [] # if left empty consider all the vars
    enabledProcesses = ['d-rays', 'Primary', 'HInhelastic', 'Hadronic', 'PositronNuclear', 'ElectronNuclear', 'Pair']
    doLayerStudy = True # if true, study L0 and L6 clusters
    doLayerFracStudy = False # if true, study the fraction of d-rays on each layer and close 
    #----------------------------------------------------------------

    # read data
    df = uproot.open(data)['MCtree'].arrays(library='pd')
    df_sel = df
    df_sel['E_mev'] = df_sel['E'] * 1000
 
    if query != '':
        print(f'\033[1m\033[93mApplying query: {query}\033[0m')
        df_sel = df_sel.query(query, inplace=False)

    df_sel_proc = []
    labels = []
    colors = [kAzure+4, kRed+1, kSpring+3, kOrange+1, kPink+1, kCyan+1, kMagenta+1]

    print(f'\033[1mEnabled processes: {enabledProcesses}\033[0m')
    for i, idProcess in enumerate(df_sel['ProcessID'].unique()):
        label = MCProcess(idProcess)
        if label not in enabledProcesses:
            continue
        labels.append(label)
        df_sel_proc.append(df_sel.query(f'ProcessID == {idProcess}'))

    # delta rays energy distribution
    subdir = outFile.mkdir('EkinStudy')
    outFile.cd('EkinStudy')
    if 'd-rays' in enabledProcesses:
        df_kin_study = df_sel_proc[labels.index('d-rays')]
        print(f'\033[1mEkin study for d-rays: \033[0m')
        hEkin = TH1F('hEkin', 'hEkin;E_{tot} [MeV]; Events', 1000, 0, 1)
        hEtotal = TH1F('hEtotal', 'hEtotal;E_{kin} [MeV];Events', 1000, 0, 1)
        hEkin_Clsize_corr = [TH2F(f'hEkin_Clsize_corr_L{i}',
                                  f'hEkin_Clsize_corr_L{i};E_{{kin}} [MeV];Cluster size',
                                  1000, 0, 1, 100, 0, 100) for i in range(7)]

        for i, (E_mev, clsize, layer) in enumerate(zip(df_kin_study['E_mev'],
                                                       df_kin_study['CLsize'],
                                                       df_kin_study['Layer'])):
            hEtotal.Fill(E_mev)
            Ekin = E_mev - 0.511 # electron mass
            hEkin.Fill(Ekin)
            hEkin_Clsize_corr[layer].Fill(Ekin, clsize)
        hEtotal.Write()
        hEkin.Write()
        for h in hEkin_Clsize_corr:
            h.Write()
    else:
        print(f'Ekin study implemented only for d-rays. Make sure d-rays are the first process in the list of enabled processes!')
    outFile.cd()

    if doLayerFracStudy:
        print(f'\033[1mLayer fraction study\033[0m')
        subdir = outFile.mkdir('LayerFracStudy')
        if enabledProcesses == ['d-rays']:
            hLayers = TH1F('hLayers', 'hLayers', 7, -0.5, 6.5)
            hLayersOver40 = TH1F('hLayersOver40', 'hLayersOver40', 7, -0.5, 6.5)
            for i, (layer, clsize) in enumerate(zip(df_sel_proc[0]['Layer'], df_sel_proc[0]['CLsize'])):
                hLayers.Fill(layer)
                if clsize > 40:
                    hLayersOver40.Fill(layer)
            #SetHistStyle(hLayers, kBlack, kOpenCircle, 'Layer', '#delta-rays per layer')
            #SetHistStyle(hLayersOver40, kRed, kOpenCircle, 'Layer', '#delta-rays > 40 per layer')
            hLayers.Write()
            hLayersOver40.Write()
            outFile.Write()
        else:
            print(f'Layer fraction study not implemented for {enabledProcesses}. Continue.')

    print(f'\033[1mEnabled variables: {Vars}\033[0m')
    if not Vars:
        Vars = df.keys()
    binning, mins, maxs = ([] for i in range(3))
    for var in Vars:
        if var == 'eta':
            binning.append(100)
            mins.append(-3.14)
            maxs.append(3.14)

        elif var == 'phi':
            binning.append(100)
            mins.append(0)
            maxs.append(6.28)

        elif var == 'X' or var == 'Y' or var == 'Z':
            binning.append(100)
            mins.append(-100)
            maxs.append(100)

        elif var == 'CLsize':
            binning.append(100)
            mins.append(0.)
            maxs.append(100)

        elif var == 'Layer':
            binning.append(7)
            mins.append(-0.5)
            maxs.append(6.5)

        elif var == 'p':
            binning.append(100)
            mins.append(0)
            maxs.append(1)

        else:
            binning.append(100)
            mins.append(0.)
            maxs.append(100)

    with alive_bar(len(Vars), title="Plotting variables") as bar:
        for i, (var, bins, minvar, maxvar) in enumerate(zip(Vars, binning, mins, maxs)):
            print(f'\033[1mPlotting variable: {var}\033[0m')
            hVar = TH1F(f'h{var}', f';{var}; counts', bins, minvar, maxvar)

            if doLayerStudy and var == 'CLsize':
                hCL_vs_layer = TH2F('hCL_vs_layer', 'hCL_vs_layer; Layer; Cluster size', 7, -0.5, 6.5, 100, 0, 100)
                hCL = [TH1F(f'hCL{i}', f'; Cluster size; Counts', 100, 0, 100) for i in range(7)]

            if 'X' in Vars and 'Y' in Vars and 'Z' in Vars and var == 'X':
                hXY = TH2F(f'hXY', ';X (cm);Y (cm)', 1200, -60, 60, 1200, -60, 60)
                hXZ = TH2F(f'hXZ', ';X (cm);Z (cm)', 1200, -60, 60, 1200, -60, 60)
                hYZ = TH2F(f'hYZ', ';Y (cm);Z (cm)', 1200, -60, 60, 1200, -60, 60)

            hVarSel_proc, hCL_proc = [], [] # list of TH1F for each process
            hXY_proc, hXZ_proc, hYZ_proc = [], [], [] # list of TH2F for each process
            for k, label in enumerate(labels):
                hVarSel_proc.append(TH1F(f'h{var}Sel_proc{label}', f';{var}; counts', bins, minvar, maxvar))
                if 'X' in Vars and 'Y' in Vars and 'Z' in Vars and var == 'X':
                    hXY_proc.append(TH2F(f'hXY_proc{label}', ';X (cm);Y (cm)', 1200, -60, 60, 1200, -60, 60))
                    hXZ_proc.append(TH2F(f'hXZ_proc{label}', ';X (cm);Z (cm)', 1200, -60, 60, 1200, -60, 60))
                    hYZ_proc.append(TH2F(f'hYZ_proc{label}', ';Y (cm);Z (cm)', 1200, -60, 60, 1200, -60, 60))
                if doLayerStudy and var == 'CLsize':
                    hCL_proc.append([TH1F(f'hCL{i}proc{label}', f'hCL_L{i}proc{label}; Cluster size', 100, 0, 100) for i in range(7)])
                    SetObjectStyle(hCL_proc[k][i], linecolor=colors[k], fillcolor=colors[k], markerstyle=20, markercolor=colors[k], markersize=0.5)
    
            # filling histograms
            for i in (df_sel[f'{var}']): # loop over all events
                hVar.Fill(i)

            if 'X' in Vars and 'Y' in Vars and 'Z' in Vars and var == 'X':
                for i, (x, y, z) in enumerate(zip(df_sel['X'], df_sel['Y'], df_sel['Z'])):
                    hXY.Fill(x, y)
                    hXZ.Fill(x, z)
                    hYZ.Fill(y, z)
                for i, dfproc in enumerate(df_sel_proc):
                    for j, (x, y, z) in enumerate(zip(dfproc['X'], dfproc['Y'], dfproc['Z'])):
                        hXY_proc[i].Fill(x, y)
                        hXZ_proc[i].Fill(x, z)
                        hYZ_proc[i].Fill(y, z)

            if doLayerStudy and var == 'CLsize':
                for i, (clsize, layer) in enumerate(zip(df_sel[f'{var}'], df_sel['Layer'])):
                    hCL[layer].Fill(clsize)
                    hCL_vs_layer.Fill(layer, clsize)
                for i, dfproc in enumerate(df_sel_proc):
                    for j, (clsize, layer) in enumerate(zip(dfproc[f'{var}'], dfproc['Layer'])):
                        hCL_proc[i][layer].Fill(clsize)
                cClvsLayer_contrib = []
                hRatio = []
                leg = TLegend(0.5, 0.6, 0.7, 0.8)
                leg.SetBorderSize(0)
                for i in range(7):
                    hRatio.append([])
                    cClvsLayer_contrib.append(TCanvas(f"cClvsLayer{i}_contrib", "", 1600, 900))
                    cClvsLayer_contrib[i].Divide(2, 1)
                    cClvsLayer_contrib[i].cd(1).DrawFrame(0, 0, 100, 100, '; Cluster size L; Counts')
                    cClvsLayer_contrib[i].cd(1).SetLogy()
                    SetObjectStyle(hCL[i], linecolor=kBlack, fillcolor=kBlack, marker=kOpenCircle, fillalpha=0.2)
                    hCL[i].SetStats(0)
                    hCL[i].Draw('hist')
                    if i == 0:
                        leg.AddEntry(hCL[i], 'All', 'l')
                    for j, h in enumerate(hCL_proc):
                        SetObjectStyle(h[i], linecolor=colors[j], markercolor=colors[j],
                                       fillcolor=colors[j], marker=kOpenCircle, fillalpha=0.2)
                        h[i].Draw('samehist')
                        if i == 0:
                            leg.AddEntry(h[i], labels[j], 'l')
                        proc = labels[j]
                        hRatio[i].append(h[i].Clone(f'hRatio{i}proc{proc}'))
                        hRatio[i][j].Divide(hCL[i])
                        SetObjectStyle(hRatio[i][j], linecolor=colors[j], markercolor=colors[j],
                                       fillcolor=colors[j], marker=kOpenCircle, fillalpha=0.2)
                    leg.Draw('same')
                    cClvsLayer_contrib[i].cd(2).SetLogy()
                    cClvsLayer_contrib[i].cd(2).DrawFrame(0, 0.001, 100, 2.0, '; Cluster size; Ratio')
                    for j, h in enumerate(hRatio[i]):
                        h.Draw('samehist')
                    
            leg_proc = TLegend(0.5, 0.6, 0.7, 0.8)
            leg_proc.SetBorderSize(0)
            leg_proc.AddEntry(hVar, 'All', 'l')
            for i, dfproc in enumerate(df_sel_proc): # loop over all processes
                for j in (dfproc[f'{var}']): # loop over all events
                    hVarSel_proc[i].Fill(j)
                leg_proc.AddEntry(hVarSel_proc[i], labels[i], 'l')

            SetObjectStyle(hVar, markercolor=kBlack, marker=kOpenCircle, fillalpha=0.5, linewidth=2, fillcolor=kBlack)
            for i, h in enumerate(hVarSel_proc):
                SetObjectStyle(h, color=colors[i], fillcolor=colors[i], fillalpha=0.5, linewidth=2, marker=kOpenCircle, markercolor=colors[i])

            outFile.mkdir(f'{var}') if not outFile.Get(f'{var}') else None
            outFile.cd(f'{var}')
            c1 = TCanvas(f"c{var}", "", 1800, 1200)
            hVar.Draw('histesame')
            hVar.Write()
            for i, h in enumerate(hVarSel_proc):
                h.Draw('histesame')
                h.Write()
            leg_proc.Draw('same')
            c1.Write()
            outFile.cd('../')

            if 'X' in Vars and 'Y' in Vars and 'Z' in Vars and var == 'X':
                outFile.mkdir('space_correlation') if not outFile.GetDirectory('space_correlation') else None
                outFile.cd('space_correlation')
                hXY.Write()
                hXZ.Write()
                hYZ.Write()
                for i, (hXY, hXZ, hYZ) in enumerate(zip(hXY_proc, hXZ_proc, hYZ_proc)):
                    hXY.Write()
                    hXZ.Write()
                    hYZ.Write()
                outFile.cd('../')

            if doLayerStudy and var == 'CLsize':
                outFile.mkdir('CL_vs_layer') if not outFile.GetDirectory('CL_vs_layer') else None
                outFile.cd('CL_vs_layer')
                hCL_vs_layer.Write()
                for i, h in enumerate(hCL):
                    h.Write()
                for i, hproc in enumerate(hCL_proc):
                    for j, h in enumerate(hproc):
                        h.Write()
                outFile.mkdir('CL_vs_layer/contributions') if not outFile.GetDirectory('CL_vs_layer/contributions') else None
                outFile.cd('CL_vs_layer/contributions')
                for i, h in enumerate(hRatio):
                    for j, h in enumerate(h):
                        h.Write()
                for i in range(7):
                    cClvsLayer_contrib[i].Write()
                    cClvsLayer_contrib[i].SaveAs(f'cClvsLayer{i}_contrib.png')
                outFile.cd('../../')
            bar()

    outFile.Close()
    input('Press enter to exit')
    sys.exit()

main()
