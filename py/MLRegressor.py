'''
python script to run regression
run: python MLRegressor.py cfgFileNameML.yml
'''
from ROOT import TFile, gStyle, TH1, TH2F, TH1F, TH1D, TCanvas, TLegend, TGraph, TGraphErrors, kRed, kBlue, kAzure, kSpring, kGreen, kOrange, kGray, kBlack, TGaxis,gPad, TLatex, kFullCircle, kFullSquare, kFullCross # pylint: disable=import-error,no-name-in-module
import os
import sys
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import uproot
import optuna
from optuna import Trial, visualization
import shap
import logging
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from hipe4ml import plot_utils
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from flaml import AutoML

gStyle.SetPalette(52)

#
# General varibales
#__________________________________
piMass = 0.13957000
kMass = 0.4937
pMass = 0.93827200

labels = ['#pi', 'K', 'p']
colors = [kAzure+2, kOrange-2, kSpring+2]
markers = [kFullSquare, kFullCircle, kFullCross]

#
# Tag conditions
#__________________________________
eTag = '-2 < nSigmaE < 1 and nSigmaK > 4 and nSigmaPi > 2 and p < 0.2'
piTag = 'nSigmaPiAbs < 1 and nSigmaKAbs > 3 and p <= 0.7'
kTag = 'nSigmaKAbs < 1 and nSigmaPiAbs > 3 and p <= 0.7'
pTag = 'nSigmaPAbs < 1 and nSigmaKAbs > 3 and nSigmaPiAbs > 3 and p <= 0.7' 

def SetHistStyle(histo, color, marker=kFullCircle, markerSize=1, xtitle='', ytitle='', style=1):
    '''
    Method to set histogram style
    '''
    histo.SetLineColor(color)
    histo.SetLineStyle(style)
    histo.SetLineWidth(2)
    histo.SetMarkerColor(color)
    histo.SetMarkerStyle(20)
    histo.SetMarkerSize(markerSize)
    histo.SetMarkerStyle(marker)
    if isinstance(histo, TH1):
        histo.SetStats(0)
    if xtitle != '':
        histo.GetXaxis().SetTitle(xtitle)
    histo.GetXaxis().SetTitleSize(0.06)
    histo.GetXaxis().SetTitleOffset(0.8)
    histo.GetXaxis().SetLabelSize(0.04)
    if ytitle != '':
        histo.GetYaxis().SetTitle(ytitle)
    histo.GetYaxis().SetTitleSize(0.05)
    histo.GetYaxis().SetTitleOffset(1.1)
    histo.GetYaxis().SetLabelSize(0.04)


def augment_particles(df_train, l_mother, l_dau, pmin, pmax):
    '''
    Method to generate candidate clones flat wrt beta and momentum
    '''
    m = [piMass, kMass, pMass]
    m_mother =  m[l_mother]
    m_dau =  m[l_dau]

    len_df_mother = len(df_train.query(f'label == {l_mother} and {pmin} < p < {pmax}'))
    len_df_dau = len(df_train.query(f'label == {l_dau} and {pmin} < p < {pmax}'))

    momentum_range_mother = [pmin*m_mother/m_dau, pmax*m_mother/m_dau]
    df_train_sel_mother = df_train.query(f'label == {l_mother} and {momentum_range_mother[0]} < p < {momentum_range_mother[1]}')

    if (len_df_mother - len_df_dau) <= 0:
        return 0
    nsamples = min(len_df_mother - len_df_dau, len(df_train_sel_mother))

    df_dau = df_train_sel_mother.sample(nsamples)
    df_dau['p'] = df_dau['p']*(m_dau/m_mother)
    df_dau['isclone'] = 1
    df_dau['label'] = l_dau

    return df_dau

def augment_particles_raw(df_train, l_mother, l_dau, bmin, bmax):
    '''
    Method to generate candidate clones in a given beta interval
    '''
    m = [piMass,  kMass, pMass]
    m_mother =  m[l_mother]
    m_dau =  m[l_dau]

    df_dau = df_train.query(f'label == {l_mother} and {bmin} < beta < {bmax}')

    df_dau['p'] = df_dau['p']*(m_dau/m_mother)
    df_dau['isclone'] = 1
    df_dau['label'] = l_dau

    return df_dau

def data_prep(inputCfg, OutPutDir, Df): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for data preparation
    '''
    input_data = inputCfg["input"]["data"]
    print(f'input data: {input_data}')
    df = uproot.open(input_data)['ITStreeML'].arrays(library='pd')

    # Adding columns in training and test
    #_____________________________________________
    for i in range(7):
        df[f'ClSizeL{i}'] = np.where(df[f'ClSizeL{i}'] < 0, float("nan"), df[f'ClSizeL{i}'])
    df['L'] = np.arctan(df['tgL'])
    #df['nSigmaEAbs'] = abs(df['nSigmaE'])
    df['nSigmaPiAbs'] = abs(df['nSigmaPi'])
    df['nSigmaKAbs'] = abs(df['nSigmaK'])
    df['nSigmaPAbs'] = abs(df['nSigmaP'])
    df.eval('mean_patt_ID = (PattIDL0 + PattIDL1 + PattIDL2 + PattIDL3 + PattIDL4 + PattIDL5 + PattIDL6)/7', inplace=True)
    df.eval('mean_SnPhi = (SnPhiL0 + SnPhiL1 + SnPhiL2 + SnPhiL3 + SnPhiL4 + SnPhiL5 + SnPhiL6)/7', inplace=True)
    presel = inputCfg['data_prep']['presel']
    df_sel = df.query(presel)

    # Application set preparation
    #_____________________________________________
    if inputCfg['input']['separateAppl']:
        print('\033[93mDifferent .root file for application and training\033[0m')
        input_data_appl = inputCfg["input"]["data_appl"]
        print(f'application input data: {input_data_appl}')
        ApplDf = uproot.open(input_data_appl)['ITStreeML'].arrays(library='pd')
        for i in range(7):
            ApplDf[f'ClSizeL{i}'] = np.where(ApplDf[f'ClSizeL{i}'] < 0, float("nan"), ApplDf[f'ClSizeL{i}'])
        ApplDf = ApplDf.query(presel)
    else:
        ApplDf = df_sel.copy()
    ApplDf.query('0 < p < 50', inplace=True)
    #ApplDf['nSigmaEAbs'] = abs(ApplDf['nSigmaE'])
    ApplDf['nSigmaPiAbs'] = abs(ApplDf['nSigmaPi'])
    ApplDf['nSigmaKAbs'] = abs(ApplDf['nSigmaK'])
    ApplDf['nSigmaPAbs'] = abs(ApplDf['nSigmaP'])
    #ApplDf['nSigmaE'] = abs(ApplDf['nSigmaE'])
    ApplDf['nSigmaPi'] = abs(ApplDf['nSigmaPi'])
    ApplDf['nSigmaK'] = abs(ApplDf['nSigmaK'])
    ApplDf['nSigmaP'] = abs(ApplDf['nSigmaP'])
    ApplDf.eval('mean_patt_ID = (PattIDL0 + PattIDL1 + PattIDL2 + PattIDL3 + PattIDL4 + PattIDL5 + PattIDL6)/7', inplace=True)
    ApplDf.eval('mean_SnPhi = (SnPhiL0 + SnPhiL1 + SnPhiL2 + SnPhiL3 + SnPhiL4 + SnPhiL5 + SnPhiL6)/7', inplace=True)

    # Particle species df preparation
    #_____________________________________________
    # e
    #df_e = df_sel.query(f'{eTag}', inplace=False)
    #df_e = pd.concat([df_e, df_sel.query('nSigmaEAbs < 1 and p > 0.7')])
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

    doPlots = inputCfg['plots']['doPlots']
    training_conf = inputCfg['data_prep']['training_conf']
    outlabel = inputCfg['output']['model_outlabel']

    if doPlots:
        print('Plotting variable distribution for pions and protons')
        c1 = TCanvas("c1", "", 1200, 1200)
        c1.Divide(4,3)
        Vars = ["ClSizeL0", "ClSizeL1", "ClSizeL2",
                "ClSizeL3", "ClSizeL4", "ClSizeL5", 
                "ClSizeL6", "meanClsize", "p", "mean_patt_ID", "tgL"]
        outFile = TFile(f'RegVars{outlabel}.root', 'recreate')
        leg = TLegend(0.4,0.4,0.8,0.8)
        leg.SetBorderSize(0)
        df_pi_sel = df_pi.query('0.6 < beta < 0.7')
        df_p_sel = df_P.query('0.6 < beta < 0.7')
        for i, v in enumerate(Vars):
            pad = c1.cd(i+1)
            pad.SetLogy()
            if 0 <= i < 8:
                minvar = 0.5
                maxvar = 10.5
                bins = 10
            elif i == 8:
                minvar = 0.1
                maxvar = 2.1
                bins = 2000
            elif i == 9:
                minvar = 0.
                maxvar = 80
                bins = 40
            else:
                minvar = min(min(df_pi[Vars[i]]), min(df_P[Vars[i]]))
                maxvar = max(max(df_pi[Vars[i]]), max(df_P[Vars[i]]))
                bins = int((maxvar - minvar)*100)
            hVarpi = TH1F(f'h{Vars[i]}_pi', f';{Vars[i]}; counts', bins, minvar, maxvar)
            hVarp = TH1F(f'h{Vars[i]}_p', f';{Vars[i]}; counts', bins, minvar, maxvar)
            SetHistStyle(hVarpi, colors[0])
            SetHistStyle(hVarp, colors[2])
            for pi in df_pi_sel[f'{Vars[i]}']:
                hVarpi.Fill(pi)
            for p in df_p_sel[f'{Vars[i]}']:
                hVarp.Fill(p)
            hVarpi.DrawNormalized('hist')
            hVarp.DrawNormalized('histsame')
            if i+1 == 11:
                leg.SetHeader('0.6 < #beta < 0.7')
                leg.AddEntry(hVarpi,"#pi","l")
                leg.AddEntry(hVarp,"p","l")
                c1.cd(12)
                leg.Draw()
            c1.Update()
            hVarpi.Write()
            hVarp.Write()
        c1.Write()
        outFile.Close()
        c1.SaveAs(f'RegVars{outlabel}.png')
        print('Training variables plot completed.\n')


    seed_split = inputCfg['data_prep']['seed_split']
    test_frac = inputCfg['data_prep']['test_fraction']
    TotDf = pd.concat([df_pi, df_K, df_P], sort=True)
    TrainSet, TestSet, yTrain, yTest = train_test_split(TotDf, TotDf['beta'], test_size=test_frac, random_state=seed_split)
    data_conf = ''

    # Data preparation configuration
    if 'equal' in training_conf:
        # same number of candidates for each species
        #_____________________________________________
        print('EQUAL CONFIGURATION SELECTED')
        nCandToKeep = min(len(df_pi), len(df_P), len(df_K))
        print(f'\033[93mSame number of candidates ({nCandToKeep}) for each class\033[0m')
        TrainSet = pd.concat([TrainSet.query('label == 0').iloc[:50000], TrainSet.query('label == 1').iloc[:nCandToKeep], TrainSet.query('label == 2').iloc[:nCandToKeep]], sort=False)
        yTrain = TrainSet['beta']
        data_conf += '_equal'

    if 'augmentation' in training_conf:
        # augement number of candidates
        #_____________________________________________
        print('AUGMENTATION CONFIGURATION SELECTED')
        betamin = inputCfg['data_prep']['betamin']
        betamax = inputCfg['data_prep']['betamax']
        TrainSet['isclone'] = 0
        df_aug_list = []
 
        df_k_pi_aug =  augment_particles_raw(TrainSet, 1, 0, betamin[1], betamax[1])
        df_p_pi_aug = augment_particles_raw(TrainSet, 2, 0, betamin[2], betamax[2])

        df_pi_k_aug = augment_particles_raw(TrainSet, 0, 1, betamin[0], betamax[0])
        df_p_k_aug = augment_particles_raw(TrainSet, 2, 1, betamin[2], betamax[2])
            
        df_pi_p_aug = augment_particles_raw(TrainSet, 0, 2, betamin[0], betamax[0])
        df_k_p_aug = augment_particles_raw(TrainSet, 1, 2, betamin[1], betamax[1])

        if type(df_k_pi_aug) != int:
            df_aug_list.append(df_k_pi_aug)
        if type(df_p_pi_aug) != int:
            df_aug_list.append(df_p_pi_aug)
        if type(df_pi_k_aug) != int:
            df_aug_list.append(df_pi_k_aug)
        if type(df_p_k_aug) != int:
            df_aug_list.append(df_p_k_aug)
        if type(df_pi_p_aug) != int:
            df_aug_list.append(df_pi_p_aug)
        if type(df_k_p_aug) != int:
            df_aug_list.append(df_k_p_aug)


        df_aug_list.append(TrainSet)
        TrainSet = pd.concat(df_aug_list)
        yTrain = TrainSet['beta']
        print('Augmented candidates in training:')
        print(f'N pi:\t{len(TrainSet.query("label == 0 and isclone == 1"))}\nN K:\t{len(TrainSet.query("label == 1 and isclone == 1"))}\nN p:\t{len(TrainSet.query("label == 2 and isclone == 1"))}\n')
        data_conf += '_augmented'
        
        if doPlots:
            print('Plotting variable distribution for pions and protons vs clones')
            c1 = TCanvas("c1", "", 1200, 1200)
            c1.Divide(4,3)
            Vars = ["ClSizeL0", "ClSizeL1", "ClSizeL2",
                    "ClSizeL3", "ClSizeL4", "ClSizeL5", 
                    "ClSizeL6", "meanClsize", "p", "mean_patt_ID", "tgL"]
            outFile = TFile(f'RegVarsClones{outlabel}.root', 'recreate')
            leg = TLegend(0.4,0.4,0.8,0.8)
            leg.SetBorderSize(0)
            df_pi_sel = TrainSet.query('label == 0 and 0.1 < p < 0.7 and isclone == 0')
            df_pi_sel_clone = TrainSet.query('label == 0 and 0.1 < p < 0.7 and isclone == 1')
            df_p_sel = TrainSet.query('label == 2 and 0.1 < p < 0.7 and isclone == 0')
            df_p_sel_clone = TrainSet.query('label == 2 and 0.1 < p < 0.7 and isclone == 1')
            for i, v in enumerate(Vars):
                pad = c1.cd(i+1)
                pad.SetLogy()
                if 0 <= i < 8:
                    minvar = 0.5
                    maxvar = 10.5
                    bins = 10
                elif i == 8:
                    minvar = 0.1
                    maxvar = 0.7
                    bins = 100
                elif i == 9:
                    minvar = 0.
                    maxvar = 80
                    bins = 40
                else:
                    minvar = min(min(df_pi[Vars[i]]), min(df_P[Vars[i]]))
                    maxvar = max(max(df_pi[Vars[i]]), max(df_P[Vars[i]]))
                    bins = int((maxvar - minvar)*100)
                hVarpi = TH1F(f'h{Vars[i]}_pi', f';{Vars[i]}; counts', bins, minvar, maxvar)
                hVarpi_clones = TH1F(f'h{Vars[i]}_pi_clones', f';{Vars[i]}; counts', bins, minvar, maxvar)
                hVarp = TH1F(f'h{Vars[i]}_p', f';{Vars[i]}; counts', bins, minvar, maxvar)
                hVarp_clones = TH1F(f'h{Vars[i]}_p_clones', f';{Vars[i]}; counts', bins, minvar, maxvar)
                SetHistStyle(hVarpi, colors[0])
                SetHistStyle(hVarpi_clones, colors[0], 2)
                SetHistStyle(hVarp, colors[2])
                SetHistStyle(hVarp_clones, colors[2], 2)
                for pi in df_pi_sel[f'{Vars[i]}']:
                    hVarpi.Fill(pi)
                for p in df_p_sel[f'{Vars[i]}']:
                    hVarp.Fill(p)
                for pi in df_pi_sel_clone[f'{Vars[i]}']:
                    hVarpi_clones.Fill(pi)
                for p in df_p_sel_clone[f'{Vars[i]}']:
                    hVarp_clones.Fill(p)
                hVarpi.DrawNormalized('hist')
                hVarpi_clones.DrawNormalized('hist')
                hVarp.DrawNormalized('histsame')
                hVarp_clones.DrawNormalized('histsame')
                if i+1 == 11:
                    leg.AddEntry(hVarpi,"#pi","l")
                    leg.AddEntry(hVarpi_clones,"#pi - clones","l")
                    leg.AddEntry(hVarp,"p","l")
                    leg.AddEntry(hVarp_clones,"p - clones","l")
                    c1.cd(12)
                    leg.Draw()
                c1.Update()
                hVarpi.Write()
                hVarpi_clones.Write()
                hVarp.Write()
                hVarp_clones.Write()
            c1.Write()
            outFile.Close()
            c1.SaveAs(f'RegVarsClones{outlabel}.png')
            print('Training variables plot completed.\n')

            dfPi = TrainSet.query('label == 0 and isclone == 0 and 0.5 < beta < 0.85', inplace=False)
            dfPiClone = TrainSet.query('label == 0 and isclone == 1 and 0.5 < beta < 0.85', inplace=False)
            dfP = TrainSet.query('label == 2 and isclone == 0 and 0.5 < beta < 0.85', inplace=False)
            dfPClone = TrainSet.query('label == 2 and isclone == 1 and 0.5 < beta < 0.85', inplace=False)
            dfK = TrainSet.query('label == 1 and isclone == 0 and 0.5 < beta < 0.85', inplace=False)
            dfKClone = TrainSet.query('label == 1 and isclone == 1 and 0.5 < beta < 0.85', inplace=False)
            plt.scatter(dfPi['p'], dfPi['beta'], facecolors='none', edgecolors='b', label='pi')
            plt.plot(dfPiClone['p'], dfPiClone['beta'], 'x', alpha=0.9, color='b', label='pi clone')
            plt.ylabel('b')
            plt.xlabel('p')
            plt.legend(frameon=False, fontsize=12, loc='best')
            plt.savefig(f'{OutPutDir}/betaVspVSClones_Pi{outlabel}.png')
            plt.close('all')
    
            plt.scatter(dfK['p'], dfK['beta'], facecolors='none', edgecolors='orange', label='k')
            plt.plot(dfKClone['p'], dfKClone['beta'], 'x', alpha=0.9, color='orange', label='k clone')
            plt.ylabel('beta')
            plt.xlabel('p')
            plt.legend(frameon=False, fontsize=12, loc='best')
            plt.savefig(f'{OutPutDir}/betaVspVSClones_K{outlabel}.png')
            plt.close('all')

            plt.scatter(TrainSet['p'], TrainSet['beta'], facecolors='none', edgecolors='g', label='all')
            plt.ylabel('b')
            plt.xlabel('p')
            plt.legend(frameon=False, fontsize=12, loc='best')
            plt.savefig(f'{OutPutDir}/BVspAll{outlabel}.png')
            plt.close('all')

    if 'betaflat' in training_conf:
        # candidates species weighted for beta
        #_____________________________________________
        print('BETA FLAT CONFIGURATION SELECTED')
        yTrainWeights = np.array(yTrain)
        print(f'\033[93mSame number of candidates over beta interval\033[0m')
        betamins = [0.2, 0.4, 0.6, 0.8]
        betamaxs = [0.4, 0.6, 0.8, 1.0]
        weigths = []
        for ibeta, (betamin, betamax) in enumerate(zip(betamins, betamaxs)):
            weigths.append(len(TrainSet.query(f' {betamin} <= beta < {betamax}')) / len(TrainSet))
        candw = np.zeros(len(TrainSet))
        with alive_bar(len(yTrain), title="Weight calculation") as bar:
            for icand in range(len(yTrain)):
                for iweights, weight in enumerate(weigths):
                    if (betamins[iweights] <= yTrainWeights[icand] < betamaxs[iweights]):
                        candw[icand] = 1./weight
                bar()
        data_conf += '_betaflat'

        if doPlots:
            plt.hist(yTrainWeights, weights=candw, color='b', alpha=0.5, range=[0, 1], bins=5,
             label=f'weigths')
            plt.legend(frameon=False, fontsize=12, loc='best')
            plt.xlabel('w')
            plt.savefig(f'{OutPutDir}/Weights{outlabel}.png')
            plt.close('all')


    TrainTestData = [TrainSet, yTrain, TestSet, yTest]
    if 'betaflat' in training_conf:
        TrainTestData.append(candw)
    
    print('Candidates in training:')
    print(f'N pi:\t{len(TrainSet.query("label == 0"))}\nN K:\t{len(TrainSet.query("label == 1"))}\nN p:\t{len(TrainSet.query("label == 2 "))}\n')
    print('Candidates in test:')
    print(f'N pi:\t{len(TestSet.query("label == 0"))}\nN K:\t{len(TestSet.query("label == 1"))}\nN p:\t{len(TestSet.query("label == 2"))}\n')
    print('Candidates in application:')
    print(f'N pi:\t{len(ApplDf.query(f"{piTag} or (nSigmaPi < 1 and p > 0.7)"))}\nN K:\t{len(ApplDf.query(f"{kTag} or (nSigmaK < 1 and p > 0.7)"))}\nN p:\t{len(ApplDf.query(f"{pTag} or (nSigmaP < 1 and p > 0.7)"))}\n')

    # preparation plot
    #_____________________________________________
    if doPlots:
        LegLabels = [inputCfg['output']['leg_labels']['pi'],
                     inputCfg['output']['leg_labels']['Kaon'],
                     inputCfg['output']['leg_labels']['Proton']]
        VarsToDraw = inputCfg['plots']['plotting_columns']
        OutputLabels = inputCfg['output']['out_labels']

        list_df = [df_pi, df_K, df_P]
        plot_utils.plot_distr(list_df, VarsToDraw, 100, LegLabels, figsize=(24, 14),
                              alpha=0.3, log=True, grid=False, density=True)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
        plt.savefig(f'{OutPutDir}/DistributionsAllTrainTest{outlabel}.png')
        plt.close('all')

        plot_utils.plot_distr([TestSet, ApplDf], VarsToDraw, 100, ['test', 'appl'], figsize=(24, 14),
                              alpha=0.3, log=True, grid=False, density=True)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
        plt.savefig(f'{OutPutDir}/DistributionsAllAppl{outlabel}.png')
        plt.close('all')

        CorrMatrixFig = plot_utils.plot_corr(list_df, VarsToDraw, LegLabels)
        for Fig, Lab in zip(CorrMatrixFig, OutputLabels):
            plt.figure(Fig.number)
            plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
            Fig.savefig(f'{OutPutDir}/CorrMatrix{Lab}{outlabel}.png')

    if inputCfg['output']['save_prepared_data']:
        dfTrainSet = pd.DataFrame(TrainSet)
        dfyTrain = pd.DataFrame(yTrain)
        dfTestSet = pd.DataFrame(TestSet)
        dfyTest = pd.DataFrame(yTest)

        print(f'Training, Test and Appl sample saved in {OutPutDir}')
        dfTestSet.to_parquet(f'{OutPutDir}/TestSet{data_conf}{outlabel}.parquet.gzip')
        dfTrainSet.to_parquet(f'{OutPutDir}/TrainSet{data_conf}{outlabel}.parquet.gzip')
        dfyTrain.to_parquet(f'{OutPutDir}/yTrain{data_conf}{outlabel}.parquet.gzip')
        dfyTest.to_parquet(f'{OutPutDir}/yTest{data_conf}{outlabel}.parquet.gzip')
        if 'betaflat' in training_conf:
            dfcandw = pd.DataFrame()
            dfcandw['candw'] = candw
            dfcandw.to_parquet(f'{OutPutDir}/candw{data_conf}{outlabel}.parquet.gzip')
        ApplDf.to_parquet(f'{OutPutDir}/ApplDf{data_conf}{outlabel}.parquet.gzip')

    return TrainTestData, ApplDf


def regression(inputCfg, OutPutDir, TrainSet, yTrain, TestSet, yTest, weigth=None): #pylint: disable=too-many-statements, too-many-branches
    '''
    function to perform regression
    '''
    HyperPars = inputCfg['ml']['hyper_par']
    TrainCols = inputCfg['ml']['training_columns']
    isXGB = inputCfg['ml']['isXGB']
    doPlots = inputCfg['plots']['doPlots']
    outlabel = inputCfg['output']['model_outlabel']

    # Hyper-pars opt
    #_____________________________________________
    if inputCfg['ml']['do_hyp_opt']:
        print('\033[93mPerforming Optuna hyper-parameters optimisation\033[0m')
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial : optimization(trial, inputCfg, TrainSet, yTrain), n_trials=10, timeout=36000, show_progress_bar=True)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(f'{OutPutDir}/OptHistoryOptuna{outlabel}.png')
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(f'{OutPutDir}/ParamImportanceOptuna{outlabel}.png')
    
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(f'{OutPutDir}/ParalleCoordinatesOptuna{outlabel}.png')
        fig = optuna.visualization.plot_contour(study)
        fig.write_image(f'{OutPutDir}/ContourPlotOptuna{outlabel}.png')

        HyperPars = study.best_trial.params
    print(f'Hyper-pars: {HyperPars}')

    if isXGB:
        print('XGB model selected')
        modelReg = xgb.XGBRegressor(**HyperPars)
    else:
        print('AutoML model selected')
        modelReg = AutoML()

    # Train model
    #_____________________________________________
    print('Model training...\r')
    if 'betaflat' in inputCfg['data_prep']['training_conf']:
        dotrain = True
        print('Weights applied in training')
        if isXGB:
            print('XGBoost model selected')
            dotrain = True
            with alive_bar(title="Training") as bar:
                modelReg.fit(TrainSet[TrainCols], yTrain, sample_weight=weigth)
        else:
            print('automl model selected')
            modelReg.fit(X_train=TrainSet[TrainCols], y_train=np.array(yTrain), **HyperPars)
    else:
        print('No weights applied in training')
        if isXGB:
            print('XGBoost model selected')
            dotrain = True
            with alive_bar(title="Training") as bar:
                modelReg.fit(TrainSet[TrainCols], yTrain)
        else:
            print('automl model selected')
            modelReg.fit(X_train=TrainSet[TrainCols], y_train=np.array(yTrain), **HyperPars)
    print('Training completed.')

    # Save model in pickle
    #_____________________________________________
    if inputCfg['output']['save_model'] and isXGB:
        print(f'Saving regressor model in {OutPutDir}...')
        with open(f'{OutPutDir}/RegModel{outlabel}.pickle', "wb") as output_file:
            pickle.dump(modelReg, output_file)
        print('Model saved.')

    # Plots
    #_____________________________________________
    if True:
        if isXGB:
            print('XGB feature importance plot')
            # xgb feature importance
            #_____________________________________________
            FeatureImportance = xgb.plot_importance(modelReg)
            plt.savefig(f'{OutPutDir}/RegFeatureImportance{outlabel}.png')
            plt.close('all')

            # SHAP feature importance
            #_____________________________________________
            explainer = shap.Explainer(modelReg) 
            TestSetShap = [] 
            thr = min(len(TestSet.query('label == 0')), len(TestSet.query('label == 1')), len(TestSet.query('label == 2')), 1000)
            for i, species in enumerate(['pi', 'k', 'p']):
                TestSetShap.append(TestSet.query(f'label == {i}'))
                data = TestSetShap[i].iloc[:thr]
                shap_val = explainer.shap_values(data[TrainCols])
                shap_obj = explainer(data[TrainCols])
                plt.figure(figsize=(28, 9))
                shap.plots.beeswarm(shap_obj, max_display=len(TrainCols))
                plt.savefig(f'{OutPutDir}/ShapImportance{species}{outlabel}.png', bbox_inches='tight')
                plt.close('all')

        print('Reg vs beta in Training and Test plot')
        yPredTrain = []
        yPredTest = []
        dfTest_sel = []
        for i in range(3):
            dfTest_sel.append(TrainSet.query(f'label == {i} and 0.1 < p < 0.7', inplace=False))
            yPredTest.append(modelReg.predict(dfTest_sel[i][TrainCols]))
        
        c1 = TCanvas("c1", "", 1200, 1200)
        outFile = TFile(f'TrainTestRegOut{outlabel}.root', 'recreate')
        leg = TLegend(0.4,0.4,0.8,0.8)
        leg.SetBorderSize(0)

        # Reg vs beta
        hBetapi = TH1F('hBeta_pi', ';#beta; counts', 100, 0, 1)
        hBetak = TH1F('hBeta_k', ';#beta; counts', 100, 0, 1)
        hBetap = TH1F('hBeta_p', ';#beta; counts', 100, 0, 1)
        hRegOutpi = TH1F('hRegOut_pi', ';#beta; counts', 100, 0, 1)
        hRegOutk = TH1F('hRegOut_k', ';#beta; counts', 100, 0, 1)
        hRegOutp = TH1F('hRegOut_p', ';#beta; counts', 100, 0, 1)

        SetHistStyle(hBetapi, colors[0])
        SetHistStyle(hBetak, colors[1])
        SetHistStyle(hBetap, colors[2])
        SetHistStyle(hRegOutpi, colors[0], style=2)
        SetHistStyle(hRegOutk, colors[1], style=2)
        SetHistStyle(hRegOutp, colors[2], style=2)

        # Reg performance evaluation
        hRegPerformance = TH2F('hRegPerformance', ';#beta;  #beta - Reg out', 1000, 0., 1, 10000, -5., 5.)
        hRegPerformancePi = TH2F('hRegPerformancePi', ';#beta;  #beta - Reg out (#pi)', 1000, 0., 1, 10000, -5., 5.)
        hRegPerformanceK = TH2F('hRegPerformanceK', ';#beta;  #beta - Reg out (K)', 1000, 0., 1, 10000, -5., 5.)
        hRegPerformanceP = TH2F('hRegPerformanceP', ';#beta;  #beta - Reg out (p)', 1000, 0., 1, 10000, -5., 5.)
        gRegPerformance = TGraphErrors('gRegPerformance')
        gRegPerformancePi = TGraphErrors('gRegPerformancePi')
        gRegPerformanceK = TGraphErrors('gRegPerformanceK')
        gRegPerformanceP = TGraphErrors('gRegPerformanceP')

        SetHistStyle(gRegPerformancePi, colors[0], marker=markers[0], markerSize=2)
        SetHistStyle(gRegPerformanceK, colors[1], marker=markers[1], markerSize=2)
        SetHistStyle(gRegPerformanceP, colors[2], marker=markers[2], markerSize=2)

        for i, (beta, regout) in enumerate(zip(dfTest_sel[0]['beta'], yPredTest[0])):
            hBetapi.Fill(beta)
            hRegOutpi.Fill(regout)
            hRegPerformancePi.Fill(beta, beta - regout)
            hRegPerformance.Fill(beta, beta - regout)
        gRegPerformance.SetPoint(0, 0, hRegPerformancePi.GetMean(2))
        gRegPerformance.SetPointError(0, 1.e-10, hRegPerformancePi.GetMeanError(2))
        gRegPerformancePi.SetPoint(0, 0.3, hRegPerformancePi.GetMean(2))
        gRegPerformancePi.SetPointError(0, 1.e-10, hRegPerformancePi.GetMeanError(2))
        for i, (beta, regout) in enumerate(zip(dfTest_sel[1]['beta'], yPredTest[1])):
            hBetak.Fill(beta)
            hRegOutk.Fill(regout)
            hRegPerformanceK.Fill(beta, beta - regout)
            hRegPerformance.Fill(beta, beta - regout)
        gRegPerformance.SetPoint(1, 1, hRegPerformanceK.GetMean(2))
        gRegPerformance.SetPointError(1, 1.e-10, hRegPerformanceK.GetMeanError(2))
        gRegPerformanceK.SetPoint(0, 0.6, hRegPerformanceK.GetMean(2))
        gRegPerformanceK.SetPointError(0, 1.e-10, hRegPerformanceK.GetMeanError(2))
        for i, (beta, regout) in enumerate(zip(dfTest_sel[2]['beta'], yPredTest[2])):
            hBetap.Fill(beta)
            hRegOutp.Fill(regout)
            hRegPerformanceP.Fill(beta, beta - regout)
            hRegPerformance.Fill(beta, beta - regout)
        gRegPerformance.SetPoint(2, 2, hRegPerformanceP.GetMean(2))
        gRegPerformance.SetPointError(2, 1.e-10, hRegPerformanceP.GetMeanError(2))
        gRegPerformanceP.SetPoint(0, 0.9, hRegPerformanceP.GetMean(2))
        gRegPerformanceP.SetPointError(0, 1.e-10, hRegPerformanceP.GetMeanError(2))

        hBetapi.DrawNormalized('hist')
        hRegOutpi.DrawNormalized('hist')
        hBetak.DrawNormalized('hist')
        hRegOutk.DrawNormalized('hist')
        hBetap.DrawNormalized('hist')
        hRegOutp.DrawNormalized('hist')

        c1.cd()
        hFrame = c1.cd().DrawFrame(0, -0.4, 1, 0.4, ';#beta; #beta - Reg out')
        hFrame.GetXaxis().SetMoreLogLabels()
        hFrame.GetYaxis().SetDecimals()
        hRegPerformance.Draw('colz')
        gRegPerformance.SetMarkerSize(1.5)
        gRegPerformance.SetMarkerColor(kRed+1)
        gRegPerformancePi.Draw('p same')
        gRegPerformanceK.Draw('p same')
        gRegPerformanceP.Draw('p same')

        hBetapi.Write()
        hRegOutpi.Write()
        hBetak.Write()
        hRegOutk.Write()
        hBetap.Write()
        hRegOutp.Write()
        hRegPerformance.Write()
        hRegPerformancePi.Write()
        hRegPerformanceK.Write()
        hRegPerformanceP.Write()
        gRegPerformance.Write()
        c1.Write()
        outFile.Close()

    return modelReg

def appl(inputCfg, OutPutDir, Model, ApplDf):
    print('Applying ML model...', end='\r')

    TrainCols = inputCfg['ml']['training_columns']
    doPlots = inputCfg['plots']['doPlots']
    with alive_bar(title="Application") as bar:
        Pred = Model.predict(ApplDf[TrainCols])
        bar()
    ApplDf['Reg_output'] = Pred
    print('ML model application completed')

    outlabel = inputCfg['output']['model_outlabel']
    if inputCfg['ml']['isXGB']:
        outlabel += 'XGB'
    else:
        outlabel += 'AutoML'
    ApplDf.to_parquet(f'{OutPutDir}/RegApplied_{outlabel}.parquet.gzip')
    print(f'Final dataframe:\n{ApplDf}')


def optimization(trial: Trial, inputCfg, TrainData, yTrain):
    HyperPars = {#'n_jobs':40,
                 'base_score': 0.5,
                 #"metric": 'r2',
                 #"task": 'regression',
                 #"log_file_name": "bdt.log",
                 #"estimator_list" : ['xgboost'],
                 #'booster': 'gbtree',
                 #'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.1, 1.),
                 #'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.5, 0.7, 0.9, 1.0]),
                 #'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
                 #'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 1.),
                 #'n_estimators': trial.suggest_categorical('n_estimators', [200, 300, 400, 500, 700, 800, 1000, 1200]),
                 #'max_depth': trial.suggest_categorical('max_depth', [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 20]),
                 'min_child_weight': trial.suggest_int('min_child_weight', 0, 100),
                 "eval_metric": trial.suggest_categorical('eval_metric', ["logloss", "rmse", "rmsle"]),
                 "objective": "reg:squarederror",
                 "max_leaves": 118,
                 "importance_type": 'gain',
                 #"gpu_id": 1,
                 "grow_policy": 'lossguide',
                 "max_delta_step": 0,
                 #"reg_alpha": trial.suggest_uniform("reg_alpha", 0.0, 1.), 
                 #"reg_lambda": trial.suggest_uniform("reg_lambda", 0.0, 1.)
                }

    TrainCols = inputCfg['ml']['training_columns']
    modelReg = xgb.XGBRegressor(**HyperPars)

    # cross validation
    #_____________________________________________
    cv_score = cross_val_score(modelReg, TrainData[TrainCols], yTrain, cv=5, scoring='neg_mean_squared_error')
    print(f'\033[93mMean cross_val_score: {np.mean(cv_score):.6f}\033[0m')
    print('\033[93m==============================\033[0m')
    return abs(np.mean(cv_score))


def main(): #pylint: disable=too-many-statements
    # Read config file
    #_____________________________________________
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)

    print('Loading and preparing data files: ...', end='\r')
    Data = inputCfg['input']['data']
    OutPutDir = inputCfg['output']['dir']
    if os.path.isdir(OutPutDir):
        print((f'\033[93mWARNING: Output directory \'{OutPutDir}\' already exists,'
                ' overwrites possibly ongoing!\033[0m'))
    else:
        os.makedirs(OutPutDir)

    # Data preparation
    #_____________________________________________
    if not inputCfg['data_prep']['skip_data_prep']:
        print('\nDATA PREPARATION')
        TrainTestData, ApplDf = data_prep(inputCfg, OutPutDir, Data)
    else:
        print('\nSKIPPING DATA PREPARATION')
        if 'betaflat' in inputCfg['data_prep']['training_conf']:
            print('Be sure you the weights are present in the prapred data')
        preparedData = inputCfg['data_prep']['preparedData']
        TrainTestData = []
        for i, data in enumerate(preparedData):
            TrainTestData.append(pd.read_parquet(data))
        print(f'Application Df taken from {inputCfg["data_prep"]["preparedData"][-1]}')
        ApplDf = pd.read_parquet(f'{inputCfg["data_prep"]["preparedData"][-1]}')

    # Training
    #_____________________________________________
    if not inputCfg['appl']['stand_alone']:
        print('\nTRAINING')
        if 'betaflat' in inputCfg['data_prep']['training_conf']:
            ModelHandl = regression(inputCfg, OutPutDir, TrainTestData[0], TrainTestData[1], TrainTestData[2], TrainTestData[3], TrainTestData[4])
        else:
            ModelHandl = regression(inputCfg, OutPutDir, TrainTestData[0], TrainTestData[1], TrainTestData[2], TrainTestData[3])

    # Application
    #_____________________________________________
    if inputCfg['appl']['stand_alone']:
        if inputCfg['ml']['isXGB']:
            ModelHandl =  pickle.load(open(inputCfg['appl']['saved_model'], "rb"))
            ModelHandl.training_columns = inputCfg['ml']['training_columns']
            hypPar = ModelHandl.get_params()
            ModelHandl.set_params(**hypPar)
    print('\nAPPLICATION')
    appl(inputCfg, OutPutDir, ModelHandl, ApplDf)

    # Delete dataframes to release memory
    #_____________________________________________
    del TrainTestData, ApplDf

main()
