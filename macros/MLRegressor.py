'''
python scri to run basic training and application using the hipe4ml package
run: python MLClassification.py cfgFileNameML.yml [--train, --apply]
'''
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
#from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from hipe4ml import plot_utils
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from flaml import AutoML

from ROOT import TFile, TH2F, TH1F, TH1D, TCanvas, TLegend, TGraph, kRed, kBlue, kAzure, kSpring, kGreen, kOrange, kGray, kBlack, TGaxis,gPad, TLatex # pylint: disable=import-error,no-name-in-module

def data_prep(inputCfg, OutPutDir, Df): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for data preparation
    '''
    df = uproot.open("/data/fmazzasc/its_data/TreePID_505658.root")['ITStreeML'].arrays(library='pd')
    if 'TestApplSame' in inputCfg['data_prep']['training_conf']:
        additionalDf = uproot.open("/data/fmazzasc/its_data/pid505673.root")['ITStreeML'].arrays(library='pd')
        df = pd.concat([df, additionalDf], sort=True)
    for i in range(7):
        df[f'ClSizeL{i}'] = np.where(df[f'ClSizeL{i}'] < 0, df['meanClsize'], df[f'ClSizeL{i}'])
    df['L'] = np.arctan(df['tgL'])
    #df['nSigmaE'] = abs(df['nSigmaE'])
    df['nSigmaP'] = abs(df['nSigmaP'])
    df['nSigmaPi'] = abs(df['nSigmaPi'])
    df['nSigmaK'] = abs(df['nSigmaK'])
    df.eval('mean_patt_ID = (ClPattIDL0 + ClPattIDL1 + ClPattIDL2 + ClPattIDL3 + ClPattIDL4 + ClPattIDL5 + ClPattIDL6)/7', inplace=True)
    presel = inputCfg['data_prep']['presel']
    df_sel = df.query(presel)
    if inputCfg['input']['separateAppl']:
        print('\033[93mDifferent .root file for application and training\033[0m')
        ApplDf = uproot.open("/data/fmazzasc/its_data/TreePID_505673.root")['ITStreeML'].arrays(library='pd')
        for i in range(7):
            ApplDf[f'ClSizeL{i}'] = np.where(ApplDf[f'ClSizeL{i}'] < 0, ApplDf['meanClsize'], ApplDf[f'ClSizeL{i}'])
        ApplDf = ApplDf.query(presel)
    else:
        ApplDf = df_sel.copy()

    ApplDf.query('0 < p < 50', inplace=True)
    ApplDf['nSigmaE'] = abs(ApplDf['nSigmaE'])
    ApplDf['nSigmaP'] = abs(ApplDf['nSigmaP'])
    ApplDf['nSigmaPi'] = abs(ApplDf['nSigmaPi'])
    ApplDf['nSigmaK'] = abs(ApplDf['nSigmaK'])

    df_P = df_sel.query('nSigmaP < 1 and p < 1.', inplace=False)
    df_P.eval('bg = p/sqrt(0.93827200**2 + p**2)',  inplace=True)
    df_pi = df_sel.query('nSigmaPi < 1 and nSigmaK > 2 and p < 1.', inplace=False)
    df_pi.eval('bg = p/sqrt(0.13957000**2 + p**2)',  inplace=True)
    df_K = df_sel.query('nSigmaK < 1 and nSigmaPi > 1 and p < 1.', inplace=False)
    df_K.eval('bg = p/sqrt(0.4937**2 + p**2)',  inplace=True)
    df_e = df_sel.query('-2 < nSigmaE < 1 and nSigmaK > 4 and nSigmaPi > 2 and p < 0.2', inplace=False)
    df_e.eval('bg = p/sqrt(0.000510998**2 + p**2)',  inplace=True)
    df_d = df_sel.query('nSigmaDeu < 1.', inplace=False)
    df_d.eval('bg = p/sqrt(2.2**2 + p**2)',  inplace=True)

    seed_split = inputCfg['data_prep']['seed_split']
    test_frac = inputCfg['data_prep']['test_fraction']
    TotDf = pd.concat([df_pi, df_K, df_P], sort=True)

    if 'fulltraining' in inputCfg['data_prep']['training_conf']:
        print('\033[93mFull sample adopted for training\033[0m')
        TrainSet = TotDf.copy()
        yTrain = TrainSet['bg']
        TestSet = ApplDf.copy()
        yTest = TestSet['bg']
    else:
        TrainSet, TestSet, yTrain, yTest = train_test_split(TotDf, TotDf['bg'], test_size=test_frac, random_state=seed_split)

    if 'equal' in inputCfg['data_prep']['training_conf']:
        nCandToKeep = min(len(df_pi), len(df_P), len(df_K)) if not 'proton' in inputCfg['data_prep']['training_conf'] else len(df_P)
        print(f'\033[93mSame number of candidates ({nCandToKeep}) for each class\033[0m')
        TrainSet = pd.concat([TrainSet.query('nSigmaPi < 1').iloc[:50000], TrainSet.query('nSigmaK < 1 and nSigmaPi > 1').iloc[:nCandToKeep], TrainSet.query('nSigmaP < 1').iloc[:nCandToKeep]], sort=False)
        yTrain = TrainSet['bg']
    if 'bgflat' in inputCfg['data_prep']['training_conf']:
        yTrainWeights = np.array(yTrain)
        print(f'\033[93mSame number of candidates over bg interval\033[0m')
        bgmins = [0.1]
        bgmaxs = [0.7]
        weigths = []
        for ibg, (bgmin, bgmax) in enumerate(zip(bgmins, bgmaxs)):
            weigths.append(len(TrainSet.query(f' {bgmin} <= p < {bgmax}')) / len(TrainSet))
        candw = np.zeros(len(TrainSet))
        for icand in range(len(yTrain)):
            for iweights, weight in enumerate(weigths):
                if (bgmins[iweights] <= yTrainWeights[icand] < bgmaxs[iweights]):
                    candw[icand] = 1./weight
        #nCandToKeep = []
        #TrainSets = []
        #for ibg, (bgmin, bgmax) in enumerate(zip(bgmins, bgmaxs)):
        #    nCandToKeep.append(len(TrainSet.query(f'{bgmin} < bg < {bgmax}')))
        #    print(f'\033[93mNumber of candidates in {bgmin} < bg < {bgmax}: {nCandToKeep[ibg]}\033[0m')
        #minCandToKeep = min(i for i in nCandToKeep if i > 0)
        #print(minCandToKeep)
        #for ibg, (bgmin, bgmax) in enumerate(zip(bgmins, bgmaxs)):
        #    TrainSets.append(TrainSet.query(f'{bgmin} < bg < {bgmax}').iloc[:minCandToKeep])
        #TrainSet = pd.concat(TrainSets, sort=True)
        #yTrain = TrainSet['bg']
        #for ibg, (bgmin, bgmax) in enumerate(zip(bgmins, bgmaxs)):
        #    print(f'\033[93mNumber of candidates in {bgmin} < bg < {bgmax}: {len(TrainSet.query(f"{bgmin} < bg < {bgmax}"))}\033[0m')
    if 'augmentation' in inputCfg['data_prep']['training_conf']:
        TrainSet['isclone'] = 0

        #print('Augemented contribution of K from p')
        #df_k_aug_from_p = TrainSet.query('nSigmaP < 1 and 0.3 < bg < 0.5', inplace=False)
        #df_k_aug_from_p['p'] = df_k_aug_from_p['p']*(0.4937/0.93827200)
        #df_k_aug_from_p['nSigmaK'] = 0
        #df_k_aug_from_p['nSigmaP'] = 1000
        #df_k_aug_from_p['isclone'] = 1

        print('Augemented contribution of K from pi')
        df_k_aug_from_pi = TrainSet.query('nSigmaPi < 1 and 0.5 < bg < 0.85', inplace=False)
        print(df_k_aug_from_pi['p'])
        df_k_aug_from_pi['p'] = df_k_aug_from_pi['p']*(0.4937/0.13957000)
        print(df_k_aug_from_pi['p'])
        df_k_aug_from_pi['nSigmaK'] = 0
        df_k_aug_from_pi['nSigmaPi'] = 1000
        df_k_aug_from_pi['isclone'] = 1

        TrainSet = pd.concat([TrainSet, df_k_aug_from_pi], sort=True)
        yTrain = TrainSet['bg']
    
        dfPi = TrainSet.query('nSigmaPi < 1 and isclone == 0 and 0.5 < bg < 0.85', inplace=False)
        dfPiClone = TrainSet.query('nSigmaPi < 1 and isclone == 1 and 0.5 < bg < 0.85', inplace=False)
        dfP = TrainSet.query('nSigmaP < 1 and isclone == 0 and 0.5 < bg < 0.85', inplace=False)
        dfPClone = TrainSet.query('nSigmaP < 1 and isclone == 1 and 0.5 < bg < 0.85', inplace=False)
        dfK = TrainSet.query('nSigmaK < 1 and isclone == 0 and 0.5 < bg < 0.85', inplace=False)
        dfKClone = TrainSet.query('nSigmaK < 1 and isclone == 1 and 0.5 < bg < 0.85', inplace=False)
        plt.scatter(dfPi['p'], dfPi['bg'], facecolors='none', edgecolors='b', label='pi')
        plt.plot(dfPiClone['p'], dfPiClone['bg'], 'x', alpha=0.9, color='b', label='pi clone')
        plt.ylabel('b')
        plt.xlabel('p')
        plt.legend(frameon=False, fontsize=12, loc='best')
        plt.savefig(f'{OutPutDir}/BgVspVSClones_Pi.png')
        plt.close('all')
    
        plt.scatter(dfK['p'], dfK['bg'], facecolors='none', edgecolors='orange', label='k')
        plt.plot(dfKClone['p'], dfKClone['bg'], 'x', alpha=0.9, color='orange', label='k clone')
        plt.ylabel('bg')
        plt.xlabel('p')
        plt.legend(frameon=False, fontsize=12, loc='best')
        plt.savefig(f'{OutPutDir}/BgVspVSClones_K.png')
        plt.close('all')
    
        plt.scatter(dfP['p'], dfP['bg'], facecolors='none', edgecolors='g', label='p')
        plt.plot(dfPClone['p'], dfPClone['bg'], 'x', alpha=0.9, color='g', label='p clone')
        plt.ylabel('bg')
        plt.xlabel('p')
        plt.legend(frameon=False, fontsize=12, loc='best')
        plt.savefig(f'{OutPutDir}/BgVspVSClones_P.png')
        plt.close('all')
        input()

        print(f'\033[93mSame number of candidates over bg interval\033[0m')
    if 'TestApplSame' in inputCfg['data_prep']['training_conf']:
        ApplDf = TestSet

    #plt.hist(yTrainWeights, weights=candw, color='b', alpha=0.5, range=[0, 1], bins=5,
    #         label=f'weigths')
    #plt.legend(frameon=False, fontsize=12, loc='best')
    #plt.xlabel('w')
    #plt.savefig(f'{OutPutDir}/Weights.png')
    #plt.close('all')

    TrainTestData = [TrainSet, yTrain, TestSet, yTest]
    ApplDf.eval('mean_patt_ID = (ClPattIDL0 + ClPattIDL1 + ClPattIDL2 + ClPattIDL3 + ClPattIDL4 + ClPattIDL5 + ClPattIDL6)/7', inplace=True)
    print('Candidates in training:')
    print(f'N pi:\t{len(TrainSet.query("nSigmaPi < 1"))}\nN K:\t{len(TrainSet.query("nSigmaK < 1 and nSigmaPi > 1"))}\nN p:\t{len(TrainSet.query("nSigmaP < 1"))}\nN e:\t{len(TrainSet.query("nSigmaE < 1 and nSigmaK > 1 and nSigmaPi > 1"))}\n')
    print('Candidates in test:')
    print(f'N pi:\t{len(TestSet.query("nSigmaPi < 1"))}\nN K:\t{len(TestSet.query("nSigmaK < 1 and nSigmaPi > 1"))}\nN p:\t{len(TestSet.query("nSigmaP < 1"))}\nN d:\t{len(TestSet.query("label == 3"))}\nN e:\t{len(TestSet.query("nSigmaE < 1 and nSigmaK > 1 and nSigmaPi > 1"))}\n')
    print('Candidates in application:')
    print(f'N pi:\t{len(ApplDf.query("nSigmaPi < 1"))}\nN K:\t{len(ApplDf.query("nSigmaK < 1 and nSigmaPi > 1"))}\nN p:\t{len(ApplDf.query("nSigmaP < 1"))}\nN d:\t{len(ApplDf.query("label == 3"))}\n')

    #_____________________________________________
    LegLabels = [inputCfg['output']['leg_labels']['pi'],
                 inputCfg['output']['leg_labels']['Kaon'],
                 inputCfg['output']['leg_labels']['Proton']]
    VarsToDraw = inputCfg['plots']['plotting_columns']
    OutputLabels = inputCfg['output']['out_labels']

    list_df = [df_pi, df_K, df_P]
    plot_utils.plot_distr(list_df, VarsToDraw, 100, LegLabels, figsize=(24, 14),
                          alpha=0.3, log=True, grid=False, density=True)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(f'{OutPutDir}/DistributionsAllTrainTest.png')
    plt.close('all')

    plot_utils.plot_distr([TestSet, ApplDf], VarsToDraw, 100, ['test', 'appl'], figsize=(24, 14),
                          alpha=0.3, log=True, grid=False, density=True)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(f'{OutPutDir}/DistributionsAllAppl.png')
    plt.close('all')

    CorrMatrixFig = plot_utils.plot_corr(list_df, VarsToDraw, LegLabels)
    for Fig, Lab in zip(CorrMatrixFig, OutputLabels):
        plt.figure(Fig.number)
        plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
        Fig.savefig(f'{OutPutDir}/CorrMatrix{Lab}.png')

    return TrainTestData, ApplDf


def regression(inputCfg, OutPutDir, TrainTestData): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for regressor training and testing
    '''
    HyperPars = inputCfg['ml']['hyper_par']
    TrainCols = inputCfg['ml']['training_columns']
    modelReg = AutoML() #xgb.XGBRegressor(**HyperPars)
    if not isinstance(TrainCols, list):
        print('\033[91mERROR: training columns must be defined!\033[0m')
        sys.exit()
    if not isinstance(HyperPars, dict):
        print('\033[91mERROR: hyper-parameters must be defined or be an emy dict!\033[0m')
        sys.exit()

    #TrainCols = ["ClSizeL0", "ClSizeL1", "ClSizeL2", "ClSizeL3", "ClSizeL4", "ClSizeL5", "ClSizeL6", "tgL", "meanClsize", 'mean_patt_ID', 'pt']

    # model definition
    #ModelHandl = ModelHandler(modelReg, TrainCols)
    automl_settings = {
        "metric": 'rmse',
        'n_jobs':40, 
        #'max_depth':5, 
        #'learning_rate':0.06, 
        #'n_estimators':1200, 
        #'min_child_weight':243, 
        #'subsample':0.6, 
        #'colsample_bytree':0.7, 
    }
    TrainSet = TrainTestData[0]
    # train and test the model
    modelReg.fit(TrainSet[TrainCols], TrainTestData[1], dataframe=TrainCols, task = 'regression', time_budget=10, **automl_settings)
    yPredTest = modelReg.predict(TrainTestData[2])
    #yPredTest = ModelHandl.train_test_regrressor(TrainTestData, return_prediction=True, nfold=5, njobs=3)
    yPredTrain = modelReg.predict(TrainTestData[0])

    # save model handler in pickle
    #ModelHandl.dump_model_handler(f'{OutPutDir}/RegHandler.pickle')
    #ModelHandl.dump_original_model(f'{OutPutDir}/XGBoostRegressor.model', True)

    #with open(f"{OutPutDir}/automl.pkl", "wb") as f:
    #    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    # plots
    #_____________________________________________
    plt.hist(yPredTrain, color='b', alpha=0.5, range=[0, 10], bins=100,
             histtype='stepfilled', label=f'Reg. out training')
    plt.hist(TrainTestData[1], color='g', alpha=0.5, range=[0, 10], bins=100,
             histtype='stepfilled', label=f'True training')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/RegBgOutTrainingAll.png')
    plt.close('all')

    plt.hist(yPredTrain, color='b', alpha=0.5, range=[0, 5], bins=500,
             histtype='stepfilled', label=f'bg training')
    plt.hist(yPredTest, color='g', alpha=0.5, range=[0, 5], bins=500,
             histtype='stepfilled', label=f'bg test')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.xlabel('Regressor output')
    plt.ylabel('counts')
    plt.savefig(f'{OutPutDir}/RegBgOutTrainingTest.png')
    plt.close('all')

    plt.hist(yPredTest, color='b', alpha=0.5, range=[0, 8], bins=400,
             histtype='stepfilled', label=f'Reg. out test')
    plt.hist(TrainTestData[3], color='g', alpha=0.5, range=[0, 8], bins=400,
             histtype='stepfilled', label=f'True test')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/RegBgOutTestAll.png')
    plt.close('all')

    RegPrecision = (yPredTest - TrainTestData[3])/TrainTestData[3]
    rms = np.sqrt(sum(RegPrecision**2)/len(RegPrecision))
    mean = round(np.mean(RegPrecision), 2)


    plt.hist(RegPrecision, color='r', alpha=0.8, range=[-5, 5], bins=1000,
             histtype='stepfilled', label=f'Mean = {mean}\nRMS = {round(rms, 3)}')
    plt.yscale('log')
    plt.xlabel('(RegOut - bg) / bg')
    plt.legend(frameon=False, fontsize=8, loc='best')
    plt.savefig(f'{OutPutDir}/RegPrecision.png')
    plt.close('all')

    #FeatureImportance = xgb.plot_importance(ModelHandler.get_original_model(ModelHandl))
    #plt.savefig(f'{OutPutDir}/RegFeatureImportance.png')
    #plt.close('all')

    #shap_val = shap.TreeExplainer(modelReg.shap_values(TrainTestData[1]))
    #shapFig = plt.figure(figsize=(18, 9))
    #shap.summary_plot(shap_val[1], TrainTestData[1], plot_size=(18, 9), class_names=labels, show=False)
    #plt.savefig(f'{OutPutDir}/ShapImportance.png')
    #plt.close('all')

    return modelReg, yPredTest


def appl(inputCfg, OutPutDir, ModelHandl, ApplDf, TetsDf, yPredTest):
    print('Applying ML model: ...', end='\r')
    #with open(f"{OutPutDir}/automl.pkl", "rb") as f:
    #    ModelHandl = pickle.load(f)
    Pred = ModelHandl.predict(ApplDf)
    ApplDf['Reg_output'] = Pred
    ApplDf.to_parquet(f'{OutPutDir}/RegApplied_wE.parquet.gzip')
    print(f'Final dataframe:\n{ApplDf}')
    print('ML model application: Done!')

    TetsDf['Reg_output'] = yPredTest

    print('Final plot preparation')

    dfPi = TetsDf.query('nSigmaPi < 1', inplace=False)
    dfE = TetsDf.query('-2 < nSigmaE < 1 and nSigmaK > 4 and nSigmaPi > 2 and p < 0.2', inplace=False)
    plt.hist(dfPi['Reg_output'], color='b', alpha=0.5, range=[0, 2], bins=5000,
             histtype='stepfilled', label=f'Reg. out test - pi')
    plt.hist(dfPi['bg'], color='g', alpha=0.5, range=[0, 2], bins=5000,
             histtype='stepfilled', label=f'True test - pi')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/RegBgOutTestPi.png')
    plt.close('all')

    dfP = TetsDf.query('nSigmaP < 1', inplace=False)
    plt.hist(dfP['Reg_output'], color='b', alpha=0.5, range=[0, 1], bins=1000,
             histtype='stepfilled', label=f'Reg. out test - p')
    plt.hist(dfP['bg'], color='g', alpha=0.5, range=[0, 1], bins=1000,
             histtype='stepfilled', label=f'True test - p')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/RegBgOutTestP.png')
    plt.close('all')

    dfK = TetsDf.query('nSigmaK < 1 and nSigmaPi > 1', inplace=False)
    plt.hist(dfK['Reg_output'], color='b', alpha=0.5, range=[0, 1], bins=1000,
             histtype='stepfilled', label=f'Reg. out test - k')
    plt.hist(dfK['bg'], color='g', alpha=0.5, range=[0, 1], bins=1000,
             histtype='stepfilled', label=f'True test - k')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/RegBgOutTestK.png')
    plt.close('all')

    diffRegOutBg = (dfP['Reg_output'] - dfP['bg'])/dfP['bg']
    plt.plot(dfP['bg'], diffRegOutBg, 'or')
    plt.savefig(f'{OutPutDir}/ScatterPlot.png')
    plt.close('all')

    plt.plot(dfPi['Reg_output'], dfPi['meanClsize'],  '.', alpha=0.4, color='b', label='pi')
    plt.plot(dfK['Reg_output'], dfK['meanClsize'],  '.', alpha=0.4, color='orange', label='k')
    plt.plot(dfP['Reg_output'], dfP['meanClsize'],  '.', alpha=0.4, color='g', label='p')
    plt.ylabel('<CL size>')
    plt.xlabel('Reg. output')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/RegScoreVsMeanClSizePi_K_P.png')
    plt.close('all')

    dfDeu = TetsDf.query('nSigmaDeu < 0.8', inplace=False)
    plt.plot(dfPi['bg'], dfPi['meanClsize'], '.', alpha=0.9, color='b', label='pi')
    plt.plot(dfK['bg'], dfK['meanClsize'], '.', alpha=0.9, color='orange', label='k')
    plt.plot(dfP['bg'], dfP['meanClsize'], '.', alpha=0.9, color='g', label='p')
    plt.plot(dfE['bg'], dfE['meanClsize'], '.', alpha=0.9, color='r', label='e')
    plt.ylabel('<CL size>')
    plt.xlabel('bg')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/BgVsMeanClSizePi_K_P_e.png')
    plt.close('all')

    plt.plot(dfPi['p'], dfPi['meanClsize'], '.', alpha=0.9, color='b', label='pi')
    plt.plot(dfK['p'], dfK['meanClsize'], '.', alpha=0.9, color='orange', label='k')
    plt.plot(dfP['p'], dfP['meanClsize'], '.', alpha=0.9, color='g', label='p')
    plt.plot(dfE['p'], dfE['meanClsize'], '.', alpha=0.9, color='r', label='e')
    plt.ylabel('<CL size>')
    plt.xlabel('p')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/MeanClSIzeVsMomentum_Pi_K_P_e.png')
    plt.close('all')


    plt.plot(dfPi['Reg_output'], dfPi['dedx'], '.', alpha=0.9, color='b', label='pi')
    plt.plot(dfK['Reg_output'], dfK['dedx'], '.', alpha=0.9, color='orange', label='k')
    plt.plot(dfP['Reg_output'], dfP['dedx'], '.', alpha=0.9, color='g', label='p')
    plt.ylabel('dE/dx')
    plt.xlabel('Reg. output')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/dEdxVsMeanClSizePi_K_P.png')
    plt.close('all')

    plt.plot(dfPi['p'], dfPi['bg'], '.', alpha=0.9, color='b', label='pi')
    plt.plot(dfK['p'], dfK['bg'], '.', alpha=0.9, color='orange', label='k')
    plt.plot(dfP['p'], dfP['bg'], '.', alpha=0.9, color='g', label='p')
    plt.plot(dfE['p'], dfE['bg'], '.', alpha=0.9, color='r', label='e')
    plt.ylabel('bg')
    plt.xlabel('p')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/BgVsp_Pi_K_P.png')
    plt.close('all')

    plt.plot(dfPi['Reg_output'], dfPi['bg'], '.', alpha=0.3, color='b', label='pi')
    plt.plot(dfK['Reg_output'], dfK['bg'], '.', alpha=0.3, color='orange', label='k')
    plt.plot(dfP['Reg_output'], dfP['bg'], '.', alpha=0.3, color='g', label='p')
    plt.ylabel('bg')
    plt.xlabel('Reg. output')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/RegScoreVsbgPi_K_P.png')
    plt.close('all')

    plt.plot(dfPi['p'], dfPi['dedx'], '.', alpha=0.9, color='b', label='pi')
    plt.plot(dfK['p'], dfK['dedx'], '.', alpha=0.9, color='orange', label='k')
    plt.plot(dfP['p'], dfP['dedx'], '.', alpha=0.9, color='g', label='p')
    plt.ylabel('dE/dx')
    plt.xlabel('p')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/dEdxVspPi_P.png')
    plt.close('all')

    plt.hist(dfPi['Reg_output'], alpha=0.5, range=[0, 1.5], bins=150, color='b', histtype='stepfilled', label=f'pi')
    plt.hist(dfK['Reg_output'], alpha=0.5, range=[0, 1.5], bins=150, color='orange', histtype='stepfilled', label=f'k')
    plt.hist(dfP['Reg_output'], alpha=0.5, range=[0, 1.5], bins=150, color='g', histtype='stepfilled', label=f'p')
    plt.ylabel('counts')
    plt.xlabel('m')
    plt.yscale('log')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/massPi_K_P_Test.png')
    plt.close('all')

    RegPrecisionP = (dfP['Reg_output'] - dfP['bg'])/dfP['bg']
    RegPrecisionK = (dfK['Reg_output'] - dfK['bg'])/dfK['bg']
    RegPrecisionPi = (dfPi['Reg_output'] - dfPi['bg'])/dfPi['bg']
    RegPrecisionE = (dfE['Reg_output'] - dfE['bg'])/dfE['bg']
    plt.hist(RegPrecisionPi, alpha=0.2, range=[-1.5, 1.5], bins=300, color='b', histtype='stepfilled', label=f'pi - {round(np.mean(RegPrecisionPi), 2)}')
    plt.hist(RegPrecisionK, alpha=0.2, range=[-.15, 1.5], bins=300, color='orange', histtype='stepfilled', label=f'k - {round(np.mean(RegPrecisionK), 2)}')
    plt.hist(RegPrecisionP, alpha=0.2, range=[-.15, 1.5], bins=300, color='g', histtype='stepfilled', label=f'p - {round(np.mean(RegPrecisionP), 2)}')
    plt.hist(RegPrecisionE, alpha=0.2, range=[-.15, 1.5], bins=300, color='yellow', histtype='stepfilled', label=f'e - {round(np.mean(RegPrecisionE), 2)}')
    plt.yscale('log')
    plt.xlabel('(RegOut - bg) / bg')
    plt.legend(frameon=False, fontsize=8, loc='best')
    plt.savefig(f'{OutPutDir}/RegPrecision_pi_k_p_e.png')
    plt.close('all')

    plt.hist(ApplDf['p']/ApplDf['Reg_output'], alpha=0.5, range=[0, 1.5], bins=150, color='r', histtype='stepfilled', label=f'Appl')
    plt.ylabel('counts')
    plt.xlabel('m')
    plt.yscale('log')
    plt.legend(frameon=False, fontsize=12, loc='best')
    plt.savefig(f'{OutPutDir}/massDistrAppl.png')
    plt.close('all')



    #for iL in range(7):
    #    for ibg in range(0, 20):
    #        ibg *= 0.2
    #        ibg += 0.2
    #        dfPi_sel = dfPi.query(f'{ibg} < bg < {ibg+0.2}')
    #        plt.hist(dfPi_sel[f'ClSizeL{iL}'], alpha=0.5, range=[-0.5, 12.5], bins=12, color='b', histtype='stepfilled', label=f'pi - {round(ibg, 2)} < bg < {round(ibg+0.2, 2)}')
    #        dfK_sel = dfK.query(f'{ibg} < bg < {ibg+0.2}')
    #        plt.hist(dfK_sel[f'ClSizeL{iL}'], alpha=0.5, range=[-0.5, 12.5], bins=12, color='orange', histtype='stepfilled', label=f'k - {round(ibg, 2)} < bg < {round(ibg+0.2, 2)}')
    #        dfP_sel = dfP.query(f'{ibg} < bg < {ibg+0.2}')
    #        plt.hist(dfP_sel[f'ClSizeL{iL}'], alpha=0.5, range=[-0.5, 12.5], bins=12, color='g', histtype='stepfilled', label=f'p - {round(ibg, 2)} < bg < {round(ibg+0.2, 2)}')
    #        plt.legend(frameon=False, fontsize=12, loc='best')
    #        plt.xlabel(f'CL size L{iL}')
    #        plt.ylabel('counts')
    #        plt.savefig(f'{OutPutDir}/ClSizeL{iL}_bgCut_{round(ibg, 2)}_{round(ibg+0.2, 2)}.png')
    #        plt.close('all')  


def optimization(trial: Trial, inputCfg, TrainTestData):
    HyperPars = {'tree_method':'hist',  # this parameter means using the GPU when training our model to speedup the training process
                 'n_jobs':40,
                 'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.5, 0.7, 0.9, 1.0]),
                 'subsample': trial.suggest_categorical('subsample', [0.4, 0.6, 0.8, 1.0]),
                 'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.014, 0.016, 0.018, 0.02, 0.04, 0.06]),
                 'n_estimators': trial.suggest_categorical('n_estimators', [400, 500, 700, 800, 1000, 1200]),
                 'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 9, 11, 13, 15, 17, 20]),
                 'min_child_weight': trial.suggest_int('min_child_weight', 100, 300),
                 "eval_metric": trial.suggest_categorical('eval_metric', ["logloss", "rmse", "rmsle"]),
                 "objective": "reg:squarederror"
                }
    TrainCols = inputCfg['ml']['training_columns']
    modelReg = xgb.XGBRegressor(**HyperPars)
    if ('p' in TrainCols or 'pt' in TrainCols):
        print('\033[91mERROR: p or pt in training columns! Exit.\033[0m')
        sys.exit()
    if not isinstance(TrainCols, list):
        print('\033[91mERROR: training columns must be defined!\033[0m')
        sys.exit()
    if not isinstance(HyperPars, dict):
        print('\033[91mERROR: hyper-parameters must be defined or be an emy dict!\033[0m')
        sys.exit()
   
    # model definition
    ModelHandl = ModelHandler(modelReg, TrainCols)

    # train and test the model
    yPredTest = ModelHandl.train_test_regrressor(TrainTestData, return_prediction=True, nfold=5, njobs=3)

    # cross validation
    print('\033[93mCross validation!\033[0m')
    cv_score = cross_val_score(ModelHandl.get_original_model(), TrainTestData[0], TrainTestData[1], cv=5, scoring='neg_mean_squared_error')
    for icv in cv_score:
        print(f'\033[93mcross_val_score: {icv:.6f}\033[0m')
    print('\033[93m==============================\033[0m')
    return abs(np.mean(cv_score))


def main(): #pylint: disable=too-many-statements
    # read config file
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)

    print('Loading and preparing data files: ...', end='\r')
    Data = inputCfg['input']['data']

    print(f'\n\033[94mStarting ML analysis\033[0m')
    OutPutDir = inputCfg['output']['dir']
    if os.path.isdir(OutPutDir):
        print((f'\033[93mWARNING: Output directory \'{OutPutDir}\' already exists,'
                ' overwrites possibly ongoing!\033[0m'))
    else:
        os.makedirs(OutPutDir)
    # data preparation
    #_____________________________________________
    TrainTestData, ApplDf = data_prep(inputCfg, OutPutDir, Data)

    # hyper-pars. optimization
    #_____________________________________________ 
    if inputCfg['ml']['hyper_par_opt']['do_hyp_opt']:
        print('\033[93mPerforming hyper parameter optimisation\033[0m')
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial : optimization(trial, inputCfg, TrainTestData), n_trials=10, show_progress_bar=True)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(f'{OutPutDir}/OptHistoryOptuna.png')

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(f'{OutPutDir}/ParamImportanceOptuna.png')
        
        fig = optuna.visualization.plot_intermediate_values(study)
        fig.write_image(f'{OutPutDir}/LearningCurvesOptuna.png')
        
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(f'{OutPutDir}/ParalleCoordinatesOptuna.png')

        fig = optuna.visualization.plot_contour(study)
        fig.write_image(f'{OutPutDir}/ContourPlotOptuna.png')

        sys.stdout = open(f"{OutPutDir}/HyperParOptOptuna.txt", "a")
        print('Best trial:', study.best_trial.params)

    # training, testing
    #_____________________________________________
    ModelHandl, yPredTest = regression(inputCfg, OutPutDir, TrainTestData)

    # model application
    #_____________________________________________
    appl(inputCfg, OutPutDir, ModelHandl, ApplDf, TrainTestData[2], yPredTest)

    # delete dataframes to release memory
    del TrainTestData, ApplDf

main()
