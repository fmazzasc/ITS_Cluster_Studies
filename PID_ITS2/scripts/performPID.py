#
#   Script to perform PID using machine learning techniques
#

import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from src.preprocessing import *
from src.dataVisual import DataVisual, DataVisualSlow
from src.loadData import LoadData
from src.mlPerformance import *

from utils.particles import particleMasses, particlePDG
from utils.color import color
from utils.formatPath import addSuffix

from sklearn.model_selection import train_test_split
import xgboost as xgb

from ROOT import TFile, TH1D

# PID step by step

def preprocess(cfgGeneral):
    '''
    Preprocess the data and create visualizations

    Parameters
    ----------
    inData (pandas.DataFrame): input data
    '''

    print(color.BOLD+color.YELLOW+'\nPreprocessing data...'+color.END)
    with open(cfgGeneral['cfgFile'][cfgGeneral['opt']], 'r') as f:    cfg = yaml.safe_load(f)

    RegressionSet = LoadData(cfg['inFiles'])
    ApplicationSet = LoadData(cfg['applicationFiles'])
    if ApplicationSet.is_empty():   RegressionSet, ApplicationSet = train_test_split(RegressionSet, test_size=cfg['applicationSize'], random_state=cfg['randomState'])

    # preprocess train and test data
    dpRegression = DataPreprocessor.CreatePreprocessor(RegressionSet, cfg, cfgGeneral['opt'])
    dpRegression.Preprocess()
    dpRegression.ApplyCuts()
    dpRegression.ParticleID()
    dpRegression.CleanData()
    dpRegression.DropUnidentified()

    TrainSet, TestSet = train_test_split(dpRegression.data, test_size=cfg['testSize'], random_state=cfg['randomState'])

    # define weights for training
    dpTrainSet = DataPreprocessor.CreatePreprocessor(TrainSet, cfg, cfgGeneral['opt'])
    dpTrainSet.DefineWeights()   
    TrainSet = dpTrainSet.data

    # visualize train and test data
    outPath = addSuffix(cfg['outFilePreprocess'], f"_{cfg['weights']['type']}")
    outFile = TFile(outPath, 'RECREATE')
    print('Creating output file:'+color.UNDERLINE+color.BLUE+f'{outPath}'+color.END)

    for set, label in zip([TrainSet, TestSet], ['train', 'test']):
        directory = outFile.mkdir(label)
        dv = DataVisual(set, directory, cfgGeneral['cfgVisualFile'])
        dv.createHistos()
        dv.create2DHistos()
        if label == 'train':    dv.visualWeights(cfg)

        del directory, dv
    
    for particle in cfg['species']:
        directory = outFile.mkdir(f'{particle}_train')
        dv = DataVisual(TrainSet.filter(pl.col('partID') == particlePDG[particle]), directory, cfgGeneral['cfgVisualFile'])
        dv.createHistos()
        dv.create2DHistos()

        del directory, dv

    outFile.Close()

    # preprocess application data
    dpApplication = DataPreprocessor.CreatePreprocessor(ApplicationSet, cfg, cfgGeneral['opt'])
    dpApplication.Preprocess()
    dpApplication.ApplyCuts()
    dpApplication.ParticleID()
    dpApplication.CleanData()
    ApplicationSet = dpApplication.data
    
    return TrainSet, TestSet, ApplicationSet

def regression(cfgGeneral, TrainSet, TestSet):

    print(color.BOLD+color.YELLOW+'\nPerforming regression...'+color.END)
    with open(cfgGeneral['cfgFile'][cfgGeneral['opt']], 'r') as f:    cfg = yaml.safe_load(f)

    TrainSet = TrainSet.with_columns(betaML=np.nan)
    TestSet = TestSet.with_columns(betaML=np.nan)
    
    outPath = addSuffix(cfg['outFileRegression'], f"_{cfg['weights']['type']}")
    outFile = TFile(outPath, 'RECREATE')
    print('Creating output file:'+color.UNDERLINE+color.BLUE+f'{outPath}'+color.END)

    regressor = Regressor.createRegressor(cfg, outFile, [TrainSet, TrainSet['beta'], TestSet, TestSet['beta']])
    regressor.trainModel()
    regressor.visualizeResults(cfgGeneral)
    #regressor.saveModel()

    outFile.Close()
    return regressor.model

def application(cfgGeneral, ApplicationSet, model):

    print(color.BOLD+color.YELLOW+'\nApplying regression model...'+color.END)
    with open(cfgGeneral['cfgFile'][cfgGeneral['opt']], 'r') as f:    cfg = yaml.safe_load(f)
    outPath = addSuffix(cfg['outFileApplication'], f"_{cfg['weights']['type']}")
    outFile = TFile(outPath, 'RECREATE')
    print('Creating output file:'+color.UNDERLINE+color.BLUE+f'{outPath}'+color.END)

    # apply regression model
    ApplicationSet = ApplicationSet.with_columns(pl.Series( name='betaML', values=model.predict(ApplicationSet[cfg['regressionColumns']]) ))
    ApplicationSet = ApplicationSet.with_columns(deltaBeta=((pl.col('betaML') - pl.col('beta')) / pl.col('beta')))

    # visualize application results
    directory = outFile.mkdir('application')
    dv = DataVisual(ApplicationSet, directory, cfgGeneral['cfgVisualFile'])
    dv.createHistos()
    dv.create2DHistos()
    dvs = DataVisualSlow(ApplicationSet, directory)
    dvs.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
    dvs.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
    dvs.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
    dvs.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

    del directory, dv

    for particle in cfg['species']:
        directory = outFile.mkdir(f'{particle}_application')
        dv = DataVisualSlow(ApplicationSet.filter(pl.col('partID') == particlePDG[particle]), directory)
        dv.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
        dv.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
        dv.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
        dv.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

        del directory, dv

    outFile.Close()




# Launch PID

def performPID(cfgPath):

    with open(cfgPath, 'r') as f:   cfgGeneral = yaml.safe_load(f)
    TrainSet, TestSet, ApplicationSet = preprocess(cfgGeneral)
    model = regression(cfgGeneral, TrainSet, TestSet)
    application(cfgGeneral, ApplicationSet, model)

    pass

if __name__ == '__main__':
    
    cfgPath = '/home/galucia/ITS_Cluster_Studies/PID_ITS2/configs/cfgPID_general.yml'
    
    performPID(cfgPath)
