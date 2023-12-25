#
#

import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod
from alive_progress import alive_bar

import xgboost as xgb
import optuna
from hipe4ml.model_handler import ModelHandler
from ROOT import TH1D, TCanvas, TFile

import sys
sys.path.append('..')
from src.dataVisual import DataVisual, DataVisualSlow
from utils.particles import particleMasses, particlePDG
from utils.matplotlibToRoot import saveMatplotlibToRootFile
from utils.formatPath import addSuffix

# create a model and train it

class Regressor(ABC):

    def __init__(self, cfg, outFile, TrainTestData):
        self.cfg = cfg
        self.outFile = outFile
        self.TrainSet = TrainTestData[0]
        self.yTrain = TrainTestData[1]
        self.TestSet = TrainTestData[2]
        self.yTest = TrainTestData[3]

    @abstractmethod
    def initializeModel(self):  pass
        
    @abstractmethod
    def trainModel(self):       pass

    @abstractmethod
    def visualizeResults(self, cfgGeneral): pass    

    @abstractmethod
    def saveModel(self):        pass
        
    @classmethod
    def createRegressor(cls, cfg, outFile, TrainTestData):
        if cfg['MLtype'] == 'XGB':          return XGBoostRegressor(cfg, outFile, TrainTestData)
        elif cfg['MLtype'] == 'XGBoptuna':  return XGBoostRegressorOptuna(cfg, outFile, TrainTestData)
        else:                               raise ValueError(f"Unknown ML type: {cfg['MLtype']}")

class XGBoostRegressor(Regressor):

    def __init__(self, cfg, outFile, TrainTestData):

        super().__init__(cfg, outFile, TrainTestData)
        self.initializeModel()

    def initializeModel(self):
        
        self.model = xgb.XGBRegressor(**self.cfg['defaultHyperparameters'], random_state=self.cfg['randomState'])

    def trainModel(self):
        
        if self.cfg['weights']['type'] == 'PandSpecies':     self.model.fit(self.TrainSet[self.cfg['regressionColumns']], self.yTrain, sample_weight=self.TrainSet['weightsPS'])
        elif self.cfg['weights']['type'] == 'None':          self.model.fit(self.TrainSet[self.cfg['regressionColumns']], self.yTrain)

    def visualizeResults(self, cfgGeneral):

        directory = self.outFile.mkdir('ML')

        # variable importance
        fig, ax = plt.subplots(figsize=(20, 10))
        xgb.plot_importance(self.model, ax=ax)
        
        # restore original yticks
        yticks = list(ax.get_yticklabels())
        dictColumns = dict(enumerate(self.cfg['regressionColumns']))
        yticks = [yticks[i].get_text().lstrip('f') for i in range(len(yticks))]
        yticks = [dictColumns[int(ytick)] for ytick in yticks]
        ax.set_yticklabels(yticks)
        saveMatplotlibToRootFile(fig, directory, 'variableImportance')
        
        plt.close('all')

        # test regression model
        self.TestSet = self.TestSet.with_columns(pl.Series( name='betaML', values=self.model.predict(self.TestSet[self.cfg['regressionColumns']]) ))
        self.TestSet = self.TestSet.with_columns(deltaBeta=((pl.col('betaML') - pl.col('beta')) / pl.col('beta')))

        # visualize regression results
        directory = self.outFile.mkdir('regression')
        dv = DataVisual(self.TestSet, directory, cfgGeneral['cfgVisualFile'])
        dv.createHistos()
        dv.create2DHistos()
        dvs = DataVisualSlow(self.TestSet, directory)
        dvs.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
        dvs.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

        del directory, dv

        for particle in self.cfg['species']:
            directory = self.outFile.mkdir(f'{particle}_regression')
            dv = DataVisualSlow(self.TestSet.filter(pl.col('partID') == particlePDG[particle]), directory)
            dv.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
            dv.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

            del directory, dv 

    def saveModel(self):

        pass

class XGBoostRegressorOptuna(Regressor):

    def __init__(self, cfg, outFile, TrainTestData):
        super().__init__(cfg, outFile, TrainTestData)
        self.study = None
        self.initializeModel()

    def initializeModel(self):

        modelRef = xgb.XGBRegressor(**self.cfg['defaultHyperparameters'], random_state=self.cfg['randomState'])
        modelHandler = ModelHandler(modelRef, self.cfg['regressionColumns'])

        # hyperparameters optimization
        with alive_bar(title='Optuna parameter optimisation...') as bar:
            self.study = modelHandler.optimize_params_optuna([self.TrainSet[self.cfg['regressionColumns']], self.yTrain, self.TestSet[self.cfg['regressionColumns']], self.yTest], self.cfg['hyperparametersRange'],
                                                direction='maximize', n_trials=25, cross_val_scoring='neg_root_mean_squared_error')
        
        hyperparameters = self.cfg['defaultHyperparameters']
        hyperparameters.update(self.study.best_trial.params)

        self.model = xgb.XGBRegressor(**(hyperparameters), random_state=self.cfg['randomState'])
        
    def trainModel(self):
        
        if self.cfg['weights']['type'] == 'PandSpecies':    self.model.fit(self.TrainSet[self.cfg['regressionColumns']], self.yTrain, sample_weight=self.TrainSet['weightsPS'])
        elif self.cfg['weights']['type'] == 'None':         self.model.fit(self.TrainSet[self.cfg['regressionColumns']], self.yTrain)

    def visualizeResults(self, cfgGeneral):

        directory = self.outFile.mkdir('ML')

        # variable importance
        fig, ax = plt.subplots(figsize=(20, 10))
        xgb.plot_importance(self.model, ax=ax)
        
        # restore original yticks
        yticks = list(ax.get_yticklabels())
        dictColumns = dict(enumerate(self.cfg['regressionColumns']))
        yticks = [yticks[i].get_text().lstrip('f') for i in range(len(yticks))]
        yticks = [dictColumns[int(ytick)] for ytick in yticks]
        ax.set_yticklabels(yticks)
        saveMatplotlibToRootFile(fig, directory, 'variableImportance')

        # visualize optuna studies
        outFileOptuna = addSuffix(self.cfg['outFileOptuna'], f"_{self.cfg['weights']['type']}")
        with open(outFileOptuna, 'w') as f:  f.write(f'Best trial: \n{self.study.best_trial.params}')

        fig = optuna.visualization.plot_optimization_history(self.study)
        saveMatplotlibToRootFile(fig, directory, 'plotOptimizationHistory')

        fig = optuna.visualization.plot_param_importances(self.study)
        saveMatplotlibToRootFile(fig, directory, 'plotParamImportances')

        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        saveMatplotlibToRootFile(fig, directory, 'plotParallelCoordinate')

        fig = optuna.visualization.plot_contour(self.study)
        saveMatplotlibToRootFile(fig, directory, 'plotContour')


        # test regression model
        TestSet = self.TestSet.with_columns(pl.Series( name='betaML', values=self.model.predict(self.TestSet[self.cfg['regressionColumns']]) ))
        TestSet = self.TestSet.with_columns(deltaBeta=((pl.col('betaML') - pl.col('beta')) / pl.col('beta')))

        # visualize regression results
        directory = self.outFile.mkdir('regression')
        dv = DataVisual(TestSet, directory, cfgGeneral['cfgVisualFile'])
        dv.createHistos()
        dv.create2DHistos()
        dvs = DataVisualSlow(TestSet, directory)
        dvs.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
        dvs.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

        del directory, dv

        for particle in self.cfg['species']:
            directory = self.outFile.mkdir(f'{particle}_regression')
            dv = DataVisualSlow(TestSet.filter(pl.col('partID') == particlePDG[particle]), directory)
            dv.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
            dv.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

            del directory, dv 

    def saveModel(self):

        pass


# evaluate the performance of the model

def hyperparameterScoreCanvas(hyperparameterImportance):

    hyperparameterImportance = sorted(hyperparameterImportance.items(), key=lambda x: x[1], reverse=True)
    h = TH1D('h', '', len(hyperparameterImportance), 0, len(hyperparameterImportance))
    for ientry, (hyperparameter, score) in enumerate(hyperparameterImportance.items()):
        h.SetBinContent(ientry+1, score)
        h.SetBinLabel(ientry+1, hyperparameter)

    c = TCanvas('hyperparameterImportance', 'Hyperparamter importance; Score; Hyperparameter', 800, 600)
    h.Draw('bar hist')
    
    return c

        