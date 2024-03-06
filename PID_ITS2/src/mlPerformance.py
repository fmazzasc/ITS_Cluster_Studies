#
#

import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod
from alive_progress import alive_bar

import xgboost as xgb
import optuna
import shap
#shap.initjs()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

from hipe4ml.model_handler import ModelHandler
from ROOT import TH1D, TCanvas, TFile, TGraph

import sys
sys.path.append('..')
from src.dataVisual import DataVisual, DataVisualSlow
from utils.particles import particleMasses, particlePDG
from utils.matplotlibToRoot import saveMatplotlibToRootFile
from utils.formatPath import addSuffix

# Boosted Decision Trees

class Regressor(ABC):

    def __init__(self, cfg, outFile, TrainTestData):
        self.cfg = cfg
        self.outFile = outFile
        self.TrainSet = TrainTestData[0]
        self.yTrain = TrainTestData[1]
        self.TestSet = TrainTestData[2]
        self.yTest = TrainTestData[3]

        self.MLdirectory = None
        self.regressionDirectory = None
        self.particleDirectories = None


    @abstractmethod
    def initializeModel(self):  pass
        
    @abstractmethod
    def trainModel(self):       pass

    @abstractmethod
    def MLResults(self, cfgGeneral): 

        # create dataset joining train and test
        RegressionSet = pl.concat([self.TrainSet[self.cfg['regressionColumns']], self.TestSet[self.cfg['regressionColumns']]], rechunk=True)
        yRegression = pl.concat([self.yTrain, self.yTest], rechunk=True)

        # variable importance
        if self.cfg['SHAP']:
            with alive_bar(title='Calculating variable importance...') as bar:

                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer(RegressionSet)
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, RegressionSet, plot_type='bar', show=False, color='cornflowerblue')

                vals = np.abs(shap_values.values).mean(0)
                featureImportance = {'columnName': RegressionSet.columns, 'featureImportanceVals': vals}
                featureImportance = pl.DataFrame(featureImportance)
                featureImportance.sort(by=['featureImportanceVals'], descending=True)

                ax.set_yticklabels(featureImportance['columnName'])
                ax.set_xlabel('|SHAP value|')
                saveMatplotlibToRootFile(fig, self.MLdirectory, 'SHAPvariableImportance')

            plt.close('all')

    @abstractmethod
    def testResults(self, cfgGeneral): 

        # test regression model
        self.TestSet = self.TestSet.with_columns(pl.Series( name='betaML', values=self.model.predict(self.TestSet[self.cfg['regressionColumns']]) ))
        self.TestSet = self.TestSet.with_columns(deltaBeta=((pl.col('betaML') - pl.col('beta')) / pl.col('beta')))

        # visualize regression results
        dv = DataVisual(self.TestSet, self.regressionDirectory, cfgGeneral['cfgVisualFile'])
        dv.createHistos()
        dv.create2DHistos()
        dvs = DataVisualSlow(self.TestSet, self.regressionDirectory)
        dvs.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
        dvs.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)
        dvs.create2DHisto('pTPC', 'betaML', '#beta_{ML} vs #it{p}_{TPC}; #it{p}_{TPC} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

        del dv

        for particle in self.cfg['species']:
            dv = DataVisualSlow(self.TestSet.filter(pl.col('partID') == particlePDG[particle]), self.particleDirectories[particle])
            dv.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
            dv.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)
            dv.create2DHisto('pTPC', 'betaML', '#beta_{ML} vs #it{p}_{TPC}; #it{p}_{TPC} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

            del dv 

    @abstractmethod
    def purityAndEfficiency(self, cfgGeneral): 
        
        '''
        Define the purity and efficiency of the model for different particles.
        This is evaluated on the test set.

        '''
        
        pmin = self.cfg['EffPur']['pmin']
        pmax = self.cfg['EffPur']['pmax']
        for particle in self.cfg['species']:    self.purityAndEfficiencySingleSpecies(pmin, pmax, particle)

    @abstractmethod
    def purityAndEfficiencySingleSpecies(self, pmin, pmax, particle):

        inputData = self.TestSet.filter((pl.col('p') > pmin) &
                                        (pl.col('p') < pmax))
        inputData = inputData.with_columns(betaE=(pl.col('p') / np.sqrt(pl.col('p')**2 + particleMasses['E']**2)),
                                           betaPi=(pl.col('p') / np.sqrt(pl.col('p')**2 + particleMasses['Pi']**2)),
                                           betaK=(pl.col('p') / np.sqrt(pl.col('p')**2 + particleMasses['K']**2)),
                                           betaP=(pl.col('p') / np.sqrt(pl.col('p')**2 + particleMasses['P']**2)),
                                           betaDeu=(pl.col('p') / np.sqrt(pl.col('p')**2 + particleMasses['Deu']**2))
                                           )
        inputData = inputData.with_columns(deltaBetaE=(pl.col('betaE') - pl.col('betaML')),
                                           deltaBetaPi=(pl.col('betaPi') - pl.col('betaML')),
                                           deltaBetaK=(pl.col('betaK') - pl.col('betaML')),
                                           deltaBetaP=(pl.col('betaP') - pl.col('betaML')),
                                           deltaBetaDeu=(pl.col('betaDeu') - pl.col('betaML'))
                                           ) 

        inputParticle = inputData.filter(pl.col('partID') == particlePDG[particle])
        purity = []
        efficiency = []
        thresholds = np.arange(self.cfg['EffPur']['thresholds'][particle][0], 
                                  self.cfg['EffPur']['thresholds'][particle][1], 
                                  self.cfg['EffPur']['thresholds'][particle][2])

        for threshold in thresholds:

            tmpInputData = inputData.filter(np.abs(pl.col(f'deltaBeta{particle}')) < threshold)
            tmpInputParticle = inputParticle.filter(np.abs(pl.col(f'deltaBeta{particle}')) < threshold)

            tmpPurity = tmpInputParticle.shape[0] / tmpInputData.shape[0]
            tmpEfficiency = tmpInputParticle.shape[0] / inputParticle.shape[0]

            purity.append(tmpPurity)
            efficiency.append(tmpEfficiency)
    
        self.particleDirectories[particle].cd()
        graph = TGraph(len(thresholds), np.asarray(thresholds, dtype=float), np.asarray(efficiency, dtype=float))
        graph.SetTitle(r'; Threshold ; Efficiency')
        graph.Write(f'efficiency_{particle}')

        graph = TGraph(len(thresholds), np.asarray(thresholds, dtype=float), np.asarray(purity, dtype=float))
        graph.SetTitle(r'; Threshold ; Purity')
        graph.Write(f'purity_{particle}')

        graph = TGraph(len(efficiency), np.asarray(efficiency, dtype=float), np.asarray(purity, dtype=float))
        graph.SetTitle(r'; Efficiency ; Purity')
        graph.Write(f'purityVsEfficiency_{particle}')
        
    @abstractmethod
    def visualizeResults(self, cfgGeneral): 

        # create output directories

        self.MLdirectory = self.outFile.mkdir('ML')
        self.regressionDirectory = self.outFile.mkdir('regression')
        self.particleDirectories = {}
        for particle in self.cfg['species']:
            dir = self.outFile.mkdir(f'{particle}_regression')
            self.particleDirectories[particle] = dir

        self.MLResults(cfgGeneral)
        self.testResults(cfgGeneral) 
        self.purityAndEfficiency(cfgGeneral)   

    @abstractmethod
    def applyOnTrainSet(self, cfgGeneral): pass

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
        
        print('Training model...')
        if self.cfg['weights']['type'] == 'PandSpecies':     self.model.fit(self.TrainSet[self.cfg['regressionColumns']], self.yTrain, sample_weight=self.TrainSet['weightsPS'])
        elif self.cfg['weights']['type'] == 'None':          self.model.fit(self.TrainSet[self.cfg['regressionColumns']], self.yTrain)

    def MLResults(self, cfgGeneral):            super().MLResults(cfgGeneral)

    def testResults(self, cfgGeneral):          super().testResults(cfgGeneral)

    def purityAndEfficiency(self, cfgGeneral):  super().purityAndEfficiency(cfgGeneral)

    def purityAndEfficiencySingleSpecies(self, pmin, pmax, particle):  
        super().purityAndEfficiencySingleSpecies(pmin, pmax, particle)

    def visualizeResults(self, cfgGeneral):     super().visualizeResults(cfgGeneral) 

    def applyOnTrainSet(self, cfgGeneral):

        # apply on train set
        self.TrainSet = self.TrainSet.with_columns(pl.Series( name='betaML', values=self.model.predict(self.TrainSet[self.cfg['regressionColumns']]) ))
        self.TrainSet = self.TrainSet.with_columns(deltaBeta=((pl.col('betaML') - pl.col('beta')) / pl.col('beta')))

        # visualize regression results
        directory = self.outFile.mkdir('regression_train')
        dv = DataVisual(self.TrainSet, directory, cfgGeneral['cfgVisualFile'])
        dv.createHistos()
        dv.create2DHistos()
        dvs = DataVisualSlow(self.TrainSet, directory)
        dvs.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
        dvs.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)
        dvs.create2DHisto('pTPC', 'betaML', '#beta_{ML} vs #it{p}_{TPC}; #it{p}_{TPC} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

        del directory, dv

        for particle in self.cfg['species']:
            directory = self.outFile.mkdir(f'{particle}_regression_train')
            dv = DataVisualSlow(self.TrainSet.filter(pl.col('partID') == particlePDG[particle]), directory)
            dv.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
            dv.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)
            dv.create2DHisto('pTPC', 'betaML', '#beta_{ML} vs #it{p}_{TPC}; #it{p}_{TPC} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

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
        
        print('Training model...')
        if self.cfg['weights']['type'] == 'PandSpecies':    self.model.fit(self.TrainSet[self.cfg['regressionColumns']], self.yTrain, sample_weight=self.TrainSet['weightsPS'])
        elif self.cfg['weights']['type'] == 'None':         self.model.fit(self.TrainSet[self.cfg['regressionColumns']], self.yTrain)

    def MLResults(self, cfgGeneral):     
        
        super().MLResults(cfgGeneral)

        # visualize optuna studies
        outFileOptuna = addSuffix(self.cfg['outFileOptuna'], f"_{self.cfg['weights']['type']}")
        with open(outFileOptuna, 'w') as f:  f.write(f'Best trial: \n{self.study.best_trial.params}')

        fig = optuna.visualization.plot_optimization_history(self.study)
        saveMatplotlibToRootFile(fig, self.MLdirectory, 'plotOptimizationHistory')

        fig = optuna.visualization.plot_param_importances(self.study)
        saveMatplotlibToRootFile(fig, self.MLdirectory, 'plotParamImportances')

        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        saveMatplotlibToRootFile(fig, self.MLdirectory, 'plotParallelCoordinate')

        fig = optuna.visualization.plot_contour(self.study)
        saveMatplotlibToRootFile(fig, self.MLdirectory, 'plotContour')

    def testResults(self, cfgGeneral):      super().testResults(cfgGeneral)

    def purityAndEfficiency(self, cfgGeneral):  super().purityAndEfficiency(cfgGeneral)

    def purityAndEfficiencySingleSpecies(self, pmin, pmax, particle):
        super().purityAndEfficiencySingleSpecies(pmin, pmax, particle)

    def visualizeResults(self, cfgGeneral):

        # create output directories

        self.MLdirectory = self.outFile.mkdir('ML')
        self.regressionDirectory = self.outFile.mkdir('regression')
        self.particleDirectories = {}
        for particle in self.cfg['partciles']:
            dir = self.outFile.mkdir(f'{particle}_regression')
            self.particleDirectories[particle] = dir

        self.MLResults(cfgGeneral)
        self.testResults(cfgGeneral) 
        self.purityAndEfficiency(cfgGeneral)   

        

    def applyOnTrainSet(self, cfgGeneral):

        # apply on train set
        self.TrainSet = self.TrainSet.with_columns(pl.Series( name='betaML', values=self.model.predict(self.TrainSet[self.cfg['regressionColumns']]) ))
        self.TrainSet = self.TrainSet.with_columns(deltaBeta=((pl.col('betaML') - pl.col('beta')) / pl.col('beta')))

        # visualize regression results
        directory = self.outFile.mkdir('regression_train')
        dv = DataVisual(self.TrainSet, directory, cfgGeneral['cfgVisualFile'])
        dv.createHistos()
        dv.create2DHistos()
        dvs = DataVisualSlow(self.TrainSet, directory)
        dvs.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
        dvs.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)
        dvs.create2DHisto('pTPC', 'betaML', '#beta_{ML} vs #it{p}_{TPC}; #it{p}_{TPC} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

        del directory, dv

        for particle in self.cfg['species']:
            directory = self.outFile.mkdir(f'{particle}_regression_train')
            dv = DataVisualSlow(self.TrainSet.filter(pl.col('partID') == particlePDG[particle]), directory)
            dv.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
            dv.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)
            dv.create2DHisto('pTPC', 'betaML', '#beta_{ML} vs #it{p}_{TPC}; #it{p}_{TPC} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

            del directory, dv

    def saveModel(self):

        pass



# Fully Connected Neural Network

class FCNN(nn.Module):

    def __init__(self, mode, inputSize, hiddenSize, outputSize):
        
        super(FCNN, self).__init__()

        # accepted modes: 'regression', 'classification'
        self.mode = mode
        if mode != 'regression' and mode != 'classification':    raise ValueError(f"Unknown mode: {mode}")

        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        if self.mode == 'classification':    x = F.log_softmax(x, dim=1)

        return x

class CustomDataset(Dataset):

    def __init__(self, data, cfg, target):
        '''
        Parameters
        ----------
            TrainTestData (pl.DataFrame): input data (all columns)
            cfg (yml): configuration dictionary
            target (str): target column name (partID)
        '''
        
        self.data = data
        self.regressionData = data[cfg['regressionColumns']].to_numpy()
        self.target = data[target].to_numpy()

        #labelEncoder = LabelEncoder()
        #self.target = labelEncoder.fit_transform(self.target)

        self.target = self.target.reshape(-1, 1)
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'regressionData': torch.tensor(self.regressionData[idx], dtype=torch.float),
            'target': torch.tensor(self.target[idx], dtype=torch.float)
        }

        return sample


class NeuralNetwork(ABC):

    def __init__(self, cfg, outFile, TrainTestData, target):
        
        self.cfg = cfg
        self.outFile = outFile
        self.TrainSet = TrainTestData[0]
        self.TestSet = TrainTestData[2]
        self.target = target
        self.TrainDataset = None
        self.losses = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg['randomState'])
        torch.set_num_threads(self.cfg['NN']['nThreads'])

        self.initializeModel()

    @abstractmethod
    def initializeModel(self):  pass

    @abstractmethod
    def trainModel(self): 

        trainLoader = DataLoader(self.TrainDataset, batch_size=self.cfg['NN']['batchSize'], shuffle=True)      

        with alive_bar(title='Training neural network...') as bar:
            for epoch in range(self.cfg['NN']['nEpochs']):
                for i, batch in enumerate(trainLoader):

                    inputs, labels = batch['regressionData'], batch['target']
                    outputs = self.model.forward(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    if i == 0:  
                        self.losses.append(loss.item())
                        print(f'Epoch {epoch+1}/{self.cfg["NN"]["nEpochs"]}, Batch {i+1}/{len(trainLoader)}, Loss: {loss.item()}')

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                bar()


class NeuralNetworkClassifier(NeuralNetwork):

    def __init__(self, cfg, outFile, TrainTestData, target):
        '''
        Parameters
        ----------
            TrainTestData (pl.DataFrame): input data (all columns)
            cfg (yml): configuration dictionary
            outFile (str): output file name
            target (str): target column name (partID)
        '''
        
        super().__init__(cfg, outFile, TrainTestData, target)
        self.TrainDataset = CustomDataset(self.TrainSet, self.cfg, self.target)

        print(self.TrainDataset.regressionData.shape)
        print(self.TrainDataset.target.shape)
    
    def initializeModel(self):

        self.model = FCNN('classification', len(self.cfg['regressionColumns']), self.cfg['NN']['hiddenSize'], len(self.cfg['species'])) #.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)
        self.criterion = nn.CrossEntropyLoss()
        #if cfg['weights']['type'] == 'PandSpecies':    self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.TrainSet['weightsPS']))

    def trainModel(self):    super().trainModel()
    
    def testResults(self, cfgGeneral):

        testDataset = CustomDataset(self.TestSet, self.cfg, 'partID')
        testLoader = DataLoader(testDataset, batch_size=self.cfg['NN']['batchSize'], shuffle=True)

        self.model.eval()

        predictions = []

        with torch.no_grad():
            for batch in testLoader:
                inputs, labels = batch['regressionData'], batch['target']

                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)

                predictions.append(probabilities)
        
        predictions = torch.cat(predictions, dim=0)
        print([f'prob{part}' for part in self.cfg['species']])
        predictionsDf = pl.DataFrame(np.swapaxes(predictions.numpy(), 0, 1), schema=[f'prob{part}' for part in self.cfg['species']])

        self.TestSet = pl.concat([self.TestSet, predictionsDf], how='horizontal')
        print(self.TestSet.columns)

        # visualize regression results
        directory = self.outFile.mkdir('regression')
        dv = DataVisual(self.TestSet, directory, cfgGeneral['cfgVisualFile'])
        dv.createHistos()
        dv.create2DHistos()
        dvs = DataVisualSlow(self.TestSet, directory)
        for particle in self.cfg['species']:
            dvs.createHisto(f'prob{particle}', f'Probability of {particle}; Probability; Counts', 100, 0, 1)
            dvs.create2DHisto('p', f'prob{particle}', f'Probability of {particle} vs #it{{p}}; #it{{p}} (GeV/#it{{c}}); Probability', 1500, 0, 1.5, 1000, 0, 1)
            #dvs.create2DHisto('p', f'prob{particle}', f'Probability of {particle} vs #it{p}; #it{p} (GeV/#it{c}); Probability', 1500, 0, 1.5, 1000, 0, 1)

        graph = TGraph(len(self.losses), np.asarray(np.arange(self.cfg['NN']['nEpochs']), dtype=float), np.asarray(self.losses, dtype=float))
        graph.SetTitle(r'; Epoch ; Loss')
        graph.Write('loss')
        
        del directory, dv


class NeuralNetworkRegressor(NeuralNetwork):

    def __init__(self, cfg, outFile, TrainTestData, target):
        '''
        Parameters
        ----------
            TrainTestData (pl.DataFrame): input data (all columns)
            cfg (yml): configuration dictionary
            outFile (str): output file name
            target (str): target column name (partID)
        '''

        self.cfg = cfg
        self.outFile = outFile
        self.TrainSet = TrainTestData[0]
        self.TestSet = TrainTestData[2]
        self.target = target
        self.TrainDataset = None
        self.losses = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg['randomState'])
        torch.set_num_threads(self.cfg['NN']['nThreads'])

        self.initializeModel()

        #super().__init__(cfg, outFile, TrainTestData, target)
        self.TrainDataset = CustomDataset(self.TrainSet, self.cfg, self.target)

        print(self.TrainDataset.regressionData.shape)
        print(self.TrainDataset.target.shape)

        self.initializeModel()
    
    def initializeModel(self):

        self.learningRate = self.cfg['NN']['learningRate']

        self.model = FCNN('regression', len(self.cfg['regressionColumns']), self.cfg['NN']['hiddenSize'], 1) #.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)
        self.criterion = nn.MSELoss()

    def trainModel(self):   # super().trainModel()

        trainLoader = DataLoader(self.TrainDataset, batch_size=self.cfg['NN']['batchSize'], shuffle=True)      

        with alive_bar(title='Training neural network...') as bar:
            for epoch in range(self.cfg['NN']['nEpochs']):
                for i, batch in enumerate(trainLoader):

                    inputs, labels = batch['regressionData'], batch['target']
                    outputs = self.model.forward(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    if i == 0:  
                        self.losses.append(loss.item())
                        print(f'Epoch {epoch+1}/{self.cfg["NN"]["nEpochs"]}, Batch {i+1}/{len(trainLoader)}, Loss: {loss.item()}')

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                bar()
    
    def testResults(self, cfgGeneral):

        testDataset = CustomDataset(self.TestSet, self.cfg, 'partID')
        testLoader = DataLoader(testDataset, batch_size=self.cfg['NN']['batchSize'], shuffle=True)

        self.model.eval()

        betaML = []

        with torch.no_grad():
            for batch in testLoader:
                inputs, labels = batch['regressionData'], batch['target']

                outputs = self.model(inputs)
                output = torch.softmax(outputs, dim=1)

                betaML.append(output.numpy())
        
        # test regression model
        #betaML = np.swapaxes(torch.cat(betaML, dim=0).numpy(), 0, 1)
        betaML = np.concatenate(betaML, axis=0)
        print(betaML)

        self.TestSet = self.TestSet.with_columns(pl.Series(name='betaML', values=betaML))
        print(self.TestSet.columns)
        self.TestSet = self.TestSet.with_columns(deltaBeta=((pl.col('betaML') - pl.col('beta')) / pl.col('beta')))

        # visualize regression results
        dv = DataVisual(self.TestSet, self.regressionDirectory, cfgGeneral['cfgVisualFile'])
        dv.createHistos()
        dv.create2DHistos()
        dvs = DataVisualSlow(self.TestSet, self.regressionDirectory)
        dvs.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
        dvs.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
        dvs.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)
        dvs.create2DHisto('pTPC', 'betaML', '#beta_{ML} vs #it{p}_{TPC}; #it{p}_{TPC} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

        del dv

        for particle in self.cfg['species']:
            dv = DataVisualSlow(self.TestSet.filter(pl.col('partID') == particlePDG[particle]), self.particleDirectories[particle])
            dv.createHisto('deltaBeta', '#Delta#beta; #Delta#beta; Counts', 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'deltaBeta', '#Delta#beta vs #it{p}; #it{p} (GeV/#it{c}); #Delta#beta', 1500, 0, 1.5, 1000, -0.5, 0.5)
            dv.create2DHisto('beta', 'deltaBeta', '#Delta#beta vs #beta; #beta; #Delta#beta', 1000, 0, 1, 1000, -0.5, 0.5)
            dv.create2DHisto('p', 'betaML', '#beta_{ML} vs #it{p}; #it{p} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)
            dv.create2DHisto('pTPC', 'betaML', '#beta_{ML} vs #it{p}_{TPC}; #it{p}_{TPC} (GeV/#it{c}); #beta_{ML}', 1500, 0, 1.5, 1200, 0, 1.2)

            del dv 

        graph = TGraph(len(self.losses), np.asarray(np.arange(self.cfg['NN']['nEpochs']), dtype=float), np.asarray(self.losses, dtype=float))
        graph.SetTitle(r'; Epoch ; Loss')
        graph.Write('loss')
        
        del directory, dv





def hyperparameterScoreCanvas(hyperparameterImportance):

    hyperparameterImportance = sorted(hyperparameterImportance.items(), key=lambda x: x[1], reverse=True)
    h = TH1D('h', '', len(hyperparameterImportance), 0, len(hyperparameterImportance))
    for ientry, (hyperparameter, score) in enumerate(hyperparameterImportance.items()):
        h.SetBinContent(ientry+1, score)
        h.SetBinLabel(ientry+1, hyperparameter)

    c = TCanvas('hyperparameterImportance', 'Hyperparamter importance; Score; Hyperparameter', 800, 600)
    h.Draw('bar hist')
    
    return c

        