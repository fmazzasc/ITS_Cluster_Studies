#
#   Data visualization
#

import numpy as np
import polars as pl
import yaml

from ROOT import TDirectory, TH1D, TH2D

import sys
sys.path.append('..')
from utils.particles import particlePDG

class DataVisual:

    def __init__(self, dataset, outDirectory:TDirectory, cfgFile:str):
        
        self.dataset = dataset
        self.outDirectory = outDirectory
        with open(cfgFile, 'r') as f:   self.cfg = yaml.safe_load(f)

    def createHistos(self):

        cfgHistos = self.cfg['TH1']

        for col, settings in cfgHistos['settings'].items():
            
            h = TH1D(col, settings['title'], settings['nBins'], settings['xmin'], settings['xmax'])
            for x in self.dataset[col].to_list():       
                if x != np.nan: h.Fill(x)
            
            self.outDirectory.cd()
            h.Write()

            del h
    
    def create2DHistos(self):

        cfg2DHistos = self.cfg['TH2']

        for col, settings in cfg2DHistos['settings'].items():
                
            h = TH2D(f'{settings["y"]}_vs_{settings["x"]}', settings['title'], settings['nBinsX'], settings['xmin'], settings['xmax'], settings['nBinsY'], settings['ymin'], settings['ymax'])
            for x, y in zip(self.dataset[settings['x']].to_list(), self.dataset[settings['y']].to_list()):       
                if x != np.nan and y != np.nan: h.Fill(x, y)
            
            self.outDirectory.cd()
            h.Write()

            del h

    def visualWeights(self, cfgPID):

        if cfgPID['weights']['type'] == 'PandSpecies':  self.visualWeightsPandSpecies(cfgPID)

    def visualWeightsPandSpecies(self, cfgPID):
        
        cfg = cfgPID['weights']['weightsPandSpecies']
        h = TH2D('weightsPandSpecies', 'Weights; #it{p} (GeV/#it{c}); Species', cfg['nbins'], cfg['pmin'], cfg['pmax'], len(cfgPID['species']), 0.5, len(cfgPID['species'])+0.5)
        for i, species in enumerate(cfgPID['species']):
            h.GetYaxis().SetBinLabel(i+1, species)
            filtered = self.dataset.filter(pl.col('partID') == particlePDG[species])
            for x, w in zip(filtered['p'].to_list(), filtered['weightsPS'].to_list()):
                if x != np.nan and w != np.nan: h.Fill(x, i+1, w)

        self.outDirectory.cd()
        h.Write()

        del h


###


class DataVisualSlow:

    def __init__(self, dataset, outDirectory:TDirectory):
        
        self.dataset = dataset
        self.outDirectory = outDirectory

    def createHisto(self, col, title, nBins, xmin, xmax):
        
        h = TH1D(col, title, nBins, xmin, xmax)
        for x in self.dataset[col].to_list():       
            if x != np.nan: h.Fill(x)
        
        self.outDirectory.cd()
        h.Write()

        del h
    
    def create2DHisto(self, x, y, title, nBinsX, xmin, xmax, nBinsY, ymin, ymax):
                
        h = TH2D(f'{y}_vs_{x}', title, nBinsX, xmin, xmax, nBinsY, ymin, ymax)
        for x, y in zip(self.dataset[x].to_list(), self.dataset[y].to_list()):       
            if x != np.nan and y != np.nan: h.Fill(x, y)
        
        self.outDirectory.cd()
        h.Write()

        del h