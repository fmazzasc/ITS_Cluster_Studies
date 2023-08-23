'''
    Script to select candidates using machine learning predictions
'''

import sys
import numpy as np
import pandas as pd
import yaml

import uproot
from ROOT import TH1D, TFile, TF1, TCanvas, gStyle
from ROOT import kOrange, kGreen, kAzure, kRed

from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter

gStyle.SetOptFit(0)
gStyle.SetOptStat(0)

class MassFitter:
    '''
    Class to fit invariant mass distribution with input function. Defining a threshold (as a number of sigma of the distribution) you can study the
    purity/contamination of the selection.

    Attributes 
    ----------
        - df (pandas.DataFrame): starting dataframe (useful to calculate efficiency)
        - infilePath (str): path to the ROOT file where the distribution histogram will be extracted from
        - distributionName (str): name of the distribution to extract from the ROOT file

        - fitFunction (str): function to fit the distribution with (no bkg)
            The list of names for the signal pdfs. The possible options are: 'gaussian', 'doublegaus', 'crystalball', 'doublecb', 'cauchy', 'voigtian'
        - bkgFunction (str): function to fit the background 
            The list of names of the background pdfs. The possible options are: 'nobkg', 'expo', 'powlaw', 'expopow', 'chebpolN' (N is the order of the polynomial)

        - fitMass (float): mass estimation from fit (mean of the distribution). PARAMETER 1 OF THE FIT FUNCTION
        - fitSigma (float): sigma from the fit. PARAMETER 2 OF THE FIT FUNCTION

        - fitChi, fitNDF, fitPValue (floats): results of the fit
    '''

    def __init__(self, df, infilePath, distributionName, fitFunction, bkgFunction=None):
        
        self.df = df
        
        self.infilePath = infilePath
        self.distributionName = distributionName
        self.data_handler = None

        self.fitFunction = fitFunction
        self.bkgFunction = bkgFunction

        self.fitMass = None
        self.fitSigma = None
        
        self.fitChi = None
        self.fitNDF = None

    def fitPeaks(self, fitRange, outputPath, parSigSet=None, parBkgSet=None):
        '''
        Executes fit for the distribution (always fit on the entire mass invariant distribution, not selected by species)

        Parameters
        ----------
            - fitRange ([float]): [xmin. xmax] -> range to fit within
            - outputPath (str): path to an output ROOT file (the file should be already existing)
            - parSigSet ([[str, float, [float, float]]]): list of [par_name, value to set, [min, max]], where min and max are the limits to fit within (related to signal function)
            - parBkgSet ([[str, float, [float, float]]]): list of [par_name, value to set, [min, max]], where min and max are the limits to fit within (related to background function)
        
        Returns
        -------
            - fitter (F2MassFitter): function used to fit

        '''

        self.data_handler = DataHandler(data=self.infilePath, var_name='mass (ML) (GeV/#it{c}^2)', limits=fitRange, histoname=self.distributionName)
        fitter = F2MassFitter(self.data_handler, [self.fitFunction], [self.bkgFunction])
        
        if parSigSet is not None:
            for name, value, limits in parSigSet: fitter.set_signal_initpar(0, name, value, limits=limits)
        if parBkgSet is not None:
            for name, value, limits in parBkgSet: fitter.set_background_initpar(0, name, value, limits=limits)
        fitResults = fitter.mass_zfit()

        fitter.dump_to_root(outputPath, option='recreate')

        self.fitMass, __ = fitter.get_mass()
        self.fitSigma, __ = fitter.get_sigma()
        self.fitChi = fitter.get_chi2()
        self.fitNDF = fitter.get_ndf()

        print('Fit results: chi/NDF = ', self.fitChi, ' / ', self.fitNDF)

        return fitter

    def fitResolution(self):
        '''
        Returns the fit resolution as sigma/mean. Defines the resolution of the method
        '''

        return self.fitSigma / self.fitMass

    #def selectPeakCandidates(self, threshold):
    #    '''
    #    Returns the number of candidates in the fitted peak with a separation of (threshold) sigma from the estimated mass (mean)
#
    #    Parameters 
    #    ----------
    #        - threshold (float): number of sigma a candidate can be separated from the mean value
    #    '''
#
    #    n_candidates = 0
    #    selectedRange = [self.fitMass - threshold*self.fitSigma, self.fitMass + threshold*self.fitSigma]
    #    selectedBinRange = [self.distribution.GetBin(selectedRange[0]), self.distribution.GetBin(selectedRange[1])]
#
    #    for i in np.arange(selectedBinRange[0], selectedBinRange[1]+1):  n_candidates += self.distribution.GetBinContent(i)
    #    return n_candidates

    def estimateEfficiency(self, threshold, species):
        '''
        Returns the efficiency of the selection, defined as the ratio between the number of selected candidates of a species with a fit and the 
        number of total candidates for that species.

        Parameters 
        ----------
            - threshold (float): number of sigma a candidate can be separated from the mean value
            - species (str): species to evaluate the efficiency of
        '''

        selectedRange = [self.fitMass - threshold*self.fitSigma, self.fitMass + threshold*self.fitSigma]
        nIdentified = df.query(f'label == "{species}" and {selectedRange[0]} < mass_pred < {selectedRange[1]}', inplace=False)
        nDetected = df.query(f'label == "{species}"', inplace=False)

        return len(nIdentified)/len(nDetected)

    def estimatePurity(self, threshold, species):
        '''
        Returns the purity of the selection, defined as the ratio between the number of selected candidates of a species with a fit and the 
        number of total candidates selected by that fit.

        Parameters 
        ----------
            - threshold (float): number of sigma a candidate can be separated from the mean value
            - species (str): species to evaluate the efficiency of
        '''

        selectedRange = [self.fitMass - threshold*self.fitSigma, self.fitMass + threshold*self.fitSigma]
        nIdentified = df.query(f'label == "{species}" and {selectedRange[0]} < mass_pred < {selectedRange[1]}', inplace=False)
        nSelected = df.query(f'{selectedRange[0]} < mass_pred < {selectedRange[1]}', inplace=False)

        return len(nIdentified)/len(nSelected)

class BetaFitter:
    '''
    Class to fit invariant the beta distribution with input function. Defining a threshold (as a number of sigma of the distribution) you can study the
    purity/contamination of the selection. The selection will be depending on the momentum of the candidate and is done by fitting the beta distribution in limited 
    range of momentum. It will be possible to define the distribution with weights to eliminate the differences in how the species populate the dataset.

    Attributes 
    ----------
        - _df (pandas.DataFrame): starting dataframe (useful to calculate efficiency) 
        - _config (yaml.load): configuration file used to set all values 
        - _outFile (TFile): file used to write the output to
        


        - pUpLimit (float): minimum value of momentum (in GeV/c) to consider for the identification
        - pUpLimit (float): maximum value of momentum (in GeV/c) to consider for the identification
        - pGranularity (float): width of the ranges of momentum in which each fit should be performed



        - fitFunction (str): function to fit the distribution with (no bkg)
            The list of names for the signal pdfs. The possible options are: 'gaussian', 'doublegaus', 'crystalball', 'doublecb', 'cauchy', 'voigtian'
        - bkgFunction (str): function to fit the background 
            The list of names of the background pdfs. The possible options are: 'nobkg', 'expo', 'powlaw', 'expopow', 'chebpolN' (N is the order of the polynomial)

        - fitMass (float): mass estimation from fit (mean of the distribution). PARAMETER 1 OF THE FIT FUNCTION
        - fitSigma (float): sigma from the fit. PARAMETER 2 OF THE FIT FUNCTION

        - fitChi, fitNDF, fitPValue (floats): results of the fit
    '''

    def __init__(self, df, config):
        
        self._df = df
        self._config = config
        self._outFile = TFile(self._config['outFilePath'], 'recreate')

        self._2Ddistribution = 0

        self._pLimits = np.linspace(start=self._config['pMin'], stop=self._config['pMax'], num=self._config['pNStep'])

        # fit results storage
        self._fitMeans = {key : None for key in (self._pLimits[0:-1]+(self._pLimits[1]-self._pLimits[0])/2)}
        self._fitSigmas = {key : None for key in (self._pLimits[0:-1]+(self._pLimits[1]-self._pLimits[0])/2)}
        self._fitResults = {key : None for key in (self._pLimits[0:-1]+(self._pLimits[1]-self._pLimits[0])/2)}

        print(self._fitMeans, '\n', self._fitSigmas)

    def fitBeta(self, idx, weights=False):
        '''
        Fit multiple peaks of the beta distribution in a given momentum range. Fundamental informations will be stored in data memebers.

        Parameters
        ----------
            - idx (int): index related to the momentum range to access informations in the configuration file
            - weights (bool): if true, the distribution is generated using 
        '''

        filtDf = df.query(f'{self._pLimits[idx]} <= p < {self._pLimits[idx+1]}')
        
        suffix = ''
        if weights: suffix += '_weights'
        hist = TH1D(f'beta'+suffix+f'_{idx}', f'{self._config["graphTitle"]} {idx}; #beta (ML); Counts (a.u.)', 1100, 0, 1.1)
        if not weights: 
            for beta in filtDf['beta']:    hist.Fill(beta)
        else:
            pass
            # should set weight and properly set the error bars as well
        
        functions = []
        fitMeans = []
        fitSigmas = []
        fitResults = []

        for i, sets in enumerate(self._config['fitParInit'][idx]):

            sigFunc = TF1(f'sigFunc_{idx}_{i}', self._config['sigFunc'], sets[2], sets[3])
            bkgFunc = TF1(f'bkgFunc_{idx}_{i}', self._config['bkgFunc'], sets[2], sets[3])
            fitFunc = TF1(f'fitFunc_{idx}_{i}', self._config['sigFunc']+'+'+self._config['bkgFunc'], sets[2], sets[3])

            sigFunc.SetLineColor(kOrange-3)
            bkgFunc.SetLineColor(kAzure)
            fitFunc.SetLineColor(kGreen-3)
            hist.SetFillColorAlpha(kRed-3, 0.3)

            if sets[0] is not None: fitFunc.SetParameter(1, sets[0])
            if sets[1] is not None: fitFunc.SetParameter(2, sets[1])
            hist.Fit(fitFunc, 'rm')

            sigFunc.SetParameters(fitFunc.GetParameter(0), fitFunc.GetParameter(1), fitFunc.GetParameter(2))
            bkgFunc.SetParameters(fitFunc.GetParameter(3), fitFunc.GetParameter(4))
            functions.extend([sigFunc, bkgFunc, fitFunc])

            fitMeans.append(fitFunc.GetParameter(1))
            fitSigmas.append(fitFunc.GetParameter(2))
            fitResults.append(fitFunc.GetChisquare()/fitFunc.GetNDF())
        
        canvas = TCanvas(f'canvas_{idx}', '')
        hist.Draw('hist same')
        for function in functions:  function.Draw('same')

        self._outFile.cd()
        hist.Write()
        canvas.Write()
        
        self._fitMeans[list(self._fitMeans)[idx]] = fitMeans
        self._fitSigmas[list(self._fitSigmas)[idx]] = fitSigmas
        self._fitResults[list(self._fitResults)[idx]] = fitResults

    def fitBetaAllP(self, weights=False):
        '''
        Fit the beta distribution for all the momentum intervals specified in the configuration file. 

        Parameters
        ----------
            - weights (bool): if true, the histogram to fit will be produced using weights to flatten the distribution in momentum and species
        '''

        for idx in range(self._config['pNStep']-1):   self.fitBeta(idx=idx, weights=weights)




#    def fitPeaks(self, fitRange, outputPath, parSigSet=None, parBkgSet=None):
#        '''
#        Executes fit for the distribution (always fit on the entire mass invariant distribution, not selected by species)
#
#        Parameters
#        ----------
#            - fitRange ([float]): [xmin. xmax] -> range to fit within
#            - outputPath (str): path to an output ROOT file (the file should be already existing)
#            - parSigSet ([[str, float, [float, float]]]): list of [par_name, value to set, [min, max]], where min and max are the limits to fit within (related to signal function)
#            - parBkgSet ([[str, float, [float, float]]]): list of [par_name, value to set, [min, max]], where min and max are the limits to fit within (related to background function)
#        
#        Returns
#        -------
#            - fitter (F2MassFitter): function used to fit
#
#        '''
#
#        self.data_handler = DataHandler(data=self.infilePath, var_name='mass (ML) (GeV/#it{c}^2)', limits=fitRange, histoname=self.distributionName)
#        fitter = F2MassFitter(self.data_handler, [self.fitFunction], [self.bkgFunction])
#        
#        if parSigSet is not None:
#            for name, value, limits in parSigSet: fitter.set_signal_initpar(0, name, value, limits=limits)
#        if parBkgSet is not None:
#            for name, value, limits in parBkgSet: fitter.set_background_initpar(0, name, value, limits=limits)
#        fitResults = fitter.mass_zfit()
#
#        fitter.dump_to_root(outputPath, option='recreate')
#
#        self.fitMass, __ = fitter.get_mass()
#        self.fitSigma, __ = fitter.get_sigma()
#        self.fitChi = fitter.get_chi2()
#        self.fitNDF = fitter.get_ndf()
#
#        print('Fit results: chi/NDF = ', self.fitChi, ' / ', self.fitNDF)
#
#        return fitter
#
#    def fitResolution(self):
#        '''
#        Returns the fit resolution as sigma/mean. Defines the resolution of the method
#        '''
#
#        return self.fitSigma / self.fitMass
#
#    def selectPeakCandidates(self, threshold):
#    #    '''
#    #    Returns the number of candidates in the fitted peak with a separation of (threshold) sigma from the estimated mass (mean)
##
#    #    Parameters 
#    #    ----------
#    #        - threshold (float): number of sigma a candidate can be separated from the mean value
#    #    '''
##
#        n_candidates = 0
#    #    selectedRange = [self.fitMass - threshold*self.fitSigma, self.fitMass + threshold*self.fitSigma]
#    #    selectedBinRange = [self.distribution.GetBin(selectedRange[0]), self.distribution.GetBin(selectedRange[1])]
##
#    #    for i in np.arange(selectedBinRange[0], selectedBinRange[1]+1):  n_candidates += self.distribution.GetBinContent(i)
#        return n_candidates
#
#    def estimateEfficiency(self, threshold, species):
#        '''
#        Returns the efficiency of the selection, defined as the ratio between the number of selected candidates of a species with a fit and the 
#        number of total candidates for that species.
#
#        Parameters 
#        ----------
#            - threshold (float): number of sigma a candidate can be separated from the mean value
#            - species (str): species to evaluate the efficiency of
#        '''
#
#        selectedRange = [self.fitMass - threshold*self.fitSigma, self.fitMass + threshold*self.fitSigma]
#        nIdentified = df.query(f'label == "{species}" and {selectedRange[0]} < mass_pred < {selectedRange[1]}', inplace=False)
#        nDetected = df.query(f'label == "{species}"', inplace=False)
#
#        return len(nIdentified)/len(nDetected)
#
#    def estimatePurity(self, threshold, species):
#        '''
#        Returns the purity of the selection, defined as the ratio between the number of selected candidates of a species with a fit and the 
#        number of total candidates selected by that fit.
#
#        Parameters 
#        ----------
#            - threshold (float): number of sigma a candidate can be separated from the mean value
#            - species (str): species to evaluate the efficiency of
#        '''
#
#        selectedRange = [self.fitMass - threshold*self.fitSigma, self.fitMass + threshold*self.fitSigma]
#        nIdentified = df.query(f'label == "{species}" and {selectedRange[0]} < mass_pred < {selectedRange[1]}', inplace=False)
#        nSelected = df.query(f'{selectedRange[0]} < mass_pred < {selectedRange[1]}', inplace=False)
#
#        return len(nIdentified)/len(nSelected)





if __name__ == '__main__':

    #outputFile = TFile('../output/analysis/invariantMassDistribution.root', 'recreate')
    #outputFile.Close()

    with open('configs/config_candidateSelection.yml') as f:   config = yaml.load(f, Loader=yaml.FullLoader)
    df = pd.read_parquet(config['inFilePath'])

    bf = BetaFitter(df=df, config=config)
    bf.fitBetaAllP(weights=False)
    
    #invariantMassFile = TFile('../output/TPC/application_beta_pflat.root')
    #invariantMassDistribution = invariantMassFile.Get('mass_pred')

    #fitRange = [0.05, 0.28]
    
    #peakPi = PeakFitter(df, invariantMassDistribution, 'cauchy', 'expo')
    #peakPi = MassFitter(df=df, infilePath='../output/TPC/application_beta_pflat.root', distributionName='mass_pred', fitFunction='cauchy', bkgFunction='nobkg')
    #fittingFunction = peakPi.fitPeaks(fitRange=fitRange, outputPath='../output/analysis/invariantMassDistribution.root', parSigSet=[ ['m', 0.140, [0.135, 0.155]], ['frac', 7e5, [6e5, 8e5]], ['gamma', 0.040, [0.001, 0.08]] ])

    #efficiency = peakPi.estimateEfficiency(3, 'Pi')
    #purity = peakPi.estimatePurity(3, 'Pi')

    #print('Efficiency for pions is eff = ', efficiency, '\nPurity for pions is purity = ', purity)
    
    #canvas = TCanvas('canvas', 'canvas', 900, 900)
    
    #invariantMassDistribution.Draw('hist same')

    #outputFile.cd()
    #canvas.Write()
    #canvas.SaveAs('../report/img5/invariantMassPion.png')
    #outputFile.Close()
    