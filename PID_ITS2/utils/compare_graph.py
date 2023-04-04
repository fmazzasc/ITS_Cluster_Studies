'''
Script for the comparison of ROOT TH1s or TGraphs
'''

import os
import sys
from os.path import join
import argparse
import numpy as np
import yaml
from ROOT import TCanvas, TFile, TLegend, TLine, gStyle, TList, TPaveText, kRed, kOrange, kAzure, kGreen, kFullCircle

sys.path.append('../..')
from utils.StyleFormatter_old import SetGlobalStyle, SetObjectStyle, GetROOTColor, GetROOTMarker #pylint: disable=wrong-import-position,import-error
from utils.AnalysisUtils import ComputeRatioDiffBins, ScaleGraph, ComputeRatioGraph #pylint: disable=wrong-import-position,import-error

#################################################################################################################
#
#   General functions
#________________________________________________________________________________________________________________

def StatLegendPosition(x, y, w, h):
    gStyle.SetStatX(x)
    gStyle.SetStatY(y)
    gStyle.SetStatW(w)
    gStyle.SetStatH(h)

def min_size(*args):
    if not args:    return 0
    list = [len(arg) for arg in args]
    return min(list)

#################################################################################################################
#
#   Compare Graph
#________________________________________________________________________________________________________________

class CompareGraph_config:
    """
    Class to load all necessary information to do graph comparison

    Attributes
    ----------
        mode (str): Type of analysis to run (preprocessing steps depend on the dataset provided)
        skip_data_prep, skip_training, skip_appl (bool): Whether this steps should be skipped in the run. If 
            some of them are skipped, previous results are loaded to continue the analysis
        
        fimpPath (str): File with input data for training set
        ext_appl (bool): Whether an external dataset for application is provided or the training set should be 
            split.
        applPath (str): File with input data for application
        particle_dict (dict): Dictionary (index, particle_species)
        
        seven_hits (bool): Whether particles not detected by all seven silicon layers should be discarded
        test_size (float): Fraction of training set that will be used for test
        random_state (int): Seed initialization (garantees reproducibility)

        do_augm, beta_flat, beta_p_flat, MC_weights, do_equal (bool): Whether one of this weighting methods should
            be used in the ml regression
        
        do_plots (bool): Whether plots of preprocessed data should be produced
        Prep_output_dir, ML_output_dir, Application_output_dir, delta_output_dir (str): File to save plots to 
            (for data preparation, ML, application and score results)

        RegressionColumns (list[str]): Name of the columns of the dataset that should be considered for the regression
        model_choice (str): ML model to use for the regression
        ModelParams (dict): ML model hyperparams initialization values

        do_opt (bool): Whether an optuna hyperparameter optimization should be run
        HyperParamsRange (dict): Range in which the optuna optimization should be run
        early_stop (bool): Whether the optimization should automatically stop after given time
        save_model (bool): Whether the trained ML model should be saved to a file
    """
    def __init__(self, inputCfgFile):
        """
        Parameters
        ----------
            inputCfgFile (str): yaml file to load
        """
        with open(inputCfgFile) as f:   config = yaml.load(f, Loader=yaml.FullLoader)
    
        self.inDirName = config['inputs']['dirname']
        self.inFileNames = config['inputs']['filenames']
        self.objNames = config['inputs']['objectnames']
        
        self.outputFileName = config['output']['filename']
        self.outExtensions = config['output']['extensions']
        self.outFileNames = config['output']['objectnames']
        
        self.objTypes = config['options']['ROOTobject']
        self.scales = config['options']['scale']
        self.normalizes = config['options']['normalize']
        self.colors = config['options']['colors']
        self.markers = config['options']['markers']
        self.markersize = config['options']['markersize']
        self.linewidth = config['options']['linewidth']
        self.fillstyles = config['options']['fillstyle']
        self.fillalphas = config['options']['fillalpha']
        self.drawOptions = config['options']['drawopt']
        self.rebins = config['options']['rebin']
        
        self.doRatio = config['options']['ratio']['enable']
        self.drawRatioUnc = config['options']['ratio']['uncertainties']['enable']
        self.ratioUncCorr = config['options']['ratio']['uncertainties']['corr']
        self.displayRMS = config['options']['ratio']['displayRMS']
        
        self.doCompareUnc = config['options']['errcomp']['enable']
        self.compareRelUnc = config['options']['errcomp']['relative']
        self.KS = config['options']['KS']
        
        self.wCanv = config['options']['canvas']['width']
        self.hCanv = config['options']['canvas']['heigth']
        self.xLimits = config['options']['canvas']['xlimits']
        self.yLimits = config['options']['canvas']['ylimits']
        self.yLimitsRatio = config['options']['canvas']['ylimitsratio']
        self.yLimitsUnc = config['options']['canvas']['ylimitserr']
        self.xTitle = config['options']['canvas']['xaxistitle']
        self.yTitle = config['options']['canvas']['yaxistitle']
        self.logX = inputCfg['options']['canvas']['logx']
        self.logY = config['options']['canvas']['logy']
        self.logZ = config['options']['canvas']['logz']
        self.ratioLogX = config['options']['canvas']['ratio']['logx']
        self.ratioLogY = config['options']['canvas']['ratio']['logy']
        self.uncCompLogX = config['options']['canvas']['errcomp']['logx']
        self.uncCompLogY = config['options']['canvas']['errcomp']['logy']
        
        self.avoidStatBox = config['options']['statbox']['avoid']
        self.xStatLimits = config['options']['statbox']['xlimits']
        self.yStatLimits = config['options']['statbox']['ylimits']
        self.statheader = config['options']['statbox']['header']
        self.statTextSize = config['options']['statbox']['textsize']
        
        self.avoidLeg = config['options']['legend']['avoid']
        self.xLegLimits = config['options']['legend']['xlimits']
        self.yLegLimits = config['options']['legend']['ylimits']
        self.legNames = config['options']['legend']['titles']
        self.legOpt = config['options']['legend']['options']
        self.header = config['options']['legend']['header']
        self.legTextSize = config['options']['legend']['textsize']
        self.ncolumns = config['options']['legend']['ncolumns']
        
        self.avoidStatLegend = config['options']['statlegend']['avoid']
        self.xStatLegend = config['options']['statlegend']['x']
        self.yStatLegend = config['options']['statlegend']['y']
        self.wStatLegend = config['options']['statlegend']['w']
        self.hStatLegend = config['options']['statlegend']['h']

def compare_graph(inputCfgFile):
    """
    WIP
    
    """

    opt = CompareGraph_config(inputCfgFile)

    # auto-disable ratio and compare features if a single object is created
    single = (min_size(opt.inFileNames, opt.outFileNames, opt.objNames, opt.objTypes, opt.scales, opt.normalizes, opt.colors, opt.markers, opt.fillstyles, opt.fillalphas, opt.rebins) == 1)
    if single:  opt.doRatio, opt.doCompareUnc = False, False       

    # set global style
    SetGlobalStyle(padleftmargin=0.18, padbottommargin=0.14, titleoffsety=1.5, optstat=1111)
    if avoidStatLegend:     gStyle.SetOptStat(0)

#################################################################################################################
#
#   Multi Hist Canvas  
#________________________________________________________________________________________________________________

class MultiHistCanvas_config:

    def __init__(self, inputCfgFile):

        with open(inputCfgFile) as f:   config = yaml.load(f, Loader=yaml.FullLoader)

        self.wCanv = config['options']['canvas']['width']
        self.hCanv = config['options']['canvas']['heigth']
        self.xLimits = config['options']['canvas']['xlimits']
        self.yLimits = config['options']['canvas']['ylimits']
        self.xTitle = config['options']['canvas']['xaxistitle']
        self.yTitle = config['options']['canvas']['yaxistitle']
        self.logX = config['options']['canvas']['logx']
        self.logY = config['options']['canvas']['logy']

        self.avoidStatBox = config['options']['statbox']['avoid']
        self.xStatLimits = config['options']['statbox']['xlimits']
        self.yStatLimits = config['options']['statbox']['ylimits']
        self.statheader = config['options']['statbox']['header']
        self.statTextSize = config['options']['statbox']['textsize']
        
        self.avoidLeg = config['options']['legend']['avoid']
        self.xLegLimits = config['options']['legend']['xlimits']
        self.yLegLimits = config['options']['legend']['ylimits']
        self.legNames = config['options']['legend']['titles']
        self.legOpt = config['options']['legend']['options']
        self.header = config['options']['legend']['header']
        self.legTextSize = config['options']['legend']['textsize']
        self.ncolumns = config['options']['legend']['ncolumns']
        
        self.avoidStatLegend = config['options']['statlegend']['avoid']
        self.xStatLegend = config['options']['statlegend']['x']
        self.yStatLegend = config['options']['statlegend']['y']
        self.wStatLegend = config['options']['statlegend']['w']
        self.hStatLegend = config['options']['statlegend']['h']

        # histogram options
        self.normalize = config['options']['normalize']
        self.markersize = config['options']['markersize'] 
        self.linewidth = config['options']['linewidth']
        self.fillstyle = config['options']['fillstyle']
        self.fillalpha = config['options']['fillalpha']
        self.drawOpt = config['options']['drawopt']

    def color_(self, index):
        """
        Return TColor based on the index
        """

        colors = [kRed+1, kAzure+4, kGreen+3, kOrange+2]
        list_lenght = len(colors)
        return colors[index%list_lenght]
    
    def marker_(self, index):
        """
        Return TMarker based on the index
        """

        markers = [kFullCircle]
        list_lenght = len(markers)
        return markers[index%list_lenght]

def multiHistCanvas(inputCfgFile, outFile, canvasName='outCanvas', update=False, *args):
    """
    Function to print different histograms on a same canvas and then save it on file

    Parameters
    ----------
        inputCfgFile (str): YAML configuration file
        outFile (TFile): file to save the canvas to
        update (bool, optional): whether the new TCanvas should be appended to an existing file or a new one should be created
        *args (TH1): histograms to draw on canvas
    """

    opt = MultiHistCanvas_config(inputCfgFile)

    # set global options
    SetGlobalStyle(padleftmargin=0.18, padbottommargin=0.14, titleoffsety=1.5, optstat=1111)
    if opt.avoidStatLegend:     gStyle.SetOptStat(0)
    pave = TPaveText(opt.xStatLimits[0], opt.yStatLimits[0], opt.xStatLimits[1], opt.yStatLimits[1])
    pave.SetFillStyle(0)
    pave.SetTextSize(opt.statTextSize)

    # create and customize legend
    leg = TLegend(opt.xLegLimits[0], opt.yLegLimits[0], opt.xLegLimits[1], opt.yLegLimits[1])
    leg.SetFillStyle(0)
    leg.SetTextSize(opt.legTextSize)
    leg.SetNColumns(opt.ncolumns)
    leg.SetHeader(opt.header)

    # customize histograms
    for i, hist in enumerate(args):
        SetObjectStyle(hist, color=opt.color_(i), markerstyle=opt.marker_(i),
                       markersize=opt.markersize, linewidth=opt.linewidth, fillstyle=opt.fillstyle, fillalpha=opt.fillalpha)
        
        hist.SetDirectory(0)
        if opt.normalize:       hist.Scale(1. / hist.Integral())
          
        leg.AddEntry(hist, hist.GetTitle(), 'l')

    # create and customize canvas
    outCanvas = TCanvas(canvasName, '', opt.wCanv, opt.hCanv)
    hFrame = outCanvas.cd().DrawFrame(opt.xLimits[0], opt.yLimits[0], opt.xLimits[1], opt.yLimits[1], f';{opt.xTitle};{opt.yTitle}')
    if opt.logX:    outCanvas.cd().SetLogx()
    if opt.logY:    outCanvas.cd().SetLogy()
    
    hFrame.GetYaxis().SetDecimals() 

    # save canvas
    for hist in args:       hist.Draw(opt.drawOpt)
    if  not opt.avoidLeg:   leg.Draw()
    outCanvas.Write()
    #outCanvas.SaveAs('../output/rofComparison/check.png')
    
    

#################################################################################################################
#
#   Run compare_graph from terminal
#________________________________________________________________________________________________________________

if __name__ == "__main__":
    pass