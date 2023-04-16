#
#   Classes and function to plot
#

import pandas as pd
import numpy as np

import uproot
from ROOT import TH1D, TH2D, TGraphErrors, gStyle, kBird

class Plotter:
    """
    Attributes
    ---
    - df: pd.DataFrame with data to plot
    - xVarsToPlot: names of the column to plot on the x axis
    - yVarsToPlot: names of the column to plot on the y axis
    - plotSpecifics: as required by intern functions
    - dfNames: list of names for passed dataframes
    - foutPath: path to the output file ('/path-to-file/other_name.root')
    """

    def __init__(self, df, tfile, dfNames=None):
        self.df = df
        self.dfNames = dfNames

        self.dfs = [self.df]
        self.tfile = tfile
        


    def HistPlots(self, var, plot_specifics, weights=None):
        """
        Plot the same variable for different dfs

        Parameters
        ---
        - var: name of the variable to plot
        - plot_specific: list with the following content -> [x_label, nbinsx, xlow, xup]
        - weights: weights column name (if not specified, no weights will be applied)
        """

        x_label, nbinsx, xlow, xup = plot_specifics

        for i, df in enumerate(self.dfs):
            if len(self.dfs) == 1:          hist_name = var
            elif self.dfNames is not None:  hist_name = f'{var}_{self.dfNames[i]}'
            else:                           hist_name = f'{var}_{i}'

            if weights is not None:         hist_name += '_weights'

            hist = TH1D(hist_name, hist_name, nbinsx, xlow, xup)
            
            if weights is not None:   
                for x, w in zip(df[var], df[weights]):      hist.Fill(x, w)
            else:
                for x in df[var]:                           hist.Fill(x)

            hist.GetXaxis().SetTitle(x_label)
            hist.GetYaxis().SetTitle('Counts')

            self.tfile.cd()
            hist.SetDrawOption('hist')
            hist.Write()

    def ScatterPlot(self, x, y, plot_specifics, weights=None):
        """
    
        Parameters
        ---
        - x: x column name
        - y: y column name
        - plot_specific: list with the following content -> [x_label, y_label, nbinsx, xlow, xup, nbinsy, ylow, yup]
        - weights: weights column name (if not specified, no weights will be applied)
        """
        
        [x_label, y_label, nbinsx, xlow, xup, nbinsy, ylow, yup] = plot_specifics

        for i, df in enumerate(self.dfs):
            if len(self.dfs) == 1:          plot_name = f'{y}_vs_{x}'
            elif self.dfNames is not None:  plot_name = f'{y}_vs_{x}_{self.dfNames[i]}'
            else:                           plot_name = f'{y}_vs_{x}_{i}'

            if weights is not None:         plot_name += '_weights'

            scatter_plot = TH2D(plot_name, plot_name, nbinsx, xlow, xup, nbinsy, ylow, yup)

            if weights is not None:
                for (xi, yi, wi) in zip(df[x], df[y], df[weights]):  scatter_plot.Fill(xi, yi, wi)
            else:
                for (xi, yi) in zip(df[x], df[y]):  scatter_plot.Fill(xi, yi)

            scatter_plot.GetXaxis().SetTitle(x_label)
            scatter_plot.GetYaxis().SetTitle(y_label)

            gStyle.SetPalette(kBird)
            
            self.tfile.cd()
            scatter_plot.SetDrawOption('COLZ1')
            scatter_plot.Write()
          
    def plot1D(self, xVarsToPlot, plotSpecifics, weights=None):
        """
        Create and save TH1 for all sub df and all variables
        
        - plotSpecifics: list of specifics -> [[xLabel, nbins, xlow, xup], ...] 
        """
        for var, plt_spc in zip(xVarsToPlot, plotSpecifics):   self.HistPlots(var, plt_spc, weights)   

    def plot2D(self, xVarsToPlot, yVarsToPlot, plotSpecifics, weights=None):
        """
        Create and save TH2 for all sub df and all variables
        
        - plotSpecifics: list of specifics -> [[x_label, y_label, nbinsx, xlow, xup, nbinsy, ylow, yup], ...] 
        """
        for x, y, plt_spc in zip(xVarsToPlot, yVarsToPlot, plotSpecifics):  self.ScatterPlot(x, y, plt_spc, weights)   

    def multi_df(self, label_col, label_list):
        """
        Create sub dataframes based on values of a column
        """
        self.dfs = [self.df.query(f'{label_col} == "{label}"', inplace=False) for label in label_list]
        self.dfNames = [label for label in label_list]

class TH2Handler:
    """
    Class to generate a TGraphErrors with data extracted from a TH2
    """

    def __init__(self, df: pd.DataFrame, tfile, x_name='', y_name='') -> None:
        self.df = df
        self.x_name = x_name
        self.y_name = y_name
        self.th2 = TH2D()
        self.tfile = tfile

    def import_th2(self, fimpPath):
        self.th2 = uproot.open(fimpPath)

    def build_th2(self, xbins, xlow, xup, ybins, ylow, yup):
        self.th2 = TH2D('hist', 'hist', xbins, xlow, xup, ybins, ylow, yup)
        for x, y in zip(self.df[self.x_name], self.df[self.y_name]):    self.th2.Fill(x, y)

    def TH2toLine(self, TGEname, axis, bins_per_interval):
        x = [] 
        y = [] 
        sx = []
        sy = []
    
        if axis =='x': 
            for bin in range(1, self.th2.GetNbinsY()+1, bins_per_interval):
                th1 = self.th2.ProjectionX('_px', bin, bin+bins_per_interval)
                x.append(self.th2.GetYaxis().GetBinLowEdge(bin) + self.th2.GetYaxis().GetBinWidth(bin)*0.5)
                sx.append(bins_per_interval * self.th2.GetYaxis().GetBinWidth(bin))
                y.append(th1.GetMean())
                sy.append(th1.GetStdDev())

        if axis == 'y':
            for bin in range(1, self.th2.GetNbinsX()+1, bins_per_interval):
                th1 = self.th2.ProjectionY('_py', bin, bin+bins_per_interval)
                x.append(self.th2.GetXaxis().GetBinLowEdge(bin) + self.th2.GetXaxis().GetBinWidth(bin)*0.5)
                sx.append(bins_per_interval * self.th2.GetXaxis().GetBinWidth(bin))
                y.append(th1.GetMean())
                sy.append(th1.GetStdDev())

        tge = TGraphErrors(len(x), np.array(x), np.array(y), np.array(sx), np.array(sy))
        self.tfile.cd()
        tge.SetName(TGEname)
        tge.Write()
