import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from UsefulFunctions import filtering, augmentation_fine

import uproot
from ROOT import TH1D, TH2D, TGraphErrors, gStyle, kBird



## DATA PREPARATION CLASSES 

class PrepTool:
    def __init__(self) -> None:
        self.ApplicationDf = pd.DataFrame()
        self.extAppl = True

        self.TrainSet = pd.DataFrame()
        self.TestSet = pd.DataFrame()

    def returnDfs(self):
        return self.TrainSet, self.TestSet, self.ApplicationDf

    def Preprocess(self, particle_dict, mass_dict, tag_dict=None, test_size=0.2, random_state=0):
        pass

    def Preprocess2(self, seven_hits=False):
        """
        Same for all dataset used
        """

        # negative values are intended to be nans
        if not seven_hits:
            for i in range(7):
                self.TrainSet[f'ClSizeL{i}'] = np.where(self.TrainSet[f'ClSizeL{i}'] < 0, np.nan, self.TrainSet[f'ClSizeL{i}'])
                self.TestSet[f'ClSizeL{i}'] = np.where(self.TestSet[f'ClSizeL{i}'] < 0, np.nan, self.TestSet[f'ClSizeL{i}'])
                self.ApplicationDf[f'ClSizeL{i}'] = np.where(self.ApplicationDf[f'ClSizeL{i}'] < 0, np.nan, self.ApplicationDf[f'ClSizeL{i}'])

        # consider only candidates with seven hits
        else:   
            for i in range(7):
                self.TrainSet.drop( self.TrainSet[self.TrainSet[f'ClSizeL{i}'] < 0].index, inplace=True )
                self.TestSet.drop( self.TestSet[self.TestSet[f'ClSizeL{i}'] < 0].index, inplace=True )
                self.ApplicationDf.drop( self.ApplicationDf[self.ApplicationDf[f'ClSizeL{i}'] < 0].index, inplace=True )

        yTrain = pd.Series(self.TrainSet['beta'])
        yTest = pd.Series(self.TestSet['beta'])

        return self.TrainSet, yTrain, self.TestSet, yTest

class TPC_prep(PrepTool):
    """
    Manages datat preparation for PID on ITS-TPC data

    - fimpPath: path to the input file
    - particle_dict: dictionary {number: name} used to name particles
    """
    def __init__(self, fimpPath, applPath=None) -> None:
        
        self.RegressionDf = pd.read_parquet(fimpPath)
        
        if applPath is not None:    
            self.ApplicationDf = pd.read_parquet(applPath)
            self.extAppl = True
        else:   
            self.ApplicationDf = pd.DataFrame()
            self.extAppl = False

        self.TrainSet = pd.DataFrame()
        self.TestSet = pd.DataFrame()
    


    def Preprocess(self, particle_dict, mass_dict, tag_dict=None, test_size=0.2, random_state=0):

        self.RegressionDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
        self.RegressionDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
        self.RegressionDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)
        self.RegressionDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)  
        for part in particle_dict.values():  self.RegressionDf[f'nSigma{part}Abs'] = abs(self.RegressionDf[f'nSigma{part}'])

        self.RegressionDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)  
        self.RegressionDf.query('0.1 < p <= 0.7', inplace=True)

        if self.extAppl:
            self.ApplicationDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
            self.ApplicationDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
            self.ApplicationDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)  
            self.ApplicationDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)

            self.ApplicationDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)
            self.ApplicationDf.query('0.1 < p <= 0.7', inplace=True)

            for part in particle_dict.values():  self.ApplicationDf[f'nSigma{part}Abs'] = abs(self.ApplicationDf[f'nSigma{part}'])

        else:   self.RegressionDf, self.ApplicationDf = train_test_split(self.RegressionDf, test_size=0.5, random_state=self.random_state)  

        self.TrainSet, self.TestSet = train_test_split(self.RegressionDf, test_size=test_size, random_state=random_state)

        self.TrainSet = pd.concat([filtering(self.TrainSet, name, tag=tag_dict[name], mass=mass_dict[name]) for name in particle_dict.values()])
        self.TestSet = pd.concat([filtering(self.TestSet, name, tag=tag_dict[name], mass=mass_dict[name]) for name in particle_dict.values()])
        self.ApplicationDf = pd.concat([filtering(self.ApplicationDf, name, tag=tag_dict[name], mass=mass_dict[name]) for name in particle_dict.values()])

class hybrid_prep(PrepTool):
    """
    Manages datat preparation for PID on ITS-TPC data (train = hybrid set)

    - fimpPath: path to the input file
    - particle_dict: dictionary {number: name} used to name particles
    """
    def __init__(self, fimpPath, applPath=None) -> None:
        
        self.RegressionDf = pd.read_parquet(fimpPath)
        
        if applPath is not None:    
            self.ApplicationDf = pd.read_parquet(applPath)
            self.extAppl = True
        else:   
            self.ApplicationDf = pd.DataFrame()
            self.extAppl = False

        self.TrainSet = pd.DataFrame()
        self.TestSet = pd.DataFrame()
        
    def Preprocess(self, particle_dict, mass_dict, tag_dict=None, test_size=0.2, random_state=0):

        self.RegressionDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
        self.RegressionDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
        self.RegressionDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)
        self.RegressionDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)  
        for part in particle_dict.values():  self.RegressionDf[f'nSigma{part}Abs'] = abs(self.RegressionDf[f'nSigma{part}'])

        self.RegressionDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)  
        self.RegressionDf.eval('label = particle', inplace=True)
        for number, name in particle_dict.items():  self.RegressionDf['label'].mask(self.RegressionDf['particle'] == number, name, inplace=True)

        if self.extAppl:
            self.ApplicationDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
            self.ApplicationDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
            self.ApplicationDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)  
            self.ApplicationDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)

            self.ApplicationDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)
            self.ApplicationDf.eval('label = particle', inplace=True)
            for number, name in particle_dict.items():  self.ApplicationDf['label'].mask(self.ApplicationDf['particle'] == number, name, inplace=True)

            for part in particle_dict.values():  self.ApplicationDf[f'nSigma{part}Abs'] = abs(self.ApplicationDf[f'nSigma{part}'])

        else:   self.RegressionDf, self.ApplicationDf = train_test_split(self.RegressionDf, test_size=0.5, random_state=self.random_state)  


        self.TrainSet, self.TestSet = train_test_split(self.RegressionDf, test_size=test_size, random_state=random_state)

        self.TrainSet = pd.concat([filtering(self.TrainSet, name, mass=mass_dict[name], label=False) for name in particle_dict.values()])
        self.TestSet = pd.concat([filtering(self.TestSet, name, mass=mass_dict[name], label=False) for name in particle_dict.values()])
        self.ApplicationDf = pd.concat([filtering(self.ApplicationDf, name, mass=mass_dict[name], label=False) for name in particle_dict.values()])

class MC_prep(PrepTool):
    """
    Manages datat preparation for PID on MC sample (train = MC sample)

    - fimpPath: path to the input file
    - applPath: path to the application data
    """

    def __init__(self, fimpPath, applPath=None) -> None:
        
        self.RegressionDf = uproot.open(fimpPath)['ITStreeML'].arrays(library='pd')
        
        if applPath is not None:    
            self.ApplicationDf = pd.read_parquet(applPath)
            self.extAppl = True
        else:   
            self.ApplicationDf = pd.DataFrame()
            self.extAppl = False

        self.TrainSet = pd.DataFrame()
        self.TestSet = pd.DataFrame()

        # translates PDG labels to those used in this script
        self.MC_dict = {2212: 1, 321: 2, 211: 3, 11: 4} 
    


    def Preprocess(self, particle_dict, mass_dict, tag_dict=None, test_size=0.2, random_state=0):

        self.RegressionDf.eval('p = pMC', inplace=True)
        self.RegressionDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
        self.RegressionDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
        self.RegressionDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)

        temp = []
        for i, (key, value) in enumerate(self.MC_dict.items()):   
            temp.append(self.RegressionDf.query(f'abs(pdgMC) == {key}', inplace=False).reset_index(drop=True))
            temp[i]['particle'] = value
        self.RegressionDf = pd.concat(temp)

        self.RegressionDf.query('p <= 50', inplace=True)  
        self.RegressionDf.eval('label = particle', inplace=True)
        for number, name in particle_dict.items():  self.RegressionDf['label'].mask(self.RegressionDf['particle'] == number, name, inplace=True)

        if self.extAppl:
            self.ApplicationDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
            self.ApplicationDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
            self.ApplicationDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)  

            self.ApplicationDf.query('p <= 50', inplace=True)
            
        else:   self.RegressionDf, self.ApplicationDf = train_test_split(self.RegressionDf, test_size=0.5, random_state=random_state) 

        self.TrainSet, self.TestSet = train_test_split(self.RegressionDf, test_size=test_size, random_state=random_state) 

        self.TrainSet = pd.concat([filtering(self.TrainSet, name, mass=mass_dict[name], label=False) for name in particle_dict.values()])
        self.TestSet = pd.concat([filtering(self.TestSet, name, mass=mass_dict[name], label=False) for name in particle_dict.values()])
        self.ApplicationDf = pd.concat([filtering(self.ApplicationDf, name, mass=mass_dict[name], label=False) for name in particle_dict.values()])

class MCtpc_prep(PrepTool):
    """
    Manages datat preparation for PID on MC sample (train = MC sample)

    - fimpPath: path to the input file
    - applPath: path to the application data
    """

    def __init__(self, fimpPath, applPath=None) -> None:
        
        self.RegressionDf = uproot.open(fimpPath)['ITStreeML'].arrays(library='pd')
        
        if applPath is not None:    
            self.ApplicationDf = pd.read_parquet(applPath)
            self.extAppl = True
        else:   
            self.ApplicationDf = pd.DataFrame()
            self.extAppl = False

        # translates PDG labels to those used in this script
        self.MC_dict = {2212: 1, 321: 2, 211: 3, 11: 4} 
    


    def Preprocess(self, particle_dict, mass_dict, tag_dict=None, test_size=0.2, random_state=0):

        self.RegressionDf.eval('p = pMC', inplace=True)
        self.RegressionDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
        self.RegressionDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
        self.RegressionDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)

        temp = []
        for i, (key, value) in enumerate(self.MC_dict.items()):   
            temp.append(self.RegressionDf.query(f'abs(pdgMC) == {key}', inplace=False).reset_index(drop=True))
            temp[i]['particle'] = value
        self.RegressionDf = pd.concat(temp)

        self.RegressionDf.query('p <= 50', inplace=True)  
        self.RegressionDf.eval('label = particle', inplace=True)
        for number, name in particle_dict.items():  self.RegressionDf['label'].mask(self.RegressionDf['particle'] == number, name, inplace=True)

        if self.extAppl:
            self.ApplicationDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
            self.ApplicationDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
            self.ApplicationDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True) 
            self.ApplicationDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True) 

            self.ApplicationDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)
            self.ApplicationDf.eval('label = particle', inplace=True)
            for number, name in particle_dict.items():  self.ApplicationDf['label'].mask(self.ApplicationDf['particle'] == number, name, inplace=True)

        else:   self.RegressionDf, self.ApplicationDf = train_test_split(self.RegressionDf, test_size=0.5, random_state=self.random_state)  

        TrainSet, TestSet = train_test_split(self.RegressionDf, test_size=test_size, random_state=random_state)

        dfs_train = [filtering(TrainSet, name, mass=mass_dict[name], label=False) for name in particle_dict.values()]
        dfs_test = [filtering(TestSet, name, mass=mass_dict[name], label=False) for name in particle_dict.values()]
        self.ApplicationDf = pd.concat([filtering(self.ApplicationDf, name, mass=mass_dict[name], label=False) for name in particle_dict.values()])

        return dfs_train, dfs_test

class PrepConstructor:
    def __init__(self) -> None:
        pass

    def CreatePrepTool(self, type, fimpPath, applPath=None):
        if type == 'TPC':       return TPC_prep(fimpPath, applPath)
        if type == 'hybrid':    return hybrid_prep(fimpPath, applPath)
        if type == 'MC':        return MC_prep(fimpPath, applPath)
        if type == 'MCtpc':     return MCtpc_prep(fimpPath, applPath)

####################
class SpeciesHandler:
    """
    Attributes
    ----
    - part_dict: dictionary {number: "name"} that associates a number to the name of each particle (will be used to distinguish species in the dataframe)
    - mass_dict: dictionary {"name": mass} 
    - tag_dict: dictionary {"name": tag}, where tag is the selection tag for each species 


    Methods
    ---
    - df_list(): returns a list of dfs sorted by their species

    """

    def __init__(self, df: pd.DataFrame, part_dict: dict, mass_dict: dict=None, tag_dict: dict=None):
        self.df = df
        self.part_dict = part_dict
        self.mass_dict = mass_dict
        self.tag_dict = tag_dict

    def df_list(self):
        return [self.df.query(f'particle == {num}', inplace=False) for num in self.part_dict.keys()]

    def filtering(self, use_mass=True, use_tag=True):
        if use_mass and use_tag:    return [filtering(self.df, name, tag, mass) for name, tag, mass in zip(self.part_dict.values(), self.tag_dict.values(), self.mass_dict.values())]
        elif use_mass:              return [filtering(self.df, name, mass) for name, mass in zip(self.part_dict.values(), self.mass_dict.values())]
        elif use_tag:               return [filtering(self.df, name, tag) for name, tag in zip(self.part_dict.values(), self.tag_dict.values())]
        else:                       return [filtering(self.df, name) for name in self.part_dict.values()]
    
####################    
class DataAugmenter:
    """
    Attributes
    ---
    - df: pd.DataFrame storing original data
    - daughters: list of the names of the species you want to augment
    - mothers: list of the names of the species the augmentation will be done from
    - p_ranges: list of ranges (i.e. sublists [double, double])with momentum ranges used for each augmentation process
    - mass_dict: dictionary {'particle name': mass}

    Methods
    ---
    - do_augm: does the data augmentation
    """

    def __init__(self, df, daughters, mothers, p_ranges, mass_dict):
        self.df = df
        self.daughters = daughters
        self.mothers = mothers
        self.p_ranges = p_ranges
        self.mass_dict = mass_dict
        
        self.df['is_copy'] = 0

    def do_augm(self):
        augm = []
        for daughter, mother, p_range in zip(self.daughters, self.mothers, self.p_ranges):
            pmin, pmax = p_range[0], p_range[1]
            augm_df = augmentation_fine(self.df, mother, daughter, self.mass_dict[mother], self.mass_dict[daughter], pmin, pmax)
            if type(augm_df) != int:    augm.append(augm_df)

        augm.append(self.df)
        self.df = pd.concat(augm)
        return self.df

    def print_augm(self):
        """
        Prints how many particles were created for each species
        """

        for daughter in self.daughters:    
            len_daughter = len(self.df.query(f"label == '{daughter}' and copy == 1"))
            print(f'Augmented {daughter}: {len_daughter}')

## WEIGHTS CLASSES

class WeightsHandler:
    """
    Attributes
    ---
    - df: pd.DataFrame storing data
    - weight_type: string used to identify the child class

    Methods
    ---
    - assign_weights: create a new column in the df storing evaluated weights
    """

    def __init__(self, df, weight_type):
        self.df = df
        self.weight_type = weight_type
        self.weights = []
        self.conditions = []
    
    def assign_weights(self):
        self.df[self.weight_type] = np.select(self.conditions, self.weights)
        return self.df

class SingleVariable_Flattener(WeightsHandler):
    """
    Attributes
    ---
    - column: string with the name of the variable column

    Methods
    ---
    - eval_weights: evaluate weights to flatten the distribution for a single variable 
    """

    def __init__(self, df, weight_type, column):
        super().__init__(df, weight_type)
        self.column = column

    def eval_weights(self, bins, min, max):

        N = len(self.df)
        hist = TH1D('hist', 'hist', bins, min, max)
        for x in self.df[self.column]: hist.Fill(x)

        for xbin in range(1, hist.GetNbinsX()+1):
            xmin = hist.GetXaxis().GetBinLowEdge(xbin)
            xmax = xmin + hist.GetXaxis().GetBinWidth(xbin)

            if hist.GetBinContent(xbin) == 0:   self.weights.append(0.)
            else:                               self.weights.append(N/hist.GetBinContent(xbin))

            self.conditions.append((self.df[self.column] >= xmin) & (self.df[self.column] < xmax))

class DoubleVariable_Flattener(WeightsHandler):
    """
    Attributes
    ---
    - col1: string with the name of the first variable column
    - col2: string with the name of the second variable column

    Methods
    ---
    - eval_weights: evaluate weights to flatten the distribution for a single variable 
    """

    def __init__(self, df, weight_type, col1, col2):
        super().__init__(df, weight_type)
        self.col1 = col1
        self.col2 = col2

    def eval_weights(self, xbins, xmin, xmax, ybins, ymin, ymax):

        N = len(self.df)
        hist = TH2D('hist', 'hist', xbins, xmin, xmax, ybins, ymin, ymax)
        for x, y in zip(self.df[self.col1], self.df[self.col2]): hist.Fill(x, y)

        for xbin in range(1, hist.GetNbinsX()+1):
            xxmin = hist.GetXaxis().GetBinLowEdge(xbin)
            xxmax = xxmin + hist.GetXaxis().GetBinWidth(xbin)

            for ybin in range(1, hist.GetNbinsY()+1):
                yymin = hist.GetYaxis().GetBinLowEdge(ybin)
                yymax = yymin + hist.GetYaxis().GetBinWidth(ybin)

                if hist.GetBinContent(hist.GetBin(xbin, ybin)) == 0:    self.weights.append(0.)
                else:                                                   self.weights.append(N/hist.GetBinContent(hist.GetBin(xbin, ybin)))
    
                self.conditions.append((self.df[self.col1] >= xxmin) & (self.df[self.col1] < xxmax) & (self.df[self.col2] >= yymin) & (self.df[self.col2] < yymax))

################
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

###############
class Scorer:
    """
    Attrubutes
    ---
    - model: ML model
    - df: pd.DataFrame for the entire dataset
    - RegressionColumns: list of columns used for the regression
    - nameYcol: name of the column containing the actual values for the predicted variable

    Methods
    ---
    - Delta: evaluate a resolution (y - pred)/y for the model. It will be appended as a column of the df. The df is returned

    """
    def __init__(self, model, df, RegressionColumns, nameYcol, tfile) -> None:
        self.model = model
        self.df = df
        self.RegressionColumns = RegressionColumns
        self.nameYcol = nameYcol
        self.predcol = f'{self.nameYcol}_pred'
        self.plot = Plotter(self.df, tfile)

    def Delta(self, absolute=False):
        self.df[self.predcol] = self.model.predict(self.df[self.RegressionColumns])
        if absolute:    self.df.eval(f'Delta = abs({self.nameYcol} - {self.predcol}) / {self.nameYcol}', inplace=True)
        else:           self.df.eval(f'Delta = ({self.nameYcol} - {self.predcol}) / {self.nameYcol}', inplace=True)
        return self.df

    def histScore(self, nbinsx=300, xlow=-1.5, xup=1.5):
        self.plot.plot1D(['Delta'], [['#Delta', nbinsx, xlow, xup]])

    def scatterPlotScore(self, xVariable, xLabel, nbinsx, xlow, xup, nbinsy=300, ylow=-1.5, yup=1.5):
        self.plot.plot2D([xVariable], ['Delta'], [[xLabel, '#Delta', nbinsx, xlow, xup, nbinsy, ylow, yup]])

###############
class TH2Handler:
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
