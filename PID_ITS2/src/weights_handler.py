#
#   Classes to define specific weights for a dataset that can be used in a machine learning process
#


import numpy as np
import pandas as pd

from math import sqrt

########################################
#
#   Weight classes
#_______________________________________

class WeightsHandler:
    """
    Attributes
    ---
    - df (pd.DataFrame): data
    - weight_type (str): identifies the child class

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

########################################
#
# Data Augmentation Class and Functions
#_______________________________________

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

def equal(df, column):
    """
    From a given dataframe, finds the minimum number of elements having unique values in a column. Discards elements
    having different unique values in that column until their size matches that minimum number.

    Parameters
    ----------------
    df: dataframe
    column: column where the unique values are stored

    Returns
    ----------------
    "Filtered" dataframe
    """
    
    sizes = []
    for item in df[column].unique():  sizes.append(len(df.query(f"{column} == '{item}'")))
    min_size = min(sizes)

    new_df = pd.DataFrame()
    for item in df[column].unique():  new_df = pd.concat([new_df, df.query(f"{column} == '{item}'").iloc[:min_size]], sort=False)

    return new_df

def augmentation_fine(df, mother, daughter, mass_mother, mass_daughter, pmin, pmax):
    """
    This function performs data augmentation, generating new data for the daughter species from the pre-existing data of the mother species.

    Parameters
    ----------------------------------
    - df: full dataframe of already identified particles (with a column 'label' with theie names)
    - mother: label of the mother species
    - daughter: label of the daughter species
    - mass_mother, mass_daughter: mass of the mother and the daughter
    - pmin, pmax: momentum range to perform the data augmentation in
    """

    betamin = pmin / sqrt(mass_mother**2 + pmin**2) 
    betamax = pmax / sqrt(mass_mother**2 + pmax**2) 
    mother_to_augm = df.query(f'label == "{mother}" and {betamin} <= beta < {betamax}')

    # This check should be included when working without weights
    #n_mother = len(df.query(f'label == "{mother}" and {pmin} <= p < {pmax}'))
    #n_daughter = len(df.query(f'label == "{daughter}" and {pmin} <= p < {pmax}'))
    #
    #
    #if n_mother < n_daughter:   return 0
    #else:   n_sample = min(n_mother-n_daughter, len(mother_to_augm))
    
    n_sample = len(mother_to_augm)
    augm_daughter = mother_to_augm.sample(n_sample)

    augm_daughter['p'] = augm_daughter['p'] * mass_daughter / mass_mother
    augm_daughter['label'] = daughter
    augm_daughter['copy'] = 1

    
    return augm_daughter