#
#   Load data with ML predictions and perform analysis
#

import sys

from ROOT import TFile, TH1D

sys.path.append('..')
from utils.data_reader import readFile

#################################################################################################################
#   GLOBAL  CONSTANTS

# Tags
#_____________________________________
tag_Deu = 'nSigmaDeuAbs < 1 and p <= 1.2'
tag_P = 'nSigmaPAbs < 1 and p <= 1.5'
tag_K = 'nSigmaKAbs < 1 and p <= 1.5'
tag_Pi = 'nSigmaPiAbs < 1 and p <= 1.5'
tag_E = 'nSigmaEAbs < 1 and p <= 1.5'

# Masses
#_____________________________________
mass_Deu = 1.8756
mass_P =  0.93827200
mass_K = 0.4937
mass_Pi = 0.13957000
mass_E = 0.000511

#names = ['Deu', 'P', 'K', 'Pi', 'E']
names = ['P', 'K', 'Pi', 'E']

tag_dict = dict(zip(names, [tag_Deu, tag_P, tag_K, tag_Pi, tag_E]))
mass_dict = dict(zip(names, [mass_Deu, mass_P, mass_K, mass_Pi, mass_E]))

MC_dict = {2212: 1, 321: 2, 211: 3, 11: 4}

#################################################################################################################
#   General functions
#
#################################################################################################################



#################################################################################################################
#   Specific functions
#
#################################################################################################################

def define_discrepancy(data, particle, outFile):
    """
    Define discrepancy between predicted and correct value of beta for a particle, given it belongs to a 
    defined species.

    Parameters
    ----------
        data (pd.DataFrame): Reference dataset
        particle (str): Particle species to consider (accepted values are 'Deu', 'P', 'K', 'Pi', 'E')   
        outFile (TFile): File where the distribution is saved 
    """

    data.eval(f"beta_{particle} = p / sqrt({mass_dict[particle]}**2 + p**2)", inplace=True)
    data.eval(f"discrepancy_{particle} = beta_pred - beta_{particle}", inplace=True)

    # Plot into a histogram
    hist = TH1D(f'discrepancy_{particle}', f'discrepancy_{particle}', 100, data[f'discrepancy_{particle}'].min()-0.1, data[f'discrepancy_{particle}'].max()+0.1)        
    for x in data[f'discrepancy_{particle}']:   hist.Fill(x)

    hist.GetXaxis().SetTitle(f'discrepancy_{particle}')
    hist.GetYaxis().SetTitle('Counts')

    outFile.cd()
    hist.SetDrawOption('hist')
    hist.Write()

def get_efficiency(data, particle):
    """
    Returns the efficiency of a given sample. Accepted particles have predicted beta in a window of 1 sigma from 
    the measured value. The sigma is the dispersion of the distribution for discrepancy=beta_pred-beta_true.

    Parameters
    ----------
        data (pd.DataFrame): Reference dataset
        particle (str): particle species to consider (accepted values are 'Deu', 'P', 'K', 'Pi', 'E')
    """
    sigma = data[f'discrepancy_{particle}'].std()
    mean = data[f'discrepancy_{particle}'].mean()

    n_accepted = len(data.query(f"label == '{particle}' and {mean-sigma} < (discrepancy_{particle}) < {mean+sigma}", inplace=False))
    n_total = len(data.query(f"label == '{particle}'", inplace=False))

    sample_efficiency = n_accepted / n_total
    return sample_efficiency

def get_purity(data, particle):
    """
    Returns the purity of a given sample. Accepted particles have predicted beta in a window of 1 sigma from 
    the measured value. The sigma is the dispersion of the distribution for discrepancy=beta_pred-beta_true.

    Parameters
    ----------
        data (pd.DataFrame): Reference dataset
        particle (str): particle species to consider (accepted values are 'Deu', 'P', 'K', 'Pi', 'E')
    """

    sigma = data[f'discrepancy_{particle}'].std()
    mean = data[f'discrepancy_{particle}'].mean()

    n_accepted = len(data.query(f"label == '{particle}' and {mean-sigma} < (discrepancy_{particle}) < {mean+sigma}", inplace=False))
    all_accepted = len(data.query(f"{mean-sigma} < (discrepancy_{particle}) < {mean+sigma}", inplace=False))

    sample_purity = n_accepted / all_accepted
    return sample_purity

def analysis(inputData, outFilePath):
    """
    For each particle species, evaluate purity and efficiency in the PID.

    Parameters
    ----------
        inputData (pd.DataFrame): Input dataframe which the ml model has been applied on
    """

    outFile = TFile(outFilePath, 'recreate')
    print(f'ROOT file created at {outFilePath}')

    for name in names:
        print(f'\nParticle: {name}')
        define_discrepancy(inputData, name, outFile)
        print(f'Purity: ', get_purity(inputData, name))
        print(f'Efficiency: ', get_efficiency(inputData, name))

    outFile.Close()



if __name__ == '__main__':
    
    inputFile = '../data/preprocessed/TPC/ApplicationDf_beta_pflat .parquet.gzip'
    outFilePath = '../output/analysis/discrepancy.root'
    inputData = readFile(inputFile)
    analysis(inputData, outFilePath)