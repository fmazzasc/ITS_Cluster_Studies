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

def get_range_around_value(histo, value, portion):
    """
    Find upper and lower limits of an interval containing a certain portion of the distribution. The interval
    should be centered around a given value

    Parameters
    ----------
        histo (TH1): histogram of the distribution
        value (float): value around which the interval should be centered
        portion (float): value between 0 and 1. Portion of the distribution that should be included in the interval

    Returns
    ----------
        lower_limit (float): lower limit of the interval
        upper_limit (float): upper limit of the interval
    """
    nbins = histo.GetNbinsX()
    cdf = [histo.Integral(0, i)/histo.Integral() for i in range(nbins+1)]

    # Find bin corresponding to input value
    bin_value = histo.FindBin(value)

    # Find bin corresponding to desired portion of the distribution
    bin_portion = 0
    for i in range(nbins+1):
        if cdf[i] > portion:
            bin_portion = i
            break

    # Calculate bin range that contains desired portion of distribution
    portion_content = histo.Integral(bin_value, bin_portion)/histo.Integral()
    bin_range = int((bin_portion - bin_value)/2.0)
    while portion_content < portion:
        bin_range += 1
        portion_content = histo.Integral(bin_value-bin_range, bin_portion+bin_range)/histo.Integral()
    
    # Calculate lower and upper limits of bin range
    lower_limit = histo.GetBinLowEdge(bin_value-bin_range)
    upper_limit = histo.GetBinLowEdge(bin_portion+bin_range+1)

    return lower_limit, upper_limit

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
    data_filt = data.query(f"label == '{particle}'", inplace=False)
    hist = TH1D(f'discrepancy_{particle}', f'discrepancy_{particle}', 100, data_filt[f'discrepancy_{particle}'].min()-0.1, data_filt[f'discrepancy_{particle}'].max()+0.1)        
    for x in data_filt[f'discrepancy_{particle}']:   hist.Fill(x)

    hist.GetXaxis().SetTitle(f'discrepancy_{particle}')
    hist.GetYaxis().SetTitle('Counts')

    outFile.cd()
    hist.SetDrawOption('hist')
    hist.Write()

    return hist.GetStdDev()

def get_efficiency(data, particle, sigma):
    """
    Returns the efficiency of a given sample. Accepted particles have predicted beta in a window of 1 sigma from 
    the measured value. The sigma is the dispersion of the distribution for discrepancy=beta_pred-beta_true.

    Parameters
    ----------
        data (pd.DataFrame): Reference dataset
        particle (str): particle species to consider (accepted values are 'Deu', 'P', 'K', 'Pi', 'E')
    """
    #sigma = data[f'discrepancy_{particle}'].std()
    #mean = data[f'discrepancy_{particle}'].mean()
    mean = 0.

    n_accepted = len(data.query(f"label == '{particle}' and {mean-sigma} < (discrepancy_{particle}) < {mean+sigma}", inplace=False))
    n_total = len(data.query(f"label == '{particle}'", inplace=False))

    sample_efficiency = n_accepted / n_total
    return sample_efficiency

def get_purity(data, particle, sigma):
    """
    Returns the purity of a given sample. Accepted particles have predicted beta in a window of 1 sigma from 
    the measured value. The sigma is the dispersion of the distribution for discrepancy=beta_pred-beta_true.

    Parameters
    ----------
        data (pd.DataFrame): Reference dataset
        particle (str): particle species to consider (accepted values are 'Deu', 'P', 'K', 'Pi', 'E')
    """

    #sigma = data[f'discrepancy_{particle}'].std()
    #mean = data[f'discrepancy_{particle}'].mean()
    mean = 0.

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
        sigma = define_discrepancy(inputData, name, outFile)
        print(f'Purity: ', get_purity(inputData, name, sigma))
        print(f'Efficiency: ', get_efficiency(inputData, name, sigma))

    outFile.Close()



if __name__ == '__main__':
    
    inputFile = '../data/preprocessed/TPC/ApplicationDf_beta_pflat.parquet.gzip'
    outFilePath = '../output/analysis/discrepancy.root'
    inputData = readFile(inputFile)
    analysis(inputData, outFilePath)