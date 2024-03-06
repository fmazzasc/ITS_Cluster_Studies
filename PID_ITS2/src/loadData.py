#
#

import os
import polars as pl
import uproot

import sys 
sys.path.append('..')
from utils.color import color

def LoadData(inFiles:list):
    '''
    Load data from multiple files

    Parameters
    ----------
    inFiles (list): list of input files  
    '''

    df = pl.DataFrame()
    for inFile in inFiles:
        df_tmp = LoadDataFile(inFile)
        if df_tmp is not None:  df = pl.concat([df, df_tmp])

    return df

def LoadDataFile(inFile:str):
    '''
    Load data from a single file
    
    Parameters
    ----------
    inFile (str): input file  
    '''

    # check if the file exists
    if not os.path.exists(inFile):  
        print("File not found: "+color.UNDERLINE+color.RED+f'{inFile}'+color.END)
        return None
    
    print("Loading data from: "+color.UNDERLINE+color.BLUE+f'{inFile}'+color.END)
    if inFile.endswith(".root"):        df = LoadRoot(inFile)   
    elif inFile.endswith(".parquet"):   df = LoadParquet(inFile)
    else:
        print("Unknown file type: "+color.UNDERLINE+color.RED+f'{inFile}'+color.END)
        return None

    return df

def LoadRoot(inFile):
    '''
    Load data from a ROOT file

    Parameters
    ----------
    inFile (str): input file
    '''
    
    tree = uproot.open(inFile)["outTree"]
    df = tree.arrays(library="pd", how="zip")
    df = pl.from_pandas(df)

    return df

def LoadParquet(inFile):
    '''
    Load data from a parquet file

    Parameters
    ----------
    inFile (str): input file
    '''
    
    df = pl.read_parquet(inFile)
    
    return df