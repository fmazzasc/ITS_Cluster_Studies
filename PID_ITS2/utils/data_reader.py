#
#   Functions and classes to easily import data from files
#

import pandas as pd
import os
import uproot

def readFile(filePath, treeName=None):
    """
    Function to read a file with the adequate extension and return a pandas dataframe

    Parameters
    ----------
        filePath (str): Path to the file to read
        treeName (str): Name of the TTree to open with uproot
    """

    file_ext = os.path.splitext(filePath)[1]
    if file_ext == '.gzip':     file_ext = os.path.splitext(os.path.splitext(filePath)[0])[1]

    if file_ext == '.parquet':  data = pd.read_parquet(filePath)
    elif file_ext == '.csv':    data = pd.read_csv(filePath)
    elif file_ext == '.root':   data = uproot.open(filePath)[treeName].arrays(library='pd')
    else:                       raise ValueError(f'Invalid file extension: {file_ext}')

    return data



