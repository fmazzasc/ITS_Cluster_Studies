#
#

import numpy as np
import polars as pl

import sys
sys.path.append('..')
from utils.particles import particlePDG

from ROOT import TH2D

def weightsPandSpecies(df, cfgPID):
    
    cfg = cfgPID['weights']['weightsPandSpecies']
    momentum_slices = np.linspace(cfg['pmin'], cfg['pmax'], num=cfg['nbins']+1)

    df = df.with_columns(weightsPS=np.nan)

    total = df.filter((pl.col('p') != np.nan) & (pl.col('partID')  != np.nan)).select(pl.count()).item()

    for speciesID in df.select('partID').unique().to_series():
        for i in range(len(momentum_slices) - 1):
   
            subset = df.filter((pl.col('partID') == speciesID) & 
                               (pl.col('p') >= momentum_slices[i]) & 
                               (pl.col('p') < momentum_slices[i+1]) &
                               (pl.col('p') != np.nan))
#
            ## Calculate the fraction of elements for the current species and momentum slice
            fraction = subset.select(pl.count()).item() / total
                                                
            # Assign the weight to the current species and momentum slice
            if fraction != 0 and not np.isnan(fraction):    
                df = df.with_columns(weightsPS=pl.when((pl.col('partID') == speciesID) &
                                                        (pl.col('p') >= momentum_slices[i]) & 
                                                        (pl.col('p') < momentum_slices[i+1])).then(1/fraction).otherwise(pl.col('weightsPS')))
    
    df = df.with_columns(weightsPS=pl.when(pl.col('weightsPS') == np.nan).then(0).otherwise(pl.col('weightsPS')))

    return df