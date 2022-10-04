import ROOT
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib
from enum import Enum
ROOT.gROOT.SetBatch()

def fill_hist(hist, dataframe, column):
    for val in dataframe[column]:
        hist.Fill(val)
    return hist

def fill_hist_2d(hist, dataframe, column1, column2):
    for val1, val2 in zip(dataframe[column1], dataframe[column2]):
        hist.Fill(val1, val2)
    return hist

class kParticle(Enum):
    ambiguous = -1
    proton = 1
    kaon = 2
    pion = 3
    electron = 4

#-----------------------------------------------------------------------------------------------------------------------

## few output histograms
h_proton_mean_clus = ROOT.TH2F('h_proton_mean_clus', 'h_proton_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_proton_mean_dedx = ROOT.TH2F('h_proton_mean_dedx', 'h_proton_mean_dedx',100, 0, 2, 100, 30, 400)

h_kaon_mean_clus = ROOT.TH2F('h_kaon_mean_clus', 'h_kaon_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_kaon_mean_dedx = ROOT.TH2F('h_kaon_mean_dedx', 'h_kaon_mean_dedx',100, 0, 2, 100, 30, 400)

h_pion_mean_clus = ROOT.TH2F('h_pion_mean_clus', 'h_pion_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_pion_mean_dedx = ROOT.TH2F('h_pion_mean_dedx', 'h_pion_mean_dedx',100, 0, 2, 100, 30, 400)

h_electron_mean_clus = ROOT.TH2F('h_electron_mean_clus', 'h_electron_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_electron_mean_dedx = ROOT.TH2F('h_electron_mean_dedx', 'h_electron_mean_dedx',100, 0, 2, 100, 30, 400)
#-----------------------------------------------------------------------------------------------------------------------



run_number = 520147

## get ITS-TPC and V0 trees
df_primaries = uproot.open(f'../results/ITSTPCClusterTree{run_number}.root')["ITStreeML"].arrays(library='pd')
print("Assembling primaries from ITS-TPC...")
df_prim = df_primaries.query('tpcITSchi2<10')

pid_cols = ['nSigmaP', 'nSigmaK', 'nSigmaPi', 'nSigmaE']
pid_arr = df_primaries[pid_cols].to_numpy()
pid_arr = np.abs(pid_arr)<3
particle_type = np.zeros(len(df_primaries))

## sum all the columns
pid_sum = np.sum(pid_arr, axis=1)

#create mask for ambiguous particles
mask_amb = np.logical_or(pid_sum==0, pid_sum>1)
mask_proton = np.logical_and(pid_arr[:,0], np.logical_not(mask_amb))
mask_kaon = np.logical_and(pid_arr[:,1], np.logical_not(mask_amb))
mask_pion = np.logical_and(pid_arr[:,2], np.logical_not(mask_amb))
mask_electron = np.logical_and(pid_arr[:,3], np.logical_not(mask_amb))
particle_type[mask_amb] = kParticle.ambiguous.value
particle_type[mask_proton] = kParticle.proton.value
particle_type[mask_kaon] = kParticle.kaon.value
particle_type[mask_pion] = kParticle.pion.value
particle_type[mask_electron] = kParticle.electron.value
df_prim['particle'] = particle_type


fill_hist_2d(h_proton_mean_clus, df_prim.query('particle==1'),'pTPC', 'clSizeCosLam')
fill_hist_2d(h_proton_mean_dedx, df_prim.query('particle==1'),'pTPC', 'dedx')

fill_hist_2d(h_kaon_mean_clus, df_prim.query('particle==2'),'pTPC', 'clSizeCosLam')
fill_hist_2d(h_kaon_mean_dedx, df_prim.query('particle==2'),'pTPC', 'dedx')  

fill_hist_2d(h_pion_mean_clus, df_prim.query('particle==3'),'pTPC', 'clSizeCosLam')
fill_hist_2d(h_pion_mean_dedx, df_prim.query('particle==3'),'pTPC', 'dedx')

fill_hist_2d(h_electron_mean_clus, df_prim.query('particle==4'),'pTPC', 'clSizeCosLam')
fill_hist_2d(h_electron_mean_dedx, df_prim.query('particle==4'),'pTPC', 'dedx')


df_prim.to_parquet(f'../results/particles_pid_{run_number}_itstpc.parquet')


outfile_root = ROOT.TFile(f'../results/particles_pid_{run_number}_itstpc.root', 'RECREATE')
h_proton_mean_clus.Write()
h_proton_mean_dedx.Write()
h_kaon_mean_clus.Write()
h_kaon_mean_dedx.Write()
h_pion_mean_clus.Write()
h_pion_mean_dedx.Write()
h_electron_mean_clus.Write()
h_electron_mean_dedx.Write()
outfile_root.Close()