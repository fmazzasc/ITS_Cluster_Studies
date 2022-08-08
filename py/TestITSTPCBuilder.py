import ROOT
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib

ROOT.gROOT.SetBatch()

def fill_hist(hist, dataframe, column):
    for val in dataframe[column]:
        hist.Fill(val)
    return hist

def fill_hist_2d(hist, dataframe, column1, column2):
    for val1, val2 in zip(dataframe[column1], dataframe[column2]):
        hist.Fill(val1, val2)
    return hist

run_number = 520143

h_proton_clus = ROOT.TH1F('h_proton_clus', 'h_proton_clus; ClSize; Counts', 80, 0.5, 10.5)
h_rof_bc = ROOT.TH1F('h_rof_bc', 'h_rof_bc; BC; Counts', 2000, -0.5, 1999.5)
h_pion_clus = ROOT.TH1F('h_pion_clus', 'h_pion_clus; ClSize; Counts', 80, 0.5, 10.5)
h_mom_res = ROOT.TH2F('h_mom_res', 'h_mom_res; pITSTPC;(pITSTPC - pTPC)/pITSTPC; Counts', 50,0,2,50, -1, 1)
h_proton_mean_dedx = ROOT.TH2F('h_proton_mean_dedx', 'h_proton_mean_dedx',100, 0, 2, 100, 30, 500)
h_pion_mean_dedx = ROOT.TH2F('h_pion_mean_dedx', 'h_pion_mean_dedx',100, 0, 2, 100, 30, 500)
h_proton_chi2_clus = ROOT.TH2F('h_proton_chi2_clus', 'h_proton_chi2_clus',20, 0, 10, 50, 0.5, 10.5)
h_proton_p_clus = ROOT.TH2F('h_proton_p_clus', 'h_proton_p_clus',10, 0, 2, 50, 0.5, 10.5)
h_pion_p_clus = ROOT.TH2F('h_pion_p_clus', 'h_pion_p_clus',10, 0, 2, 50, 0.5, 10.5)

h_proton_clus_slice = ROOT.TH1F('h_proton_clus_slice', 'h_proton_clus_slice; < Cluster Size > x Cos(#lambda); Counts', 50, 0.5, 10.5)
h_proton_clus_slice_bc0 = ROOT.TH1F('h_proton_clus_slice_0', 'h_proton_clus_slice_0; < Cluster Size > x Cos(#lambda); Counts', 50, 0.5, 10.5)
h_proton_clus_slice_bc1 = ROOT.TH1F('h_proton_clus_slice_1', 'h_proton_clus_slice_1; < Cluster Size > x Cos(#lambda); Counts', 50, 0.5, 10.5)
h_proton_clus_slice_bc2 = ROOT.TH1F('h_proton_clus_slice_2', 'h_proton_clus_slice_2; < Cluster Size > x Cos(#lambda); Counts', 50, 0.5, 10.5)

h_rof_bc0 = ROOT.TH1F('h_rof_bc0', 'h_rof_bc0; BC; Counts', 2000, -0.5, 1999.5)
h_rof_bc1 = ROOT.TH1F('h_rof_bc1', 'h_rof_bc1; BC; Counts', 2000, -0.5, 1999.5)
h_rof_bc2 = ROOT.TH1F('h_rof_bc2', 'h_rof_bc2; BC; Counts', 2000, -0.5, 1999.5)


df = uproot.open(f'../results/ITSTPCClusterTree{run_number}.root')["ITStreeML"].arrays(library='pd')
df.eval('momRes = (p - pTPC)/pTPC', inplace=True)
df.eval('momResITS = (pITS - pTPC)/pTPC', inplace=True)

df_proton = df.query('abs(nSigmaP)<1')
df_pion = df.query('abs(nSigmaPi)<1')

fill_hist(h_proton_clus_slice, df_proton.query('0.4<pTPC<0.5'), 'clSizeCosLam')
fill_hist(h_proton_clus_slice_bc0, df_proton.query('rofBC<400 and 0.4<pTPC<0.5'), 'clSizeCosLam')
fill_hist(h_proton_clus_slice_bc1, df_proton.query('rofBC>400 and rofBC<1200 and 0.4<pTPC<0.5'), 'clSizeCosLam')
fill_hist(h_proton_clus_slice_bc2, df_proton.query('rofBC>1200 and 0.4<pTPC<0.5'), 'clSizeCosLam')
fill_hist(h_rof_bc0, df_proton.query('rofBC<400'), 'rofBC')
fill_hist(h_rof_bc1, df_proton.query('rofBC>400 and rofBC<1200'), 'rofBC')
fill_hist(h_rof_bc2, df_proton.query('rofBC>1200'), 'rofBC')



fill_hist(h_proton_clus, df_proton, 'clSizeCosLam')
fill_hist(h_pion_clus, df_pion, 'clSizeCosLam')
fill_hist(h_rof_bc, df_proton, 'rofBC')

fill_hist_2d(h_proton_mean_dedx, df_proton,'pTPC', 'dedx')
fill_hist_2d(h_pion_mean_dedx, df_pion,'pTPC', 'dedx')
fill_hist_2d(h_mom_res, df_proton, 'pTPC', 'momRes')
fill_hist_2d(h_proton_chi2_clus, df_proton.query('p<0.6'),'tpcITSchi2', 'clSizeCosLam')
fill_hist_2d(h_proton_p_clus, df_proton,'pTPC', 'clSizeCosLam')
fill_hist_2d(h_pion_p_clus, df_pion,'pTPC', 'clSizeCosLam')



outfile = ROOT.TFile(f'../results/h_proton_clus_{run_number}.root', 'recreate')
h_proton_clus.Write()
h_proton_mean_dedx.Write()
h_pion_clus.Write()
h_pion_mean_dedx.Write()
h_mom_res.Write()
h_proton_p_clus.Write()
h_pion_p_clus.Write()
h_rof_bc.Write()
h_rof_bc0.Write()
h_rof_bc1.Write()
h_rof_bc2.Write()

h_proton_clus_slice.Write()
h_proton_clus_slice_bc0.Write()
h_proton_clus_slice_bc1.Write()
h_proton_clus_slice_bc2.Write()





outfile.Close()
