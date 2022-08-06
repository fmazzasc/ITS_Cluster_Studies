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

df = uproot.open(f'../results/ITSTPCClusterTree{run_number}.root')["ITStreeML"].arrays(library='pd')
df.eval('momRes = (p - pTPC)/pTPC', inplace=True)
df.eval('momResITS = (pITS - pTPC)/pTPC', inplace=True)

df_proton = df.query('abs(nSigmaP)<1 and tpcITSchi2<10 and rofITS<400')
df_pion = df.query('abs(nSigmaPi)<1')

print(len(df_proton), len(df_pion))

h_proton_clus = ROOT.TH1F('h_proton_clus', 'h_proton_clus; ClSize; Counts', 80, 0.5, 10.5)
h_proton_TPC_clus = ROOT.TH1F('h_proton_TPC_clus', 'h_proton_clus; ClSize; Counts', 200, 0, 160)

h_rof_bc = ROOT.TH1F('h_rof_bc', 'h_rof_bc; BC; Counts', 2000, -0.5, 1999.5)

h_pion_clus = ROOT.TH1F('h_pion_clus', 'h_pion_clus; ClSize; Counts', 80, 0.5, 10.5)
h_mom_res = ROOT.TH2F('h_mom_res', 'h_mom_res; pITSTPC;(pITSTPC - pTPC)/pITSTPC; Counts', 50,0,2,50, -1, 1)
h_proton_mean_dedx = ROOT.TH2F('h_proton_mean_dedx', 'h_proton_mean_dedx',100, 0, 2, 100, 30, 500)
h_pion_mean_dedx = ROOT.TH2F('h_pion_mean_dedx', 'h_pion_mean_dedx',100, 0, 2, 100, 30, 500)
h_proton_chi2_clus = ROOT.TH2F('h_proton_chi2_clus', 'h_proton_chi2_clus',20, 0, 10, 50, 0.5, 10.5)
h_proton_hits_clus = ROOT.TH2F('h_proton_hits_clus', 'h_proton_hits_clus',4, 3.5, 7.5, 50, 0.5, 10.5)
h_proton_its_chi2_low = ROOT.TH1F('h_proton_its_chi2_low', 'h_proton_its_chi2_low',100, 0., 2)
h_proton_its_chi2_high = ROOT.TH1F('h_proton_its_chi2_high', 'h_proton_its_chi2_high',100, 0., 2)
h_proton_its_mom_res_low = ROOT.TH1F('h_proton_its_mom_res_low', 'h_proton_its_mom_res_low',100, -1., 1)
h_proton_its_mom_res_high = ROOT.TH1F('h_proton_its_mom_res_high', 'h_proton_its_mom_res_high',100, -1., 1.) 

h_proton_p_clus = ROOT.TH2F('h_proton_p_clus', 'h_proton_p_clus',10, 0, 2, 50, 0.5, 10.5)
h_pion_chi2_clus = ROOT.TH2F('h_pion_chi2_clus', 'h_pion_chi2_clus',10, 0, 1, 50, 0.5, 10.5)
h_pion_p_clus = ROOT.TH2F('h_pion_p_clus', 'h_pion_p_clus',10, 0, 2, 50, 0.5, 10.5)


fill_hist(h_proton_clus, df_proton, 'meanClsize')
fill_hist(h_proton_TPC_clus, df_proton, 'nClusTPC')
fill_hist(h_pion_clus, df_pion, 'meanClsize')
fill_hist(h_rof_bc, df_proton, 'rofITS')

fill_hist_2d(h_proton_mean_dedx, df_proton,'p', 'dedx')
fill_hist_2d(h_pion_mean_dedx, df_pion,'p', 'dedx')
fill_hist_2d(h_mom_res, df_proton, 'p', 'momRes')
fill_hist_2d(h_proton_chi2_clus, df_proton.query('p<0.6'),'tpcITSchi2', 'meanClsize')
fill_hist_2d(h_proton_hits_clus, df_proton.query('p<0.6'),'nClusITS', 'meanClsize')
fill_hist_2d(h_proton_p_clus, df_proton,'p', 'meanClsize')
fill_hist_2d(h_pion_chi2_clus, df_pion,'tpcITSchi2', 'meanClsize')
fill_hist_2d(h_pion_p_clus, df_pion,'p', 'meanClsize')



pd.set_option('display.max_columns', None)
df_bug = df_proton.query('pTPC<0.6 and meanClsize<1.5')
df_proper = df_proton.query('pTPC<0.6 and meanClsize>3.5')

fill_hist(h_proton_its_chi2_low, df_bug, 'itsChi2')
fill_hist(h_proton_its_chi2_high, df_proper, 'itsChi2')
fill_hist(h_proton_its_mom_res_low, df_bug, 'momResITS')
fill_hist(h_proton_its_mom_res_high, df_proper, 'momResITS')

print(df_bug[['ClSizeL0', 'ClSizeL1', 'ClSizeL2', 'ClSizeL3', 'ClSizeL4', 'ClSizeL5', 'ClSizeL6', 'pITS', 'pTPC']])
print("----------------------------------------------------")
print(df_proper[['ClSizeL0', 'ClSizeL1', 'ClSizeL2', 'ClSizeL3', 'ClSizeL4', 'ClSizeL5', 'ClSizeL6', 'pITS', 'pTPC']])


outfile = ROOT.TFile(f'../results/h_proton_clus_{run_number}.root', 'recreate')
h_proton_clus.Write()
h_proton_TPC_clus.Write()
h_proton_mean_dedx.Write()
h_pion_clus.Write()
h_pion_mean_dedx.Write()
h_mom_res.Write()
h_proton_chi2_clus.Write()
h_proton_p_clus.Write()
h_pion_chi2_clus.Write()
h_pion_p_clus.Write()
h_proton_hits_clus.Write()

h_proton_its_chi2_low.Write()
h_proton_its_chi2_high.Write()
h_proton_its_mom_res_low.Write()
h_proton_its_mom_res_high.Write()
h_rof_bc.Write()

outfile.Close()
