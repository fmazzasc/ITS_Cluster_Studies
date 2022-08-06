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

run_num = 520150
df_V0 = uproot.open(f'../results/V0TreePIDITS_{run_num}.root')["V0Tree"].arrays(library='pd')
df_dau = uproot.open(f'../results/V0TreePIDITS_{run_num}.root')["DauTree"].arrays(library='pd')

df_V0.query('V0CosPA>0.999', inplace=True)
## selecting lambda daughters
df_V0['V0ArmenterosAlpha'] = df_V0['V0ArmenterosAlpha'].astype(float)
v0_cuts_alpha_pos = 'V0ArmenterosAlpha > 0.5 and abs(nSigmaPosDauP)<3 and abs(lamMassHyp-1.116)<0.01'
v0_cuts_alpha_neg = 'V0ArmenterosAlpha < -0.5 and abs(nSigmaNegDauP)<3 and abs(lamMassHyp-1.116)<0.01'

filter_V0_alpha_pos = df_V0.query(v0_cuts_alpha_pos)
filter_V0_alpha_neg = df_V0.query(v0_cuts_alpha_neg)
filtered_V0 = pd.concat([filter_V0_alpha_pos, filter_V0_alpha_neg])

## pass filter to df_dau
filter_dau_alpha_pos = df_dau.v0Ind.isin(filter_V0_alpha_pos['v0Ind'])
filter_dau_alpha_neg = df_dau.v0Ind.isin(filter_V0_alpha_neg['v0Ind'])

## protons --> arm alpha > 0 || anti protons --> arm alpha < 0
df_dau_protons = pd.concat([df_dau[filter_dau_alpha_pos].query('isPositive==1'), df_dau[filter_dau_alpha_neg].query('isPositive==0')], ignore_index=True)
df_dau_pi = pd.concat([df_dau[filter_dau_alpha_pos].query('isPositive==0'), df_dau[filter_dau_alpha_neg].query('isPositive==1')], ignore_index=True)


## important remove pileup events!
h_rof_bc = ROOT.TH1F('h_rof_bc', 'h_rof_bc; rof BC; Counts', 2000, -0.5, 1999.5)
fill_hist(h_rof_bc, df_dau_protons, 'rofBC')
df_dau_protons.query('rofBC < 400', inplace=True)
df_dau_pi.query('rofBC < 400', inplace=True)



h_phot_mass = ROOT.TH1F('h_phot_mass', 'h_phot_mass; M_{#gamma} (GeV/{c^{2}}); Counts' , 100, 0., 0.05)
h_k0s_mass = ROOT.TH1F('h_k0s_mass', 'h_k0s_mass; M_{k0_{s}} (GeV/{c^{2}}); Counts', 100, 0.45, 0.55)
h_lam_mass = ROOT.TH1F('h_lam_mass', 'h_v0_mass; M_{#Lambda} (GeV/{c^{2}}); Counts', 100, 1.1, 1.2)

h_th2_mass = ROOT.TH2F('h_th2_mass', 'h_th2_mass; M_{#Lambda} (GeV/c^{2}); M_{k0_{s}} (GeV/c^{2}) ; Counts', 100, 1., 1.4, 100, 0.4, 0.7)
h_th2_armenteros = ROOT.TH2F('h_th2_arm', '; #alpha^{arm}; q_{T}^{arm} ; Counts', 400, -1., 1., 400, 0., 0.3)


h_proton_mean_clus = ROOT.TH2F('h_proton_mean_clus', 'h_proton_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_pion_mean_clus = ROOT.TH2F('h_pion_mean_clus', 'h_pion_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_proton_mean_dedx = ROOT.TH2F('h_proton_mean_dedx', 'h_proton_mean_dedx',100, 0, 2, 100, 30, 400)
h_pion_mean_dedx = ROOT.TH2F('h_pion_mean_dedx', 'h_pion_mean_dedx',100, 0, 2, 100, 30, 400)


fill_hist(h_lam_mass, df_V0, 'lamMassHyp')
fill_hist(h_k0s_mass, df_V0, 'k0sMassHyp')
fill_hist(h_phot_mass, df_V0, 'photMassHyp')
fill_hist_2d(h_th2_mass, df_V0, 'lamMassHyp', 'k0sMassHyp')
fill_hist_2d(h_th2_armenteros, df_V0, 'V0ArmenterosAlpha', 'V0ArmenterosQt')
fill_hist_2d(h_proton_mean_dedx, df_dau_protons,'pTPC', 'dedx')
fill_hist_2d(h_pion_mean_dedx, df_dau_pi,'pTPC', 'dedx')
fill_hist_2d(h_proton_mean_clus, df_dau_protons,'pTPC', 'clSizeCosLam')
fill_hist_2d(h_pion_mean_clus, df_dau_pi,'pTPC', 'clSizeCosLam')

outFile = ROOT.TFile(f'../results/test_v0_builder_{run_num}.root', 'RECREATE')
h_lam_mass.Write()
h_k0s_mass.Write()
h_phot_mass.Write()
h_th2_mass.Write()
h_th2_armenteros.Write()

h_proton_mean_clus.Write()
h_pion_mean_clus.Write()
h_proton_mean_dedx.Write()
h_pion_mean_dedx.Write()

h_rof_bc.Write()
outFile.Close()

