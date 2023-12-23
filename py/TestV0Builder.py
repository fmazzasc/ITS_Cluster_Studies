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


run_num = 520271
lambda_mass = 1.116
k0s_mass = 0.4976

h_phot_mass = ROOT.TH1F('h_phot_mass', 'h_phot_mass; M_{#gamma} (GeV/{c^{2}}); Counts' , 100, 0., 0.001)
h_k0s_mass = ROOT.TH1F('h_k0s_mass', 'h_k0s_mass; M_{k0_{s}} (GeV/{c^{2}}); Counts', 100, 0.45, 0.55)
h_lam_mass = ROOT.TH1F('h_lam_mass', 'h_v0_mass; M_{#Lambda} (GeV/{c^{2}}); Counts', 100, 1.08, 1.15)
h_th2_armenteros = ROOT.TH2F('h_th2_arm', '; #alpha^{arm}; q_{T}^{arm} ; Counts', 400, -1., 1., 400, 0., 0.3)
h_rof_bc = ROOT.TH1F('h_rof_bc', 'h_rof_bc; rof BC; Counts', 2000, -0.5, 1999.5)
h_th2_mass = ROOT.TH2F('h_th2_mass', 'h_th2_mass; M_{#Lambda} (GeV/c^{2}); M_{k0_{s}} (GeV/c^{2}) ; Counts', 100, 1., 1.4, 100, 0.4, 0.7)
h_proton_mean_clus = ROOT.TH2F('h_proton_mean_clus', 'h_proton_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_pion_mean_clus = ROOT.TH2F('h_pion_mean_clus', 'h_pion_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_electron_mean_clus = ROOT.TH2F('h_electron_mean_clus', 'h_electron_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_proton_mean_dedx = ROOT.TH2F('h_proton_mean_dedx', 'h_proton_mean_dedx',100, 0, 2, 100, 30, 400)
h_pion_mean_dedx = ROOT.TH2F('h_pion_mean_dedx', 'h_pion_mean_dedx',100, 0, 2, 100, 30, 400)
h_electron_mean_dedx = ROOT.TH2F('h_electron_mean_dedx', 'h_electron_mean_dedx',100, 0, 2, 100, 30, 400)



df_V0 = uproot.open(f'../results/V0TreePIDITS_{run_num}.root')["V0Tree"].arrays(library='pd')
df_dau = uproot.open(f'../results/V0TreePIDITS_{run_num}.root')["DauTree"].arrays(library='pd')

## important remove pileup events!
fill_hist(h_rof_bc, df_dau, 'rofBC')
df_dau.query('rofBC < 400', inplace=True)


## cosPA prefiltering
df_V0.query('V0CosPA>0.999', inplace=True)
df_V0['V0ArmenterosAlpha'] = df_V0['V0ArmenterosAlpha'].astype(float)


## selecting protons from lambdas and anti-protons from anti-lambdas
lambda_cuts = 'abs(V0ArmenterosAlpha) > 0.5 and abs(nSigmaPosDauP)<3 and (k0sMassHyp<0.47 or k0sMassHyp>0.52)'
filter_lambda = df_V0.query(lambda_cuts)
df_protons = df_dau[df_dau.v0Ind.isin(filter_lambda['v0Ind'])]
df_protons.query('abs(nSigmaP)<3', inplace=True)


## selecting pions from k0s
k0s_cuts = 'abs(V0ArmenterosAlpha) < 0.8 and V0ArmenterosQt > 0.12 and (lamMassHyp<1.10 or lamMassHyp>1.13)'
filter_k0s = df_V0.query(k0s_cuts)
df_pions = df_dau[df_dau.v0Ind.isin(filter_k0s['v0Ind'])]
df_pions.query('abs(nSigmaPi)<3', inplace=True)


# selecting electrons from conversions
phot_conversion_cuts = 'abs(V0ArmenterosAlpha)<0.3 and V0ArmenterosQt < 0.01 and photMassHyp < 1e-5'
filter_phot_conversion = df_V0.query(phot_conversion_cuts)
df_electrons = df_dau[df_dau.v0Ind.isin(filter_phot_conversion['v0Ind'])]
df_electrons.query('dedx>60', inplace=True)

fill_hist(h_lam_mass, filter_lambda, 'lamMassHyp')
fill_hist(h_k0s_mass, filter_k0s, 'k0sMassHyp')
fill_hist(h_phot_mass, filter_phot_conversion, 'photMassHyp')
fill_hist_2d(h_th2_mass, df_V0, 'lamMassHyp', 'k0sMassHyp')
fill_hist_2d(h_th2_armenteros, df_V0, 'V0ArmenterosAlpha', 'V0ArmenterosQt')
fill_hist_2d(h_proton_mean_dedx, df_protons,'pTPC', 'dedx')
fill_hist_2d(h_pion_mean_dedx, df_pions,'pTPC', 'dedx')
fill_hist_2d(h_electron_mean_dedx, df_electrons,'pTPC', 'dedx')
fill_hist_2d(h_proton_mean_clus, df_protons,'pTPC', 'clSizeCosLam')
fill_hist_2d(h_pion_mean_clus, df_pions,'pTPC', 'clSizeCosLam')
fill_hist_2d(h_electron_mean_clus, df_electrons,'pTPC', 'clSizeCosLam')

outFile = ROOT.TFile(f'../results/test_v0_builder_{run_num}.root', 'RECREATE')
h_lam_mass.Write()
h_k0s_mass.Write()
h_phot_mass.Write()
h_th2_mass.Write()
h_th2_armenteros.Write()


h_proton_mean_clus.Write()
h_pion_mean_clus.Write()
h_electron_mean_clus.Write()
h_proton_mean_dedx.Write()
h_pion_mean_dedx.Write()
h_electron_mean_dedx.Write()

h_rof_bc.Write()
outFile.Close()

