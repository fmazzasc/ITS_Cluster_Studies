from enum import Enum
import ROOT
ROOT.gROOT.SetBatch()

run_number = 520271
run_period = 'JUL'
build_trees = False


def fill_hist(hist, dataframe, column):
    for val in dataframe[column]:
        hist.Fill(val)
    return hist

def fill_hist_2d(hist, dataframe, column1, column2):
    for val1, val2 in zip(dataframe[column1], dataframe[column2]):
        hist.Fill(val1, val2)
    return hist

class kParticle(Enum):
    proton = 1
    kaon = 2
    pion = 3
    electron = 4

## few output histograms
h_proton_mean_clus = ROOT.TH2F('h_proton_mean_clus', 'h_proton_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_proton_mean_dedx = ROOT.TH2F('h_proton_mean_dedx', 'h_proton_mean_dedx',100, 0, 2, 100, 30, 400)

h_kaon_mean_clus = ROOT.TH2F('h_kaon_mean_clus', 'h_kaon_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_kaon_mean_dedx = ROOT.TH2F('h_kaon_mean_dedx', 'h_kaon_mean_dedx',100, 0, 2, 100, 30, 400)

h_pion_mean_clus = ROOT.TH2F('h_pion_mean_clus', 'h_pion_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_pion_mean_dedx = ROOT.TH2F('h_pion_mean_dedx', 'h_pion_mean_dedx',100, 0, 2, 100, 30, 400)

h_electron_mean_clus = ROOT.TH2F('h_electron_mean_clus', 'h_electron_mean_clus',10, 0, 2, 50, 0.5, 10.5)
h_electron_mean_dedx = ROOT.TH2F('h_electron_mean_dedx', 'h_electron_mean_dedx',100, 0, 2, 100, 30, 400)


h_phot_mass = ROOT.TH1F('h_phot_mass', 'h_phot_mass; M_{#gamma} (GeV/{c^{2}}); Counts' , 100, 0., 0.001)
h_k0s_mass = ROOT.TH1F('h_k0s_mass', 'h_k0s_mass; M_{k0_{s}} (GeV/{c^{2}}); Counts', 100, 0.45, 0.55)
h_lam_mass = ROOT.TH1F('h_lam_mass', 'h_v0_mass; M_{#Lambda} (GeV/{c^{2}}); Counts', 100, 1.08, 1.15)

#-----------------------------------------------------------------------------------------------------------------------





if build_trees:
    # ROOT part, generate trees for primary and secondary particles
    ROOT.gROOT.LoadMacro("../macros/TreeBuilders.C++")
    from ROOT import ITSTPCClusterTreeBuilder, V0ClusterTreeBuilder
    ITSTPCClusterTreeBuilder(run_number, run_period)
    V0ClusterTreeBuilder(run_number, run_period)

else:

    import numpy as np
    import pandas as pd
    import uproot
    import matplotlib.pyplot as plt
    import matplotlib

    ## get ITS-TPC and V0 trees
    df_primaries = uproot.open(f'../results/ITSTPCClusterTree{run_number}.root')["ITStreeML"].arrays(library='pd')
    df_V0 = uproot.open(f'../results/V0TreePIDITS_{run_number}.root')["V0Tree"].arrays(library='pd')
    df_V0.query('V0CosPA>0.995', inplace=True)
    df_dau = uproot.open(f'../results/V0TreePIDITS_{run_number}.root')["DauTree"].arrays(library='pd')

    ## protons: primaries for p<0.8, from Lambdas for p>0.8
    print("Assembling protons...")
    df_proton_prim = df_primaries.query('abs(nSigmaP)<1 and p<0.8')
    lambda_cuts_pos = 'V0ArmenterosAlpha > 0.5 and (k0sMassHyp<0.47 or k0sMassHyp>0.52) and 1.09<lamMassHyp<1.14 and (abs(nSigmaPosDauP)<3 or abs(nSigmaNegDauP)<3) and V0p>0.8'
    lambda_cuts_neg = 'V0ArmenterosAlpha < 0.5 and (k0sMassHyp<0.47 or k0sMassHyp>0.52) and 1.09<lamMassHyp<1.14 and (abs(nSigmaPosDauP)<3 or abs(nSigmaNegDauP)<3) and V0p>0.8'

    filter_lambda_pos = df_V0.query(lambda_cuts_pos)
    filter_lambda_neg = df_V0.query(lambda_cuts_neg)
    df_proton_sec_pos = df_dau[df_dau.v0Ind.isin(filter_lambda_pos['v0Ind'])].query('isPositive==1')
    df_proton_sec_neg = df_dau[df_dau.v0Ind.isin(filter_lambda_neg['v0Ind'])].query('isPositive==0')

    # filtered_lambdas = pd.concat([filter_lambda_pos, filter_lambda_neg])
    # print(filtered_lambdas[['V0DauPosSource', 'V0DauNegSource', 'nSigmaPosDauP', 'nSigmaNegDauP']])

    df_proton_sec = pd.concat([df_proton_sec_pos, df_proton_sec_neg])
    df_proton_sec.reset_index(drop=True, inplace=True)
    fill_hist(h_lam_mass, pd.concat([filter_lambda_pos, filter_lambda_neg]), 'lamMassHyp')
    print("PROTONS FROM V0: ", len(df_proton_sec))
    df_proton_sec.query('abs(nSigmaP)<3', inplace=True)
    df_protons = pd.concat([df_proton_prim, df_proton_sec])
    df_protons.reset_index(drop=True, inplace=True)
    df_protons['particle'] = kParticle.proton.value
    fill_hist_2d(h_proton_mean_clus, df_protons,'pTPC', 'clSizeCosLam')
    fill_hist_2d(h_proton_mean_dedx, df_protons,'pTPC', 'dedx')

    ## kaons: only primaries for p<0.6
    print("Assembling kaons...")
    df_kaon_prim = df_primaries.query('abs(nSigmaK)<1 and tpcITSchi2<10 and p<0.6')
    df_kaons = df_kaon_prim
    df_kaons['particle'] = kParticle.kaon.value
    fill_hist_2d(h_kaon_mean_clus, df_kaons,'pTPC', 'clSizeCosLam')
    fill_hist_2d(h_kaon_mean_dedx, df_kaons,'pTPC', 'dedx')

    ## pions: from K0s only
    print("Assembling pions...")
    k0s_cuts = 'abs(V0ArmenterosAlpha) < 0.8 and V0ArmenterosQt > 0.12 and (lamMassHyp<1.10 or lamMassHyp>1.13) and 0.47<k0sMassHyp<0.52'
    filter_k0s = df_V0.query(k0s_cuts)
    df_pions = df_dau[df_dau.v0Ind.isin(filter_k0s['v0Ind'])]
    df_pions.query('abs(nSigmaPi)<3', inplace=True)
    df_pions['particle'] = kParticle.pion.value
    fill_hist(h_k0s_mass, filter_k0s, 'k0sMassHyp')
    fill_hist_2d(h_pion_mean_clus, df_pions,'pTPC', 'clSizeCosLam')
    fill_hist_2d(h_pion_mean_dedx, df_pions,'pTPC', 'dedx')


    ## electrons: from photon conversion only
    print("Assembling electrons...")
    phot_conversion_cuts = 'abs(V0ArmenterosAlpha)<0.3 and V0ArmenterosQt < 0.01 and photMassHyp < 4e-5'
    filter_phot_conversion = df_V0.query(phot_conversion_cuts)
    df_electrons = df_dau[df_dau.v0Ind.isin(filter_phot_conversion['v0Ind'])]
    df_electrons.query('dedx>60', inplace=True)
    df_electrons['particle'] = kParticle.electron.value
    fill_hist_2d(h_electron_mean_clus, df_electrons,'pTPC', 'clSizeCosLam')
    fill_hist_2d(h_electron_mean_dedx, df_electrons,'pTPC', 'dedx')


    ## merge all particles into one dataframe
    df_particles = pd.concat([df_protons, df_kaons, df_pions, df_electrons])
    df_particles.reset_index(drop=True, inplace=True)
    #dump into parquet
    df_particles.to_parquet(f'../results/particles_pid_{run_number}.parquet')


    outFile = ROOT.TFile(f'../results/particles_pid_{run_number}.root', 'RECREATE')
    h_proton_mean_clus.Write()
    h_proton_mean_dedx.Write()
    h_kaon_mean_clus.Write()
    h_kaon_mean_dedx.Write()
    h_pion_mean_clus.Write()
    h_pion_mean_dedx.Write()
    h_electron_mean_clus.Write()
    h_electron_mean_dedx.Write()
    h_lam_mass.Write()
    h_k0s_mass.Write()
    outFile.Close()