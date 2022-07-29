import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('PDF')

df_V0 = uproot.open('../macros/V0TreePIDITS.root')["V0Tree"].arrays(library='pd')
df_dau = uproot.open('../macros/V0TreePIDITS.root')["DauTree"].arrays(library='pd')

## selecting lambda daughters
v0_cuts_alpha_pos = 'V0ArmenterosAlpha > 0 and V0CosPA>0.9995 and abs(lamMassHyp-1.116)<0.01'
v0_cuts_alpha_neg = 'V0ArmenterosAlpha < 0 and V0CosPA>0.9995 and abs(lamMassHyp-1.116)<0.01'

filter_V0_alpha_pos = df_V0.query(v0_cuts_alpha_pos)
filter_V0_alpha_neg = df_V0.query(v0_cuts_alpha_neg)


## pass filter to df_dau
filter_dau_alpha_pos = df_dau.v0Ind.isin(filter_V0_alpha_pos['v0Ind'])
filter_dau_alpha_neg = df_dau.v0Ind.isin(filter_V0_alpha_neg['v0Ind'])


## adding Armenteros Alpha to daughter tree
df_dau_protons = pd.concat([df_dau[filter_dau_alpha_pos].query('isPositive==1'), df_dau[filter_dau_alpha_neg].query('isPositive==0')], ignore_index=True)
df_dau_pi = pd.concat([df_dau[filter_dau_alpha_pos].query('isPositive==0'), df_dau[filter_dau_alpha_neg].query('isPositive==1')], ignore_index=True)