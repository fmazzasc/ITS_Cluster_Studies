import pandas as pd
from UsefulFunctions import multiple_hist, MultipleKS


df = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520143_itstpc.parquet')

mass_P =  0.93827200
mass_K = 0.493677
mass_Pi = 0.13957039

masses = {1: mass_P, 2: mass_K, 3: mass_Pi}

df.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
df.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)

df.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)

dfs = [df.query(f'particle == {i+1}') for i in range(3)]
for i, df in enumerate(dfs):  
    df.eval(f'beta = pTPC / sqrt(pTPC**2 + {masses[i+1]})', inplace=True)
    #df.query('0.85 <= beta < 0.87', inplace=True)
    df.query('0.4 <= p < 0.5', inplace=True)

columns = [f'ClSizeL{i}' for i in range(7)]
columns.append('clSizeCosLam')

plt_spec = [[10, 0, 10] for i in range(7)]
plt_spec.append([100, 0, 10])

dict = dict(zip(columns, plt_spec))

for col, spec in dict.items(): multiple_hist(dfs, col, spec, '/home/galucia/PID_ITS/ITS_Cluster_Studies/PID ITS - Giorgio Alberto/code/Check_Hist/P_04_05_')
for col, spec in dict.items(): MultipleKS(dfs, col, spec, '/home/galucia/PID_ITS/ITS_Cluster_Studies/PID ITS - Giorgio Alberto/code/Check_Hist/KS_P_04_05_')

