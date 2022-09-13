import pandas as pd
from UsefulFunctions import density_scatter, filtering

mass_Deu = 1.8756
mass_P =  0.93827200
mass_K = 0.4937
mass_Pi = 0.13957000
mass_E = 0.000511

particle_dict = { 1: "P", 2: "K", 3: "Pi", 4: "E"}
mass_dict = dict(zip(particle_dict.values(), [mass_P, mass_K, mass_Pi, mass_E]))

df = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520147.parquet')




density_scatter(df['p'], df['dedx'], "dedx_appl_V0", ["p", "#frac{dE}{dx}", 1500, 0, 1.5, 600, 0, 600])




df.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
df.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)

df.eval('label = particle', inplace=True)
for number, name in particle_dict.items():  df['label'].mask(df['particle'] == number, name, inplace=True)

appl_list = [filtering(df, name, mass=mass_dict[name], label=False)for name in particle_dict.values()]
#df = pd.concat(appl_list)

density_scatter(appl_list[1]['p'], appl_list[1]['dedx'], "dedx_appl_filt_V0", ["p", "#frac{dE}{dx}", 1500, 0, 1.5, 600, 0, 600])