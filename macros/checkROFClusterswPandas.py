#
# scirpt to check for repeated valueas in cluster rof
# run: python3 checkROFClusterswPandas.py
#
import pandas as pd
import numpy as np

df = pd.read_csv("checkROFCluster_run505645.txt", header=0, delimiter=" ")
dupl_arr = np.asarray(df.duplicated(subset=['clus.getCol()', 'clus.getRow()', 'Npixels', 'TF'], keep=False))

print(len(df['rofInd+1'][dupl_arr]))
print(len(df['rofInd+1']))
print(df[dupl_arr].head(10))
