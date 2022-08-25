from unittest import result
from math import ceil, sqrt
import pandas as pd
from ROOT import TH1F, TCanvas, kCyan, kRed, TLegend
import numpy as np

results_hist = {}
results_graph = {}
factor = 1.


leg = TLegend(0.75, 0.95, 0.75, 0.95)
leg.SetFillStyle(0)
leg.SetTextSize(0.045)
leg.SetNColumns(1)

d1 = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520143.parquet')
d2 = pd.read_parquet('/home/galucia/PID_ITS/data/TPC/Df_filtered_ITS2Cluster505673.parquet.gzip')

d1.eval('meanClsize = (ClSizeL0 + ClSizeL1 + ClSizeL2 + ClSizeL3 + ClSizeL4 + ClSizeL5 + ClSizeL6)/7', inplace=True)
d2.eval('meanClsize = (ClSizeL0 + ClSizeL1 + ClSizeL2 + ClSizeL3 + ClSizeL4 + ClSizeL5 + ClSizeL6)/7', inplace=True)

d1.query('0.85 <= p < 0.9', inplace=True)
d2.query('0.85 <= p < 0.9', inplace=True)

canvas = TCanvas('canvas', 'canvas', 900, 1200)
canvas.SetTitle('Kolmogorov Test')
canvas.Divide(2, 4)


h_1 = [TH1F(f"h{2*i+1}", "V0", 10, 0, 10) for i in range(7)]
h_2 = [TH1F(f"h{2*i+2}", "TPC", 10, 0, 10) for i in range(7)]

for i, (h1, h2) in enumerate(zip(h_1, h_2)):

    leg.Clear()
    
    for n in d1[f'ClSizeL{i}']:     h1.Fill(n)
    h1.Scale(factor/h1.Integral(), 'width') 
    h1.SetError( np.array([ sqrt(h1.GetEntries()) for i in range( int(h1.GetEntries()) )], dtype='float') )
    h1.SetLineColor(kCyan)
    h1.SetMaximum(1)
    leg.AddEntry(h1, 'V0', 'l')

    for n in d2[f'ClSizeL{i}']:     h2.Fill(n)
    h2.Scale(factor/h2.Integral(), 'width')
    h2.SetError( np.array([ sqrt(h2.GetEntries()) for i in range( int(h2.GetEntries()) ) ], dtype='float') )
    h2.SetLineColor(kRed)
    h2.SetMaximum(1)
    leg.AddEntry(h2, 'TPC', 'l')

    results_hist[f'ClSizeL{i}'] = h1.KolmogorovTest(h2)
    h1.SetTitle(f'ClSizeL{i}: {results_hist[f"ClSizeL{i}"]}')

    canvas.cd(i+1)
    h1.Draw('hist')
    h2.Draw('hist same')
    leg.Draw()




leg.Clear()

h1 = TH1F("h15", "V0", 100, 0, 10)
for n in d1['meanClsize']:     h1.Fill(n)
h1.Scale(factor/h1.Integral(), 'width')
h1.SetError( np.array([ sqrt(h1.GetEntries()) for i in range( int(h1.GetEntries()) )], dtype='float') )
h1.SetLineColor(kCyan)
h1.SetMaximum(1)
leg.AddEntry(h1, 'V0', 'l')


h2 = TH1F("h16", "TPC", 100, 0, 10)
for n in d2['meanClsize']:     h2.Fill(n)
h2.Scale(factor/h2.Integral(), 'width')
h2.SetError( np.array([ sqrt(h2.GetEntries()) for i in range( int(h2.GetEntries()) ) ], dtype='float') )
h2.SetLineColor(kRed)
h2.SetMaximum(1)
leg.AddEntry(h2, 'TPC', 'l')

results_hist[f'meanClsize'] = h1.KolmogorovTest(h2)
h1.SetTitle(f'meanClsize: {results_hist["meanClsize"]}')

canvas.cd(8)
h1.Draw('hist')
h2.Draw('hist same')
leg.Draw()

canvas.SaveAs('/home/galucia/PID_ITS/ITS_Cluster_Studies/PID ITS - Giorgio Alberto/code/Kolmogorov1.root')


del h1, h2, canvas

print(results_hist)
