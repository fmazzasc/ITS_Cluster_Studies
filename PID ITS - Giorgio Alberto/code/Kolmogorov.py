from unittest import result
from ROOT import TH1F, TFile

results = {}

for i in range(7):
    f1 = TFile(f'/home/galucia/PID_ITS/data_visual_root/V0/no_options/ClSizeL{i}.root')
    f2 = TFile(f'/home/galucia/PID_ITS/data_visual_root/TPC/no_options/ClSizeL{i} copy.root')

    h1 = TH1F("h1", "V0", 50, 0.8, 0.85)
    h1 = f1.Get("2")

    h2 = TH1F("h2", "TPC", 50, 0.8, 0.85)
    h2 = f2.Get("2")

    results[f'ClSizeL{i}'] = h1.KolmogorovTest(h2)

f1 = TFile(f'/home/galucia/PID_ITS/data_visual_root/V0/no_options/meanClsize.root')
f2 = TFile(f'/home/galucia/PID_ITS/data_visual_root/TPC/no_options/meanClsize copy.root')

h1 = TH1F("h1", "V0", 50, 0.8, 0.85)
h1 = f1.Get("2")

h2 = TH1F("h2", "TPC", 50, 0.8, 0.85)
h2 = f2.Get("2")

results[f'meanClSize'] = h1.KolmogorovTest(h2)

print(results)
