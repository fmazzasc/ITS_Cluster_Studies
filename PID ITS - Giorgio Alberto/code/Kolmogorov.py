import pandas as pd

from ROOT import TH1F, TCanvas, kCyan, kRed, TLegend, gPad, TFile

from UsefulFunctions import KolmogorovHist




def selection(df, particle, tag):
    """
    From a full datatframe, creates a new one saving only data relative to a chosen particle (filtering  with instructions in its tag).
    The new dataframe will have a label column where its particle species is specified.

    Parameters:
    - df: full dataframe
    - particle: name of the particle to filter
    - tag: instructions to filter

    Returns:
    a filtered df
    """
    
    df = df.query(tag, inplace=False).reset_index(drop=True)
    df['label'] = particle
    return df

def main(d1, d2, output):
    
    results = {}

    canvas = TCanvas('canvas', 'canvas', 900, 1200)
    canvas.SetTitle('Kolmogorov Test')
    canvas.Divide(2, 4)

    leg = TLegend(0.65, 0.8, 0.75, 0.95)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.045)
    leg.SetNColumns(1)

    h_1 = [TH1F(f"h{2*i+1}", "V0", 100, -10, 10) for i in range(7)]
    h_2 = [TH1F(f"h{2*i+2}", "TPC", 100, -10, 10) for i in range(7)]

    #for i in range(7):
    #   
    #    d1.drop( d1[d1[f'ClSizeL{i}'] < 0].index, inplace=True )
    #    d2.drop( d2[d2[f'ClSizeL{i}'] < 0].index, inplace=True )

    for i, (h1, h2) in enumerate(zip(h_1, h_2)):

        name = f'TanLamL{i}'
        for n in d1[name]:     h1.Fill(n)
        for n in d2[name]:     h2.Fill(n)
        

        h1_empty, h2_empty = False, False
        if h1.GetEntries() == 0:    
            h1_empty = True
            print('First histogram is empty')
        if h2.GetEntries() == 0:    
            h2_empty = True
            print('Second histogram is empty')
        if h1_empty or h2_empty:    continue
        else:                       results[name] = KolmogorovHist(h1, h2, canvas=canvas, pad=i+1, legend=leg, name=name)


    name = 'tgL'
    h1 = TH1F("h15", "V0", 10, -1, 1)
    for n in d1[name]:     h1.Fill(n)
    h2 = TH1F("h16", "TPC", 10, -1, 1)
    for n in d2[name]:     h2.Fill(n)
    
    
    if h1.GetEntries() != 0 and h2.GetEntries() != 0:   results[name] = KolmogorovHist(h1, h2, canvas=canvas, pad=8, legend=leg, name=name)
    canvas.SaveAs(f'{output}.root')
    canvas.SaveAs(f'{output}.png')

    del h_1, h_2, h1, h2, canvas
    print(results)












particle_dict = {'P': 'nSigmaPAbs < 1',
                'K': 'nSigmaKAbs < 1',
                'Pi': 'nSigmaPiAbs < 1',
                'E':  ''
                }

d1 = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520143.parquet')
d2 = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520147_itstpc.parquet')

d1.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
d1.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)
d2.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
d2.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)

d1.eval('meanClsize = (ClSizeL0 + ClSizeL1 + ClSizeL2 + ClSizeL3 + ClSizeL4 + ClSizeL5 + ClSizeL6)/7', inplace=True)
d2.eval('meanClsize = (ClSizeL0 + ClSizeL1 + ClSizeL2 + ClSizeL3 + ClSizeL4 + ClSizeL5 + ClSizeL6)/7', inplace=True)

d1.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
d2.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)

d1.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
d2.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)

for key in particle_dict.keys():    d1.eval(f'nSigma{key}Abs = abs(nSigma{key})', inplace=True)
for key in particle_dict.keys():    d2.eval(f'nSigma{key}Abs = abs(nSigma{key})', inplace=True)

d1.loc[d1.nSigmaKAbs < 1, 'particle'] = 2

d1.drop( d1[d1.particle == 4].index, inplace=True )
d2.drop( d2[d2.nSigmaEAbs < 1].index, inplace=True )



d1.query('0.8 <= p < 0.85', inplace=True)
d2.query('0.8 <= p < 0.85', inplace=True)

# Kolmogorov on full dfs
output = '/home/galucia/PID_ITS/ITS_Cluster_Studies/PID ITS - Giorgio Alberto/code/Kolmogorov/TanLam_Kolmogorov_08_085'
print()
main(d1, d2, output=output)
print()


for i, (particle, tag) in zip(range(3), particle_dict.items()):

    new_d1 = pd.DataFrame(d1.query(f'particle == {i+1}'))
    new_d2 = pd.DataFrame(selection(d2, particle=particle, tag=tag))

    output = f'/home/galucia/PID_ITS/ITS_Cluster_Studies/PID ITS - Giorgio Alberto/code/Kolmogorov/TanLam_Kolmogorov_{particle}_08_085'
    print(f'{particle}')
    print()
    main(new_d1, new_d2, output=output)
    print()







