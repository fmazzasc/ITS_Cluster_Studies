#
#   Functions to perform Kolmogorov-Smirnov tests
#

import numpy as np
from ROOT import TCanvas, TH1F, kCyan, kRed, kGreen, gPad, TLegend


def KolmogorovHist(h1, h2, canvas, pad, legend, name, **kwargs):
    """
    Does a Kolmogorov test for two given histograms, plots the histograms and returns the result

    **kwargs
    - leg_entry1
    - leg_entry2
    """

    factor = 1.
    legend.Clear()
    
    #h1.Scale(factor/h1.Integral(), 'width') 
    h1.SetError( np.array([ sqrt(h1.GetEntries()) for i in range( int(h1.GetEntries()) )], dtype='float') )
    h1.SetLineColor(kGreen)
    #h1.SetMaximum(100)
    #h1.SetMinimum(0.00001)
    if 'leg_entry1' in kwargs:  legend.AddEntry(h1, kwargs['leg_entry1'], 'l')
    else:                       legend.AddEntry(h1, 'V0', 'l')

    #h2.Scale(factor/h2.Integral(), 'width')
    h2.SetError( np.array([ sqrt(h2.GetEntries()) for i in range( int(h2.GetEntries()) ) ], dtype='float') )
    h2.SetLineColor(kRed)
    #h2.SetMaximum(100)
    #h2.SetMinimum(0.00001)
    if 'leg_entry2' in kwargs:  legend.AddEntry(h2, kwargs['leg_entry2'], 'l')
    else:                       legend.AddEntry(h2, 'TPC', 'l')

    KS_result = h1.KolmogorovTest(h2)
    h1.SetTitle(f'KS({name}): {round(KS_result, 3)}')       

    canvas.cd(pad)      
    h1.DrawNormalized('hist e1')
    h2.DrawNormalized('hist same e1')
    gPad.SetLogy()
    legend.Draw()
    canvas.Draw()

    return KS_result

def MultipleKS(dfs, column, hist_specifics, filename, leg_options=[]):
    """
    Saves multiple histigrams (different particle species, same variable) on the same file. You can reiterate for different variables as well.
    
    Parameters
    --------------------------------------------
    - dfs: list of dataframes whose columns will fill the histograms (it also works with pd.Series)
    - column: variable you want to draw histograms of
    - hist_specific: list with the following content -> [nbinsx, xlow, xup] 
    - leg_options: [xlow, xup, ylow, yup]

    You could also pass a list of lists as dfs. If that is the case, please input hist_names and x_label. YOu should anyway pass a nonempty list for columns. Any single elements inside of it will be fine
    - hist_names: list of names for each of the histograms
    - x_label: label of the x axis 
    """
    
    nbinsx, xlow, xup = hist_specifics
    results = []

    if leg_options == []:   leg = TLegend(0.65, 0.8, 0.75, 0.95)
    else:                   leg = TLegend(leg_options[0], leg_options[1], leg_options[2], leg_options[3])
    leg.SetFillStyle(0)
    leg.SetTextSize(0.045)
    leg.SetNColumns(1)



    hists = [TH1F(f"h{i+1}", "", nbinsx, xlow, xup) for i in range(len(dfs))]
    for histo, df in zip(hists, dfs):    
        for x in df[column]:    histo.Fill(x)



    for i, h in enumerate(combinations(hists, 2)):    

        canvas = TCanvas('canvas', 'canvas', 900, 1200)

        h0_empty, h1_empty = False, False
        if h[0].GetEntries() == 0:    
            h0_empty = True
            print('First histogram is empty')
        if h[1].GetEntries() == 0:    
            h1_empty = True
            print('Second histogram is empty')
        if h0_empty or h1_empty:    continue
        else:                       results.append(KolmogorovHist(h[0], h[1], canvas=canvas, pad=1, legend=leg, name=column, leg_entry1=f'{h[0].GetName()}', leg_entry2=f'{h[1].GetName()}'))

        canvas.SaveAs(f'{filename}_{column}_{i}.root')

        del canvas
    
    del hists, leg