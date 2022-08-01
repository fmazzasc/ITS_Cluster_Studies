from ROOT import TH1, TH2, TGraph, TGraphErrors, TLegend, TCanvas, gStyle, gPad, kFALSE
import pandas as pd

# Histograms, TGraphs and TLegend settings
def set_hist_style(hist, x_label, y_label, set_opt_stat='1110', line_color=0, fill_style=0, y_title_offset=1.4, top_margin=0.15, left_margin=0.15, right_margin=0.15, title=kFALSE,):
    
    gStyle.SetOptTitle(title)
    gStyle.SetOptStat(int(set_opt_stat))

    hist.SetXTitle(x_label)
    hist.SetYTitle(y_label)

    hist.SetLineColor(line_color)
    hist.SetFillStyle(fill_style)

    hist.GetYaxis().SetTitleOffset(y_title_offset)
    gPad.SetTopMargin(top_margin)
    gPad.SetLeftMargin(left_margin)
    gPad.SetRightMargin(right_margin)

    # ..other settings



def set_graph_style(graph, x_label, y_label, line_color=0, line_width=4, y_title_offset=1.4, top_margin=0.15, left_margin=0.15, right_margin=0.15, title=kFALSE,):
    
    gStyle.SetOptTitle(title)

    graph.SetXTitle(x_label)
    graph.SetYTitle(y_label)

    graph.SetLineColor(line_color)
    graph.SetLineWidth(line_width)

    graph.GetYaxis().SetTitleOffset(y_title_offset)
    gPad.SetTopMargin(top_margin)
    gPad.SetLeftMargin(left_margin)
    gPad.SetRightMargin(right_margin)


    # ..other settings


def set_legend(legend, **kwargs):
    for hist, name in kwargs:
        legend.AddEntry(hist, name, 'l')



# Fill Histograms with lists or pd.Dataframe columns
def fill_hist(hist, x, y=None):
    
    if (type(y) == pd.Series and y.empty) or (type(y) != pd.Series and y == None): 
        if type(x) == pd.Series:
            x = [i for i in x]

        for i in x:     
            hist.Fill(i)

    else:
        if type(y) == pd.Series:
            y = [i for i in y]

        for i, j in zip(x, y):
                hist.Fill(i, j)



