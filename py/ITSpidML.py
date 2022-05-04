'''
python script for the projection of D+, Ds+ and Lc particles TTrees
run: python ITSpidML.py
'''

import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import uproot
from ROOT import TFile, TH2F, TH1F, TCanvas, TLegend, TGraph, kRed, kBlue, TGaxis,gPad, TLatex # pylint: disable=import-error,no-name-in-module

inputMLmodelpath = '/home/spolitan/Analyses/Stefano_Analysis/ITS2pidML/training/ModelApplied.parquet.gzip'

df, df_mazz = pd.DataFrame(), pd.DataFrame()
df = df.append(pd.read_parquet(inputMLmodelpath), ignore_index=True)
df_mazz = df_mazz.append(pd.read_parquet('/data/shared/ITS/tree_pid.parquet.gzip'), ignore_index=True)

hSoverB = TH1F('hSoverB', ';BDT out; p / #pi', 1000, 0., 1.)
hSoverB.SetStats(0)
hSoverB.SetLineColor(kRed)
hEff = TH1F('hEff', ';BDT out; #epsilon', 1000, 0., 1.)
hEff.SetStats(0)
hRej = TH1F('hRej', ';BDT out; rej.', 1000, 0., 1.)
hRej.SetLineColor(kRed)
hPoverPivsEffbdt = TGraph()
hPoverPivsEffbdt.SetTitle(' ;p efficiency;p/#pi')
hPoverPivsEffbdt.SetLineColor(kRed)
hPoverPivsEffbdt.SetMarkerColor(kRed)
hPoverPivsEffbdt.SetLineWidth(2)
hPoverPivsEffClsie = TGraph()
hPoverPivsEffClsie.SetLineColor(kBlue)
hPoverPivsEffClsie.SetMarkerColor(kBlue)
hPoverPivsEffClsie.SetLineStyle(9)
hPoverPivsEffClsie.SetLineWidth(2)
tot_p = len(df.query("isProton == 1"))
tot_pi = len(df.query("isProton == 0"))

for ibin in range(1, hEff.GetNbinsX()):
    iBDT = hEff.GetBinCenter(ibin)
    df_sel = df.query(f"ML_output > {iBDT}")
    df_pirej = df.query(f"ML_output > {iBDT} and isProton==0")
    nB = len(df_sel.query("isProton == 0"))
    nS = len(df_sel.query("isProton == 1"))
    eff = nS/tot_p
    rej = nB/tot_pi

    if (nB != 0):
        SoverB = nS/nB
    hSoverB.SetBinContent(ibin, SoverB)
    hEff.SetBinContent(ibin, eff)
    hRej.SetBinContent(ibin, rej)

    hPoverPivsEffbdt.AddPoint(eff, SoverB)

tot_p = len(df_mazz.query("isProton == 1"))
tot_pi = len(df_mazz.query("isProton == 0"))
for icut in np.arange(0, 12, 0.012):
    df_mazz_sel =df_mazz.query(f"mean_cos_lam > {icut}")
    eff = len(df_mazz_sel.query("isProton == 1"))/tot_p
    SoverB = len(df_mazz_sel.query("isProton == 1"))/len(df_mazz_sel.query("isProton == 0"))
    hPoverPivsEffClsie.AddPoint(eff, SoverB)


outFile = TFile('BDTsel.root', 'recreate')
hEff.Write()
hSoverB.Write()
hRej.Write()

c1 = TCanvas("c1", "", 600, 600)
c1.cd().SetLogy()
hPoverPivsEffbdt.GetXaxis().SetRangeUser(0.1, 1.)
hPoverPivsEffbdt.Draw()
hPoverPivsEffClsie.Draw("same")
leg = TLegend(0.7,0.7,0.9,0.9)
leg.SetBorderSize(0)
leg.AddEntry(hPoverPivsEffbdt,"BDT out","l")
leg.AddEntry(hPoverPivsEffClsie,"#LTCluster size#GT#timescos(#lambda)","l")
leg.Draw()
c1.Write()
c1.SaveAs("PurityVsEff_BDT_CLsize.pdf")


c = TCanvas("c", "", 600, 600)
c.cd().SetLogy()
hSoverB.DrawCopy("hist")
axis = TGaxis(gPad.GetUxmax(),gPad.GetUymin(), gPad.GetUxmax(), 110, 0.002, 1, 10,"+LG")
axis.SetTitle('#epsilon')

hEff.Scale(110)
hEff.DrawCopy("histsame")
#gPad.SetLog(1)
axis.SetLineColor(kRed)
axis.SetLabelColor(kRed)
axis.Draw()
gPad.DrawClone("same")
c.SaveAs("c.pdf")
c.Write()
outFile.Close()
