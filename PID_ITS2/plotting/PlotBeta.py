import sys
sys.path.append('..')
from src.preprocessing import *
from src.dataVisual import DataVisual, DataVisualSlow
from src.loadData import LoadData
from utils.particles import particleMasses, particlePDG

sys.path.append('../..')
from utils.StyleFormatter import SetGlobalStyle, SetObjectStyle, SetXsystForLogScale, LatLabel, kDrays, kHInelastic, kPrimary, LineAtOne, SetLegendStyle, kAzureCool, kDplusPrompt, kRed, kGreen, kOrange

from ROOT import TFile, TH1D, TH2D, TCanvas, TLegend, TLatex, gStyle, TPad, TF1
from ROOT import kRed, kAzure, kGreen, kOrange
import yaml

import polars as pl


SetGlobalStyle(padbottommargin=0.14, padleftmargin=0.14, padrightmargin=0.15,
               padtopmargin=0.05, titleoffsety=0.9, titleoffsetx=0.9, titleoffsetz=1., 
               maxdigits=2, palette=53, titlesizex=0.05, titlesizey=0.05, labelsize=0.03)


def fillTH1(hist, df, var):
    for varVal in df[var]:  hist.Fill(varVal)

def fillTH2(hist, df, var1, var2):
    for var1Val, var2Val in zip(df[var1], df[var2]):  hist.Fill(var1Val, var2Val)




def drawBetaWithSubplot(inputFile, outputFile):

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    gStyle.SetPalette(53)

    # set the general canvas
    cBeta = TCanvas('cBeta', '', 900, 800)
    cBeta.cd().SetLogz()
    hFrame = cBeta.cd().DrawFrame(-0.01, -0.1, 1., 1.37, '; #it{p} (GeV/#it{c});#beta_{ML}')

    hFrame.GetYaxis().SetDecimals()
    hFrame.GetXaxis().SetDecimals()
    hFrame.GetZaxis().SetDecimals()
    hFrame.GetYaxis().SetTitleSize(0.07)
    hFrame.GetXaxis().SetTitleSize(0.07)
    hFrame.GetZaxis().SetTitleSize(0.05)
    hFrame.GetYaxis().SetLabelSize(0.04)
    hFrame.GetXaxis().SetLabelSize(0.04)
    hFrame.GetZaxis().SetLabelSize(0.04)

    zaxis = TLatex()
    zaxis.SetTextAlign(22)
    zaxis.SetTextAngle(90)
    zaxis.SetTextSize(0.07)
    zaxis.SetTextFont(42)
    zaxis.DrawLatexNDC(0.95, 0.5, 'Counts')

    hBeta = inputFile.Get('application/betaML_vs_p')

    cBeta.cd()
    cBeta.Update()

    hBeta.Draw('colz same')

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.2, 0.89, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.2, 0.84, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')

    cBeta.SaveAs(outputFile)

    # theoretical curves

    theoreticalCurves = []
    for name, m, col in zip(['pion','Kaon', 'proton'], [0.139, 0.493, 0.938], [kAzureCool, kGreen-2, kRed+1]):
        xmin = 0.1
        if name == 'pion':  xmin = 0.08
        elif name == 'Kaon': xmin = 0.16
        elif name == 'proton': xmin = 0.25
        f = TF1(f'f_{name}', f'x/sqrt(x*x+{m**2})', xmin, 1.)
        f.SetLineColor(col)
        f.SetLineWidth(3)
        
        theoreticalCurves.append(f)
    
    cBeta.cd()
    for f in theoreticalCurves:  f.Draw('same')

    cBeta.SaveAs(outputFile.replace('.pdf', '_curves.pdf'))

    # set the subplot
    x_min = 0.3
    x_max = 0.4
    projection = hBeta.ProjectionY("projection", hBeta.GetXaxis().FindBin(x_min), hBeta.GetXaxis().FindBin(x_max))
    projection.SetTitle(';#beta_{ML}; Counts')
    projection.Rebin(8)
    projection.SetFillColorAlpha(kOrange-3, 0.3)
    projection.SetLineColor(kOrange-3)
    projection.SetFillStyle(3356)

    sub_pad = TPad("sub_pad", "", 0.52, 0.17, 0.83, 0.45)
    sub_pad.SetLogy()
    sub_pad.SetTopMargin(0.02)
    sub_pad.SetBottomMargin(0.17)
    sub_pad.SetRightMargin(0.02)
    sub_pad.SetLeftMargin(0.17)
    sub_pad.cd()
    subFrame = sub_pad.DrawFrame(0., 1, 1.1, 5e4, ";#beta_{ML}; Counts")
    
    cBeta.cd()
    sub_pad.Draw()
    sub_pad.cd()
    subFrame.GetYaxis().SetTitleSize(0.08)
    subFrame.GetXaxis().SetTitleSize(0.08)
    subFrame.GetYaxis().SetLabelSize(0.05)
    subFrame.GetXaxis().SetLabelSize(0.05)
    projection.Draw('same hist')

    sublatex = TLatex()
    sublatex.SetNDC()
    sublatex.SetTextFont(42)    
    sublatex.SetTextSize(0.08)
    sublatex.DrawLatex(0.2, 0.75, f'{x_min} < #it{{p}} < {x_max} GeV/#it{{c}}')

    cBeta.SaveAs(outputFile.replace('.pdf', '_sub.pdf'))

    



if __name__ == '__main__':

    inputFile = TFile.Open('../data/ITS-TPC/application_PandSpecies_XGB.root')
    outputFile = '../plots/WP2/MLBeta.pdf'
    drawBetaWithSubplot(inputFile, outputFile)
    inputFile.Close()