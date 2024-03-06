import sys
sys.path.append('..')
from src.preprocessing import *
from src.dataVisual import DataVisual, DataVisualSlow
from src.loadData import LoadData

from utils.particles import particleMasses, particlePDG

sys.path.append('../..')
from utils.StyleFormatter import SetGlobalStyle, SetObjectStyle, SetXsystForLogScale, LatLabel, kDrays, kHInelastic, kPrimary, LineAtOne, SetLegendStyle, kAzureCool, kDplusPrompt, kRed, kGreen, kOrange

from ROOT import TFile, TH1D, TH2D, TCanvas, TLegend, TLatex, gStyle, TPad, TF1, TGaxis
from ROOT import kRed, kAzure, kGreen, kOrange
import yaml

import polars as pl


SetGlobalStyle(padbottommargin=0.14, padleftmargin=0.14, padrightmargin=0.17,
               padtopmargin=0.05, titleoffsety=0.9, titleoffsetx=0.9, titleoffsetz=1.4, 
               maxdigits=2, palette=53, titlesizex=0.05, titlesizey=0.05, labelsize=0.03)
TGaxis.SetMaxDigits(2)

def fillTH1(hist, df, var):
    for varVal in df[var]:  hist.Fill(varVal)

def fillTH2(hist, df, var1, var2):
    for var1Val, var2Val in zip(df[var1], df[var2]):  hist.Fill(var1Val, var2Val)



def drawClSizeVsP(inputData, outputFile):

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)

    gStyle.SetPalette(55)

    histClsP = TH2D('clusterSizeVsP', r'; #it{p} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos#lambda #GT', 1500, 0, 1.5, 250, 0, 25)
    histClsP.SetTitle(r'; #it{p}^{ITS-TPC} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos(#lambda) #GT')
    fillTH2(histClsP, inputData, 'p', 'clSizeCosL')

    canvas = TCanvas('canvas', 'canvas', 1000, 800)
    hframe = canvas.DrawFrame(0.15, 0, 1, 9, r'; #it{p} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos(#lambda) #GT; Counts')

    zaxis = TLatex()
    zaxis.SetTextAlign(22)
    zaxis.SetTextAngle(90)
    zaxis.SetTextSize(0.07)
    zaxis.SetTextFont(42)
    zaxis.DrawLatexNDC(0.95, 0.5, 'Counts')

    hframe.GetYaxis().SetDecimals()
    hframe.GetXaxis().SetDecimals()
    hframe.GetZaxis().SetDecimals()
    hframe.GetYaxis().SetTitleSize(0.07)
    hframe.GetXaxis().SetTitleSize(0.07)
    hframe.GetZaxis().SetTitleSize(0.05)
    hframe.GetYaxis().SetLabelSize(0.04)
    hframe.GetXaxis().SetLabelSize(0.04)
    hframe.GetZaxis().SetLabelSize(0.04)

    histClsP.Draw('colz same')
    canvas.SetLogz()

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.2, 0.89, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.2, 0.84, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')

    canvas.SaveAs(outputFile)

def drawClSizeVsPSub(inputData, outputFile):

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)

    gStyle.SetPalette(55)

    histClsP = TH2D('clusterSizeVsP', r'; #it{p} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos(#lambda) #GT', 1500, 0, 1.5, 250, 0, 25)
    histClsP.SetTitle(r'; #it{p}^{ITS-TPC} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos(#lambda) #GT')
    fillTH2(histClsP, inputData, 'p', 'clSizeCosL')

    canvas = TCanvas('canvas2', 'canvas', 1000, 800)
    hframe = canvas.DrawFrame(0.1, -4.5, 1, 8, r'; #it{p} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos(#lambda) #GT; Counts')

    hframe.GetYaxis().SetDecimals()
    hframe.GetXaxis().SetDecimals()
    hframe.GetZaxis().SetDecimals()
    hframe.GetYaxis().SetTitleSize(0.07)
    hframe.GetXaxis().SetTitleSize(0.07)
    hframe.GetZaxis().SetTitleSize(0.05)
    hframe.GetYaxis().SetLabelSize(0.06)
    hframe.GetXaxis().SetLabelSize(0.06)
    hframe.GetZaxis().SetLabelSize(0.06)

    zaxis = TLatex()
    zaxis.SetTextAlign(22)
    zaxis.SetTextAngle(90)
    zaxis.SetTextSize(0.07)
    zaxis.SetTextFont(42)
    zaxis.DrawLatexNDC(0.95, 0.5, 'Counts')

    histClsP.Draw('colz same')
    canvas.SetLogz()

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.18, 0.29, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.18, 0.24, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')

    # add subplot

    # set the subplot
    x_min = 0.3
    x_max = 0.4
    projection = histClsP.ProjectionY("projection", histClsP.GetXaxis().FindBin(x_min), histClsP.GetXaxis().FindBin(x_max))
    projection.Rebin(2)
    projection.SetTitle(';#LT Cluster size #GT #times #LT cos(#lambda) #GT; Counts')
    projection.SetFillColorAlpha(kOrange-3, 0.3)
    projection.SetLineColor(kOrange-3)
    projection.SetFillStyle(3356)

    sub_pad = TPad("sub_pad", "", 0.54, 0.17, 0.83, 0.44)
    sub_pad.SetTopMargin(0.02)
    sub_pad.SetBottomMargin(0.2)
    sub_pad.SetRightMargin(0.02)
    sub_pad.SetLeftMargin(0.17)
    sub_pad.cd()
    subFrame = sub_pad.DrawFrame(0., 0, 8, 65e3, ";#LT Cluster size #GT #times #LT cos(#lambda) #GT; Counts")
    
    canvas.cd()
    sub_pad.Draw()
    sub_pad.cd()
    subFrame.GetYaxis().SetDecimals()
    subFrame.GetXaxis().SetDecimals()
    subFrame.GetYaxis().SetTitleSize(0.1)
    subFrame.GetXaxis().SetTitleSize(0.1)
    subFrame.GetYaxis().SetLabelSize(0.06)
    subFrame.GetXaxis().SetLabelSize(0.06)
    projection.Draw('same hist')

    sublatex = TLatex()
    sublatex.SetNDC()
    sublatex.SetTextFont(42)    
    sublatex.SetTextSize(0.08)
    sublatex.DrawLatex(0.45, 0.6, f'{x_min} < #it{{p}} < {x_max} GeV/#it{{c}}')

    canvas.SaveAs(outputFile)

def drawClSizeSelection(inputData, outputFile, pmin, pmax):

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)

    histClSizeSelected = TH1D('clusterSizeSelected', r'; #LT Cluster size #GT #times #LT cos(#lambda) #GT', 125, 0, 25)
    data = inputData.filter( (pl.col('p') > pmin) & (pl.col('p') < pmax) )
    fillTH1(histClSizeSelected, data, 'clSizeCosL')
    histClSizeSelected.SetTitle(r'; #LT Cluster size #GT #times #LT cos#lambda #GT;Counts')

    canvas = TCanvas('canvas', 'canvas', 1000, 800)
    canvas.SetLeftMargin(0.15)
    hframe = canvas.DrawFrame(0, 0, 5, .8e5, r'; #LT Cluster size #GT #times #LT cos(#lambda) #GT; Counts')
    
    #hframe.GetYaxis().SetDecimals()
    #hframe.GetXaxis().SetDecimals()
    hframe.GetXaxis().SetMaxDigits(1)
    hframe.GetYaxis().SetMaxDigits(1)
    hframe.GetYaxis().SetTitleSize(0.07)
    hframe.GetXaxis().SetTitleSize(0.07)
    hframe.GetYaxis().SetLabelSize(0.04)
    hframe.GetXaxis().SetLabelSize(0.04)
    
    histClSizeSelected.SetFillColorAlpha(kAzure+4, 0.5)
    histClSizeSelected.SetLineColor(kAzure+4)
    histClSizeSelected.SetFillStyle(3356)

    histClSizeSelected.GetXaxis().SetMaxDigits(2)
    histClSizeSelected.GetYaxis().SetMaxDigits(2)
    histClSizeSelected.Draw('hist same')

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.48, 0.75, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.48, 0.7, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')
    latex.DrawLatex(0.48, 0.65, f'{pmin} < #it{{p}} < {pmax} GeV/#it{{c}}')
    
    canvas.SaveAs(outputFile)

def drawClSizeSpecies(inputData, outputFile, pmin, pmax):

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)

    hists = []
    #for particle, name in zip(['Pi', 'K', 'P'], ['#pi', 'K', 'p']):
    for particle, name in zip(['Pi', 'P'], ['#pi', 'p']):
        hist = TH1D(f'clusterSize_{particle}', r'; #LT Cluster size #GT #times #LT cos#lambda #GT', 50, 0, 25)
        data = inputData.filter( (pl.col('p') > pmin) & (pl.col('p') < pmax) & (pl.col('partID') == particlePDG[particle]) )
        fillTH1(hist, data, 'clSizeCosL')
        hist.SetTitle(r'; #LT Cluster size #GT #times #LT cos#lambda #GT')
        hist.Scale(1. / hist.Integral())
        hists.append(hist)

    canvas = TCanvas('canvas', 'canvas', 1000, 800)
    hframe = canvas.DrawFrame(0, 0., 7, 0.5, '; #LT Cluster size #GT #times #LT cos(#lambda) #GT; Normalised counts')
    #canvas.SetLogy()

    hframe.GetYaxis().SetDecimals()
    hframe.GetXaxis().SetDecimals()
    hframe.GetXaxis().SetMaxDigits(2)
    hframe.GetYaxis().SetMaxDigits(2)
    hframe.GetYaxis().SetNoExponent()
    hframe.GetYaxis().SetTitleSize(0.07)
    hframe.GetXaxis().SetTitleSize(0.07)
    hframe.GetYaxis().SetLabelSize(0.04)
    hframe.GetXaxis().SetLabelSize(0.04)

    #for hist, col in zip(hists, [kAzure+4, kGreen+3, kRed+1]):
    for hist, col in zip(hists, [kAzure+4, kRed+1]):
        
        hist.SetFillColorAlpha(col, 0.5)
        hist.SetLineColor(col)
        hist.SetFillStyle(3356)
        canvas.cd()
        hist.Draw('hist same')

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.4, 0.85, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.4, 0.8, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')
    latex.DrawLatex(0.4, 0.75, f'{pmin} < #it{{p}} < {pmax} GeV/#it{{c}}')
    latex.DrawLatex(0.4, 0.7, 'Particles identified with the TPC')


    legend = TLegend(0.45, 0.59, 0.85, 0.69)
    legend.SetNColumns(3)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.04)
    #for hist, name in zip(hists, ['#pi', 'K', 'p']):    legend.AddEntry(hist, name, 'f')
    for hist, name in zip(hists, ['#pi', 'p']):    legend.AddEntry(hist, name, 'f')
    legend.Draw()
    
    canvas.SaveAs(outputFile)



if __name__ == '__main__':


    cfgPath = '/home/galucia/ITS_Cluster_Studies/PID_ITS2/configs/cfgPID_general.yml'
    with open(cfgPath, 'r') as f:   cfgGeneral = yaml.safe_load(f)
    with open(cfgGeneral['cfgFile'][cfgGeneral['opt']], 'r') as f:    cfg = yaml.safe_load(f)

    data = LoadData(['/data/shared/pp23_relval_debug/0720/its_PIDStudy.root', '/data/shared/pp23_relval_debug/0730/its_PIDStudy.root', '/data/shared/pp23_relval_debug/0740/its_PIDStudy.root', '/data/shared/pp23_relval_debug/0750/its_PIDStudy.root', '/data/shared/pp23_relval_debug/0800/its_PIDStudy.root'])

    # preprocess train and test data
    dpData = DataPreprocessor.CreatePreprocessor(data, cfg, cfgGeneral['opt'])
    dpData.Preprocess()
    dpData.ApplyCuts()
    dpData.ParticleID()
    dpData.CleanData()
    
    drawClSizeVsP(dpData.data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/WP2/clSizeVsP.pdf')
    drawClSizeVsPSub(dpData.data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/WP2/clSizeVsP_sub.pdf')
    drawClSizeSelection(dpData.data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/WP2/clSizeSelection.pdf', 0.4, 0.6)
    drawClSizeSpecies(dpData.data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/WP2/clSizeSpecies.pdf', 0.4, 0.6)