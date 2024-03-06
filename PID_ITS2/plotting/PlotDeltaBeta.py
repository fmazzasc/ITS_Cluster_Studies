import sys
sys.path.append('..')
from src.preprocessing import *
from src.dataVisual import DataVisual, DataVisualSlow
from src.loadData import LoadData

from utils.particles import particleMasses, particlePDG

sys.path.append('../..')
from utils.StyleFormatter import SetGlobalStyle, SetObjectStyle, SetXsystForLogScale, LatLabel, kDrays, kHInelastic, kPrimary, LineAtOne, SetLegendStyle, kAzureCool, kDplusPrompt, kRed, kGreen, kOrange

from ROOT import TFile, TH1D, TH2D, TCanvas, TLegend, TLatex, gStyle, TPad, TF1, TGraph
from ROOT import kRed, kAzure, kGreen, kOrange
import yaml

import polars as pl
from dataclasses import dataclass


# draw ranges
@dataclass
class DrawRange:

    dBetaRange: list
    dBetaSliceRange: list
    dBetaVsPrange: list

drawRanges = {'Pi': DrawRange([-.11, 10, .5, 1e5, r'; #beta^{exp}_{#pi} #minus #beta_{BDT} ; Counts'],                    # delta beta
                              [-.11, 1, .5, 2e5, r'; #beta^{exp}_{#pi} #minus #beta_{BDT} ; Counts'],                     # delta beta slice
                              [-0.01, -.4, 0.8, .1, r'; #it{p} (GeV/#it{c}); #beta^{exp}_{#pi} #minus #beta_{BDT} ; Counts']    # delta beta vs p
                            ),
              'K':  DrawRange([-.55, 10, 0.33, 1e5, r'; #beta^{exp}_{K} #minus #beta_{BDT} ; Counts'],                     # delta beta
                              [-.55, 1, .33, 2e5, r'; #beta^{exp}_{K} #minus #beta_{BDT} ; Counts'],                       # delta beta slice
                              [-0.01, -.4, 0.8, .1, r'; #beta^{exp}_{K} #minus #beta_{BDT} ; Counts']      # delta beta vs p
                              ),
              'P':  DrawRange([-.64, 10, .05, 8e5, r'; #beta^{exp}_{p} #minus #beta_{BDT} ; Counts'],                    # delta beta
                              [-.64, 1, .02, 8e5, r'; #beta^{exp}_{p} #minus #beta_{BDT} ; Counts'],                       # delta beta slice
                              [-0.01, -.4, 0.8, .1, r'; #beta^{exp}_{p} #minus #beta_{BDT} ; Counts']      # delta beta vs p
                              )
              }

colors = {'Pi': kAzure+4,
          'K': kGreen+3,
          'P': kRed+1
         }
          
SetGlobalStyle(padbottommargin=0.17, padleftmargin=0.14, padrightmargin=0.15,
               padtopmargin=0.05, titleoffsety=.9, titleoffsetx=0.9, titleoffsetz=1., 
               #maxdigits=2, 
               palette=53, titlesizex=0.05, titlesizey=0.05, labelsize=0.03)





def fillTH1(hist, df, var):
    for varVal in df[var]:  hist.Fill(varVal)

def fillTH2(hist, df, var1, var2):
    for var1Val, var2Val in zip(df[var1], df[var2]):  hist.Fill(var1Val, var2Val)







def drawDeltaBeta(inputData, outputFile, particle, name, pmin, pmax):

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    drawRage = drawRanges[particle]

    hist = TH1D('deltaBetaP', r'; #frac{#beta_{ML} - #beta_{p}}{#beta_{p}} ; Counts', 275, -4.5, 1)
    fillTH1(hist, inputData, f'deltaBeta{particle}')    #OverBeta{particle}')

    hist.SetFillColorAlpha(kOrange-3, 0.5)
    hist.SetLineColor(kOrange-3)
    hist.SetFillStyle(3356)

    canvas = TCanvas('canvas', 'canvas', 1000, 800)
    hframe = canvas.DrawFrame(drawRage.dBetaRange[0], drawRage.dBetaRange[1], drawRage.dBetaRange[2], 
                              drawRage.dBetaRange[3], drawRage.dBetaRange[4])
                              #fr'; (#beta_{{ML}} - #beta_{{{name}}})/#beta_{{{name}}} ; Counts')
    canvas.SetLogy()

    hframe.GetYaxis().SetDecimals()
    hframe.GetXaxis().SetDecimals()
    hframe.GetYaxis().SetTitleSize(0.07)
    hframe.GetXaxis().SetTitleSize(0.07)
    hframe.GetYaxis().SetLabelSize(0.04)
    hframe.GetXaxis().SetLabelSize(0.04)

    hist.Draw('same hist')

    canvas.SaveAs(outputFile)

    sliceData = inputData.filter((pl.col('p') > pmin) &
                                 (pl.col('p') < pmax))
    histSlice = TH1D('deltaBetaPSlice', r'; #frac{#beta_{ML} - #beta_{p}}{#beta_{p}} ; Counts', 150, -2.5, .5)
    fillTH1(histSlice, sliceData, f'deltaBeta{particle}') #OverBeta{particle}')

    histSlice.SetTitle(r'; #frac{#beta_{ML} - #beta_{p}}{#beta_{p}} ; Counts')
    histSlice.SetFillColorAlpha(kOrange-3, 0.5)
    histSlice.SetLineColor(kOrange-3)
    histSlice.SetFillStyle(3356)

    canvasSlice = TCanvas('canvasSlice', 'canvasSlice', 1000, 800)  
    hframeSlice = canvasSlice.DrawFrame(drawRage.dBetaSliceRange[0], drawRage.dBetaSliceRange[1], drawRage.dBetaSliceRange[2], 
                                        drawRage.dBetaSliceRange[3], drawRage.dBetaSliceRange[4])
                                        #fr'; (#beta_{{ML}} - #beta_{{{name}}})/#beta_{{{name}}} ; Counts')
    canvasSlice.SetLogy()

    hframeSlice.GetYaxis().SetDecimals()
    hframeSlice.GetXaxis().SetDecimals()
    hframeSlice.GetYaxis().SetTitleSize(0.06)
    hframeSlice.GetXaxis().SetTitleSize(0.06)
    hframeSlice.GetYaxis().SetLabelSize(0.04)
    hframeSlice.GetXaxis().SetLabelSize(0.04)

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.2, 0.88, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.2, 0.83, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')
    latex.DrawLatex(0.2, 0.78, f'{pmin} < #it{{p}} < {pmax} GeV/#it{{c}}')

    histSlice.Draw('same hist')

    canvasSlice.SaveAs(outputFile.replace('.pdf', '_slice.pdf'))

    # with tagged particles

    #taggedData = inputData.filter((pl.col('partID') == particlePDG[particle]) &
    #                              (pl.col('p') > pmin) &
    #                              (pl.col('p') < pmax))
    taggedData = sliceData.filter(pl.col('partID') == particlePDG[particle])
    histTagged = TH1D(f'deltaBeta{particle}Tagged', r'; #frac{#beta_{ML} - #beta_{p}}{#beta_{p}} ; Counts', 150, -2.5, .5)
    fillTH1(histTagged, taggedData, f'deltaBeta{particle}')# OverBeta{particle}')

    histTagged.SetTitle(r'; #frac{#beta_{ML} - #beta_{p}}{#beta_{p}} ; Counts')
    histTagged.SetFillColorAlpha(colors[particle], 0.5)
    histTagged.SetLineColor(colors[particle])
    histTagged.SetFillStyle(3356)

    pTaggedData = sliceData.filter(pl.col('partID') == particlePDG['P'])
    pHistTagged = TH1D('deltaBetapTagged', r'; #frac{#beta_{ML} - #beta_{p}}{#beta_{p}} ; Counts', 150, -2.5, .5)
    if particle == 'K': fillTH1(pHistTagged, pTaggedData, f'deltaBeta{particle}') #OverBeta{particle}')

    pHistTagged.SetFillColorAlpha(colors['P'], 0.5)
    pHistTagged.SetLineColor(colors['P'])
    pHistTagged.SetFillStyle(3356)

    canvasTagged = TCanvas('canvasTagged', 'canvasTagged', 1000, 800)
    hframeTagged = canvasTagged.DrawFrame(drawRage.dBetaSliceRange[0], drawRage.dBetaSliceRange[1], drawRage.dBetaSliceRange[2],
                                          drawRage.dBetaSliceRange[3], drawRage.dBetaSliceRange[4])
                                          #fr'; (#beta_{{ML}} - #beta_{{{name}}})/#beta_{{{name}}} ; Counts')     
    canvasTagged.SetLogy()

    hframeTagged.GetYaxis().SetDecimals()
    hframeTagged.GetXaxis().SetDecimals()
    hframeTagged.GetXaxis().SetMaxDigits(2)
    hframeTagged.GetYaxis().SetMaxDigits(2)
    hframeTagged.GetXaxis().SetNoExponent()
    hframeTagged.GetYaxis().SetTitleSize(0.07)
    hframeTagged.GetXaxis().SetTitleSize(0.07)
    hframeTagged.GetYaxis().SetLabelSize(0.04)
    hframeTagged.GetXaxis().SetLabelSize(0.04)

    histSlice.Draw('hist same')
    histTagged.Draw('hist same')
    if particle == 'K': pHistTagged.Draw('hist same')

    latex.DrawLatex(0.42, 0.88, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.42, 0.83, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')
    latex.DrawLatex(0.42, 0.78, f'{pmin} < #it{{p}} < {pmax} GeV/#it{{c}}')
    latex.DrawLatex(0.42, 0.73, f'Particles identified with the TPC') 

    legend = TLegend(0.55, 0.57, 0.77, 0.72)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.04)
    #if particle != 'K': legend.SetNColumns(2)
    legend.AddEntry(histSlice, 'All particles', 'f')
    legend.AddEntry(histTagged, f'{name}', 'f')
    if particle == 'K': legend.AddEntry(pHistTagged, 'p', 'f')

    legend.Draw('same')

    canvasTagged.SaveAs(outputFile.replace('.pdf', '_tagged.pdf'))

def drawDeltaBetaVsP(inputData, outputFile, particle):

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    drawRange = drawRanges[particle]

    hist = TH2D('deltaBetaP', r'; #it{p} (GeV/#it{c}); (#beta_{{ML}} - #beta_{{{name}}})/#beta_{{{name}}} ; Counts', 1500, 0, 1.5, 275, -4.5, 1)
    fillTH2(hist, inputData, 'p', f'deltaBeta{particle}')#OverBeta{particle}')

    canvas = TCanvas('canvas', '', 900, 800)
    canvas.cd().SetLogz()
    hFrame = canvas.cd().DrawFrame(drawRange.dBetaVsPrange[0], drawRange.dBetaVsPrange[1], drawRange.dBetaVsPrange[2],
                                   drawRange.dBetaVsPrange[3], drawRange.dBetaVsPrange[4])
                                   #it{{p}} (GeV/#it{{c}}); (#beta_{{ML}} - #beta_{{{name}}})/#beta_{{{name}}} ; Counts')

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

    hist.Draw('colz same')
    
    canvas.SaveAs(outputFile)

def drawPurityVsEfficiency(inputData, outputFile, particle, thresholds, pmin, pmax):

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)

    inputData = inputData.filter((pl.col('p') > pmin) &
                                 (pl.col('p') < pmax))
    inputParticle = inputData.filter(pl.col('partID') == particlePDG[particle])
    purity = []
    efficiency = []

    for threshold in thresholds:

        tmpInputData = inputData.filter(np.abs(pl.col(f'deltaBeta{particle}')) < threshold)
        tmpInputParticle = inputParticle.filter(np.abs(pl.col(f'deltaBeta{particle}')) < threshold)

        tmpPurity = tmpInputParticle.shape[0] / tmpInputData.shape[0]
        tmpEfficiency = tmpInputParticle.shape[0] / inputParticle.shape[0]

        purity.append(tmpPurity)
        efficiency.append(tmpEfficiency)

    # EFFICIENCY vs THRESHOLD
    
    graph = TGraph(len(thresholds), np.asarray(thresholds, dtype=float), np.asarray(efficiency, dtype=float))
    graph.SetTitle(r'; Threshold ; Efficiency')
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.5)
    graph.SetMarkerColor(kAzure+4)

    canvas = TCanvas('canvasEff', '', 900, 800)
    hframe = canvas.DrawFrame(-0.2, 0, -0.1, 1, r'; Threshold ; Efficiency')

    #hframe.GetYaxis().SetDecimals()
    #hframe.GetXaxis().SetDecimals()
    #hframe.GetYaxis().SetMaxDigits(2)
    #hframe.GetXaxis().SetMaxDigits(2)   
    hframe.GetYaxis().SetTitleSize(0.07)
    hframe.GetXaxis().SetTitleSize(0.07)
    hframe.GetYaxis().SetLabelSize(0.04)
    hframe.GetXaxis().SetLabelSize(0.04)
    
    graph.Draw('AP same')
    
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)

    latex.DrawLatex(0.3, 0.38, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.3, 0.33, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')
    latex.DrawLatex(0.3, 0.28, f'{pmin} < #it{{p}} < {pmax} GeV/#it{{c}}')
    latex.DrawLatex(0.3, 0.23, f'Particles identified with the TPC')

    canvas.SaveAs(outputFile.replace('.pdf', '_eff.pdf'))

    # PURITY vs THRESHOLD

    graph = TGraph(len(thresholds), np.asarray(thresholds, dtype=float), np.asarray(purity, dtype=float))
    graph.SetTitle(r'; Threshold ; Purity')
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.5)
    graph.SetMarkerColor(kAzure+4)

    canvas = TCanvas('canvasPur', '', 900, 800)
    hframe = canvas.DrawFrame(-0.2, 0, -0.1, 1, r'; Threshold ; Purity')

    #hframe.GetYaxis().SetDecimals()
    #hframe.GetXaxis().SetDecimals()
    #hframe.GetYaxis().SetMaxDigits(2)
    #hframe.GetXaxis().SetMaxDigits(2)
    hframe.GetYaxis().SetTitleSize(0.07)
    hframe.GetXaxis().SetTitleSize(0.07)
    hframe.GetYaxis().SetLabelSize(0.04)
    hframe.GetXaxis().SetLabelSize(0.04)
    
    graph.Draw('AP same')

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)

    latex.DrawLatex(0.2, 0.38, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.2, 0.33, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')
    latex.DrawLatex(0.2, 0.28, f'{pmin} < #it{{p}} < {pmax} GeV/#it{{c}}')
    latex.DrawLatex(0.2, 0.23, f'Particles identified with the TPC')  

    canvas.SaveAs(outputFile.replace('.pdf', '_pur.pdf'))

    # PURITY vs EFFICIENCY

    graph = TGraph(len(efficiency), np.asarray(efficiency, dtype=float), np.asarray(purity, dtype=float))
    graph.SetTitle(r'; Efficiency ; Purity')
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.5)
    graph.SetMarkerColor(kAzure+4)

    canvas = TCanvas('canvasEffPur', '', 900, 800)
    hframe = canvas.DrawFrame(0, 0, 1, 1, r'; Efficiency ; Purity')

    #hframe.GetYaxis().SetDecimals()
    #hframe.GetXaxis().SetDecimals()
    #hframe.GetYaxis().SetMaxDigits(2)
    #hframe.GetXaxis().SetMaxDigits(2)
    hframe.GetYaxis().SetTitleSize(0.07)
    hframe.GetXaxis().SetTitleSize(0.07)
    hframe.GetYaxis().SetLabelSize(0.04)
    hframe.GetXaxis().SetLabelSize(0.04)

    graph.Draw('AP same')

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)

    latex.DrawLatex(0.2, 0.38, 'ALICE Performance')
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.2, 0.33, 'Run 3, pp #sqrt{#it{s}} = 13.6 TeV')
    latex.DrawLatex(0.2, 0.28, f'{pmin} < #it{{p}} < {pmax} GeV/#it{{c}}')
    latex.DrawLatex(0.2, 0.23, f'Particles identified with the TPC') 

    canvas.SaveAs(outputFile.replace('.pdf', '_effpur.pdf'))


if __name__ == '__main__':

    data = LoadData(['/home/galucia/ITS_Cluster_Studies/PID_ITS2/data/saved/application.parquet'])

    # preprocess train and test data

    data = data.with_columns(betaPi=(pl.col('p') / np.sqrt(pl.col('p')**2 + particleMasses['Pi']**2)),
                             betaK=(pl.col('p') / np.sqrt(pl.col('p')**2 + particleMasses['K']**2)),
                             betaP=(pl.col('p') / np.sqrt(pl.col('p')**2 + particleMasses['P']**2))
                             )
    data = data.with_columns(deltaBetaPi=(pl.col('betaPi') - pl.col('betaML')),
                             deltaBetaK=(pl.col('betaK') - pl.col('betaML')),
                             deltaBetaP=(pl.col('betaP') - pl.col('betaML'))
                             )
    data = data.with_columns(deltaBetaPiOverBetaPi=(pl.col('deltaBetaPi') / pl.col('betaPi')),
                             deltaBetaKOverBetaK=(pl.col('deltaBetaK') / pl.col('betaK')),
                             deltaBetaPOverBetaP=(pl.col('deltaBetaP') / pl.col('betaP'))
                            )

    drawDeltaBeta(data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/deltaBetaPi.pdf', 'Pi', '#pi', 0.1, 0.4)
    drawDeltaBeta(data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/deltaBetaK.pdf', 'K', 'K', 0.1, 0.4)
    drawDeltaBeta(data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/deltaBetaP.pdf', 'P', 'p', 0.4, 0.6)

    drawDeltaBetaVsP(data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/deltaBetaPVsP.pdf', 'P',)


    thresholds = np.arange(0.02, 0.2, 0.01)
    thresholds_Pi = np.arange(0.02, 0.36, 0.02)
    thresholds_K = np.arange(0.02, 0.36, 0.02)

    drawPurityVsEfficiency(data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/threshold_Pi.pdf', 
                           'Pi', thresholds_Pi, 0.4, 0.6)
    drawPurityVsEfficiency(data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/threshold_K.pdf', 
                           'K', thresholds_K, 0.4, 0.6)
    drawPurityVsEfficiency(data, '/home/galucia/ITS_Cluster_Studies/PID_ITS2/plots/threshold_P.pdf', 
                           'P', thresholds, 0.4, 0.6)

    