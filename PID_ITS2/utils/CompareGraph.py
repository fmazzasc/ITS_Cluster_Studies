'''
Script for the comparison of ROOT TH1s or TGraphs
run: python CompareGraphs.py cfgFileName.yml
'''

import sys
from os.path import join
import argparse
import numpy as np
import yaml
from ROOT import TCanvas, TFile, TLegend, TLine, gStyle, TList, TPaveText # pylint: disable=import-error,no-name-in-module
sys.path.append('../..')
from utils.StyleFormatter_old import SetGlobalStyle, SetObjectStyle, GetROOTColor, GetROOTMarker #pylint: disable=wrong-import-position,import-error
from utils.AnalysisUtils import ComputeRatioDiffBins, ScaleGraph, ComputeRatioGraph #pylint: disable=wrong-import-position,import-error

def StatLegendPosition(x, y, w, h):
    gStyle.SetStatX(x)
    gStyle.SetStatY(y)
    gStyle.SetStatW(w)
    gStyle.SetStatH(h)

def min_size(*args):
    if not args:    return 0
    list = [len(arg) for arg in args]
    return min(list)


# load inputs
parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('cfgFileName', metavar='text', default='config_comparison.yml')
args = parser.parse_args()

with open(args.cfgFileName, 'r') as ymlCfgFile:
    inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)

inDirName = inputCfg['inputs']['dirname']
inFileNames = inputCfg['inputs']['filenames']
objNames = inputCfg['inputs']['objectnames']

outputFileName = inputCfg['output']['filename']
outExtensions = inputCfg['output']['extensions']
outFileNames = inputCfg['output']['objectnames']

objTypes = inputCfg['options']['ROOTobject']
scales = inputCfg['options']['scale']
normalizes = inputCfg['options']['normalize']
colors = inputCfg['options']['colors']
markers = inputCfg['options']['markers']
markersize = inputCfg['options']['markersize']
linewidth = inputCfg['options']['linewidth']
fillstyles = inputCfg['options']['fillstyle']
fillalphas = inputCfg['options']['fillalpha']
drawOptions = inputCfg['options']['drawopt']
rebins = inputCfg['options']['rebin']

doRatio = inputCfg['options']['ratio']['enable']
drawRatioUnc = inputCfg['options']['ratio']['uncertainties']['enable']
ratioUncCorr = inputCfg['options']['ratio']['uncertainties']['corr']
displayRMS = inputCfg['options']['ratio']['displayRMS']

doCompareUnc = inputCfg['options']['errcomp']['enable']
compareRelUnc = inputCfg['options']['errcomp']['relative']
KS = inputCfg['options']['KS']

wCanv = inputCfg['options']['canvas']['width']
hCanv = inputCfg['options']['canvas']['heigth']
xLimits = inputCfg['options']['canvas']['xlimits']
yLimits = inputCfg['options']['canvas']['ylimits']
yLimitsRatio = inputCfg['options']['canvas']['ylimitsratio']
yLimitsUnc = inputCfg['options']['canvas']['ylimitserr']
xTitle = inputCfg['options']['canvas']['xaxistitle']
yTitle = inputCfg['options']['canvas']['yaxistitle']
logX = inputCfg['options']['canvas']['logx']
logY = inputCfg['options']['canvas']['logy']
logZ = inputCfg['options']['canvas']['logz']
ratioLogX = inputCfg['options']['canvas']['ratio']['logx']
ratioLogY = inputCfg['options']['canvas']['ratio']['logy']
uncCompLogX = inputCfg['options']['canvas']['errcomp']['logx']
uncCompLogY = inputCfg['options']['canvas']['errcomp']['logy']

avoidStatBox = inputCfg['options']['statbox']['avoid']
xStatLimits = inputCfg['options']['statbox']['xlimits']
yStatLimits = inputCfg['options']['statbox']['ylimits']
statheader = inputCfg['options']['statbox']['header']
statTextSize = inputCfg['options']['statbox']['textsize']

avoidLeg = inputCfg['options']['legend']['avoid']
xLegLimits = inputCfg['options']['legend']['xlimits']
yLegLimits = inputCfg['options']['legend']['ylimits']
legNames = inputCfg['options']['legend']['titles']
legOpt = inputCfg['options']['legend']['options']
header = inputCfg['options']['legend']['header']
legTextSize = inputCfg['options']['legend']['textsize']
ncolumns = inputCfg['options']['legend']['ncolumns']

avoidStatLegend = inputCfg['options']['statlegend']['avoid']
xStatLegend = inputCfg['options']['statlegend']['x']
yStatLegend = inputCfg['options']['statlegend']['y']
wStatLegend = inputCfg['options']['statlegend']['w']
hStatLegend = inputCfg['options']['statlegend']['h']

single = (min_size(inFileNames, outFileNames, objNames, objTypes, scales, normalizes, colors, markers, fillstyles, fillalphas, rebins) == 1)
if single:  doRatio, doCompareUnc = False, False        # auto-disable ratio and compare features if a single object is created


# set global style
SetGlobalStyle(padleftmargin=0.18, padbottommargin=0.14, titleoffsety=1.5, optstat=1111)
if avoidStatLegend:     gStyle.SetOptStat(0)


pave = TPaveText(xStatLimits[0], yStatLimits[0], xStatLimits[1], yStatLimits[1])
pave.SetFillStyle(0)
pave.SetTextSize(statTextSize)

leg = TLegend(xLegLimits[0], yLegLimits[0], xLegLimits[1], yLegLimits[1])
leg.SetFillStyle(0)
leg.SetTextSize(legTextSize)
leg.SetNColumns(ncolumns)
leg.SetHeader(header)

hToCompare, hRatioToCompare, hUncToCompare = [], [], []
for iFile, (inFileName, outFileName, objName, objType, scale, normalize, color, marker, fillstyle, fillalpha, rebin) in \
    enumerate(zip(inFileNames, outFileNames, objNames, objTypes, scales, normalizes, colors, markers, fillstyles, fillalphas, rebins)):
    
    if inDirName:
        inFileName = join(inDirName, inFileName)
    inFile = TFile.Open(inFileName)
    if inFile == None:
        print(f"ERROR: cannot open {inFileName}. Check your config. Exit!")
        sys.exit()
    if inFile.Get(objName) == None:
        print(f"ERROR: couldn't load the histogram \'{objName}\' in \'{inFileName}\'. Check your config. Exit! ")
        sys.exit()
    hToCompare.append(inFile.Get(objName))
    if 'TH' in objType:
        if len(outFileNames) == len(inFileNames):   hToCompare[iFile].SetName(outFileName)
        else:                                       hToCompare[iFile].SetName(f'h{iFile}')
    else:
        hToCompare[iFile].SetName(f'g{iFile}')
    if 'TH' in objType:     hToCompare[iFile].Rebin(rebin)
    SetObjectStyle(hToCompare[iFile],
                   color=GetROOTColor(color),
                   markerstyle=GetROOTMarker(marker),
                   markersize=markersize,
                   linewidth=linewidth,
                   fillstyle=fillstyle,
                   fillalpha=fillalpha)
    if 'TH' in objType:
        hToCompare[iFile].SetDirectory(0)               
        if normalize:
            if scale != 1.:
                print('WARNING: you are both scaling and normalizing the histogram, check if it makes sense!')
            hToCompare[iFile].Scale(1. / hToCompare[iFile].Integral())
        hToCompare[iFile].Scale(scale)
    else:
        ScaleGraph(hToCompare[iFile], scale)
    if doRatio:
        if 'TH' in objType:
            if drawRatioUnc:
                if ratioUncCorr:
                    hRatioToCompare.append(ComputeRatioDiffBins(hToCompare[iFile], hToCompare[0], 'B'))
                else:
                    hRatioToCompare.append(ComputeRatioDiffBins(hToCompare[iFile], hToCompare[0]))
            else:
                hRatioToCompare.append(ComputeRatioDiffBins(hToCompare[iFile], hToCompare[0]))
                for iBin in range(1, hRatioToCompare[iFile].GetNbinsX()+1):
                    hRatioToCompare[iFile].SetBinError(iBin, 1.e-20)
            hRatioToCompare[iFile].SetDirectory(0)
        else:
            if drawRatioUnc:
                if ratioUncCorr:
                    print('WARNING: correlated uncertainty in ratio for TGraphs not implemented. Switching off')
                    ratioUncCorr = False
                     #TODO: extend ComputeRatioGraph to account for correlated uncertainties
                else:
                    hRatioToCompare.append(ComputeRatioGraph(hToCompare[iFile], hToCompare[0]))
            else:
                hRatioToCompare.append(ComputeRatioGraph(hToCompare[iFile], hToCompare[0]))
                for iBin in range(hRatioToCompare[iFile].GetN()):
                    hRatioToCompare[iFile].SetPointEYlow(iBin, 1.e-20)
                    hRatioToCompare[iFile].SetPointEYhigh(iBin, 1.e-20)
        #TODO: add case to manage ratio between graph and histo (utility function already available in AnalysisUtils)
        hRatioToCompare[iFile].SetName(f'hRatio{iFile}')
        SetObjectStyle(hRatioToCompare[iFile],
                       color=GetROOTColor(color),
                       markerstyle=GetROOTMarker(marker),
                       markersize=markersize,
                       linewidth=linewidth,
                       fillstyle=fillstyle,
                       fillalpha=fillalpha)
    if doCompareUnc:
        if 'TH' in objType:
            hUncToCompare.append(hToCompare[iFile].Clone(f'hUncToCompare{iFile}'))
            for iBin in range(1, hUncToCompare[iFile].GetNbinsX()+1):
                unc = hUncToCompare[iFile].GetBinError(iBin)
                cent = hUncToCompare[iFile].GetBinContent(iBin)
                if compareRelUnc:
                    unctocomp = unc/cent if cent != 0 else 0
                    hUncToCompare[iFile].SetBinContent(iBin, unctocomp)
                else:
                    hUncToCompare[iFile].SetBinContent(iBin, unc)
                hUncToCompare[iFile].SetBinError(iBin, 1.e-20)
            hUncToCompare[iFile].SetDirectory(0)
            SetObjectStyle(hUncToCompare[iFile],
                           color=GetROOTColor(color),
                           markerstyle=GetROOTMarker(marker),
                           markersize=markersize,
                           linewidth=linewidth,
                           fillstyle=fillstyle,
                           fillalpha=fillalpha)
        else:
            #TODO: add uncertainty comparison for TGraphs
            print('WARNING: uncertainty comparison for TGraphs not implemented. Switching off')
            doCompareUnc = False
    
    if KS:
        KS_val = hToCompare[iFile].KolmogorovTest(hToCompare[0])
        print(f'\033[92mKS test for {objName} in {inFileName} is {np.around(KS_val, decimals=3)}\033[0m')

    leg.AddEntry(hToCompare[iFile], legNames[iFile], legOpt[iFile])

ratios, RMS, shift = [], [], []
if doRatio and displayRMS:
    for iBin in range(hRatioToCompare[1].GetNbinsX()):
        ratios.append([])
        for iFile, _ in enumerate(inFileNames):
            if iFile == 0:
                continue
            ratios[iBin].append(hRatioToCompare[iFile].GetBinContent(iBin+1))
        aRatios = np.array(ratios[iBin])
        RMS.append(np.std(aRatios))
        shift.append(np.mean(aRatios))
print('\033[92mRMS values:', np.around(RMS, decimals=3), '\033[0m')
print('\033[92mshift values:', np.around(shift, decimals=3) - 1., '\033[0m')

cOut = TCanvas('cOutput', '', wCanv, hCanv)

if doRatio or doCompareUnc:
    if doRatio and doCompareUnc:
        cOut.Divide(3, 1)
        ratioPad = 2
        uncPad = 3
    else:
        cOut.Divide(2, 1)
        if doRatio:
            ratioPad = 2
        else:
            uncPad = 2

    hFrame = cOut.cd(1).DrawFrame(xLimits[0], yLimits[0], xLimits[1], yLimits[1], f';{xTitle};{yTitle}')
    if logX:
        cOut.cd(1).SetLogx()
    if logY:
        cOut.cd(1).SetLogy()
    if logZ:
        cOut.cd(1).SetLogz()
else:
    hFrame = cOut.cd().DrawFrame(xLimits[0], yLimits[0], xLimits[1], yLimits[1], f';{xTitle};{yTitle}')
    if logX:
        cOut.cd().SetLogx()
    if logY:
        cOut.cd().SetLogy()
    if logZ:
        cOut.cd().SetLogz()
    
hFrame.GetYaxis().SetDecimals()


list = TList()

# Draw main histograms
for ihisto, (histo, objType, drawOpt) in enumerate(zip(hToCompare, objTypes, drawOptions)):
    if 'TH' in objType:
        list.Add(histo)
        if not avoidStatLegend: StatLegendPosition(xStatLegend[ihisto], yStatLegend[ihisto], wStatLegend[ihisto], hStatLegend[ihisto])      
        histo.DrawCopy(f'{drawOpt}same')
        print( legNames[ihisto], f'Entries: {histo.GetEntries()}', f'Mean: {round(histo.GetMean(), 3)}', f'Std Dev: {round(histo.GetStdDev(), 3)}' )
    else:
        histo.Draw(drawOpt)


# Create stat box (entries, mean and stddev of all histograms drawn)
if not avoidStatBox:
    h = hToCompare[0].Clone('h')
    h.Reset()
    h.Merge(list)

    if all('TH' in objType for objType in objTypes):
            pave.AddText(0., 0.7, f'Entries: {h.GetEntries()}')
            pave.AddText(0., 0.4, f'Mean: {round(h.GetMean(), 3)}')
            pave.AddText(0., 0.1, f'Std Dev: {round(h.GetStdDev(), 3)}')
    pave.Draw()

# Draw legend
if single:  avoidLeg = True
if  not avoidLeg:   leg.Draw()

if doRatio:
    hFrameRatio = cOut.cd(ratioPad).DrawFrame(xLimits[0], yLimitsRatio[0], xLimits[1], yLimitsRatio[1],
                                              f';{xTitle};Ratio')
    hFrameRatio.GetYaxis().SetDecimals()
    if ratioLogX:
        cOut.cd(ratioPad).SetLogx()
    if ratioLogY:
        cOut.cd(ratioPad).SetLogy()
    lineAtOne = TLine(xLimits[0], 1., xLimits[1], 1.)
    lineAtOne.SetLineColor(GetROOTColor('kBlack'))
    lineAtOne.SetLineWidth(2)
    lineAtOne.SetLineStyle(9)
    lineAtOne.Draw()
    for iHisto, (histo, objType, drawOpt) in enumerate(zip(hRatioToCompare, objTypes, drawOptions)):
        if iHisto > 0:
            if 'TH' in objType:
                histo.DrawCopy(f'{drawOpt}same')
            else:
                histo.Draw(drawOpt)

if doCompareUnc:
    if compareRelUnc:
        uncTitle = 'Relative uncertainties'
    else:
        uncTitle = 'Absolute uncertainties'
    hFrameUnc = cOut.cd(uncPad).DrawFrame(xLimits[0], yLimitsUnc[0], xLimits[1], yLimitsUnc[1],
                                          f';{xTitle};{uncTitle}')
    hFrameUnc.GetYaxis().SetDecimals()
    if uncCompLogX:
        cOut.cd(uncPad).SetLogx()
    if uncCompLogY:
        cOut.cd(uncPad).SetLogy()
    for iHisto, (histo, drawOpt) in enumerate(zip(hUncToCompare, drawOptions)):
        histo.DrawCopy(f'{drawOpt}same')

for ext in outExtensions:
    if 'root' in ext:
        outFile = TFile(f'{outputFileName}.root', 'recreate')
        cOut.Write()
        for histo in hToCompare:
            histo.Write()
        if doRatio:
            for histo in hRatioToCompare:
                histo.Write()
        if doCompareUnc:
            for histo in hUncToCompare:
                histo.Write()
        outFile.Close()
    else:
        cOut.SaveAs(f'{outputFileName}.{ext}')

input("Press enter to exit")
