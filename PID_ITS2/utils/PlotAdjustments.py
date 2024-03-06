'''
Code to adjust TH1, TH2 etc
'''
from os import system
import yaml
import sys
from ROOT import TFile, TLatex, TCanvas, gStyle, TGaxis, TGraphAsymmErrors, TLegend, TH1F, TDirectoryFile, TMath, TF1, kRed, kBird # pylint: disable=import-error,no-name-in-module
sys.path.append('../..')
from utils.StyleFormatter import SetGlobalStyle, SetObjectStyle, LatLabel

SetGlobalStyle(padbottommargin=0.14, padleftmargin=0.15, padrightmargin=0.15,
               padtopmargin=0.05, titleoffsety=.8, maxdigits=3, palette=53,
               titlesizex=0.05, titlesizey=0.05, labelsize=0.03)

gStyle.SetPalette(53)   # -> Stefano
#gStyle.SetPalette(1)   # -> ROOT 5
#gStyle.SetPalette(kBird) 

#configPath = '../configs/config_adj.yml'
configPath = '../configs/config_adj2.yml'

with open(configPath, 'r') as ymlCfgFile:
    inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)

finPaths = inputCfg['finPaths']
objNames = inputCfg['objNames']

foutPaths = inputCfg['foutPaths']
outFormats = inputCfg['outFormats']

axisTitles = inputCfg['axisTitles']
axisLimits = inputCfg['axisLimits']
latexLabels = inputCfg['latexLabels']
sampleLabels = inputCfg['sampleLabels']
runLabels = inputCfg['runLabels']

c_width = inputCfg['canvas']['width']
c_height = inputCfg['canvas']['height']

for finPath, objName, foutPath, outFormat, axisTitle, axisLimit, latexLabel, sampleLabel, runLabel, width, height in zip(finPaths, objNames, foutPaths, outFormats, axisTitles, axisLimits, latexLabels, sampleLabels, runLabels, c_width, c_height ):
    
    inFile = TFile.Open(finPath)
    obj = inFile.Get(objName)
    obj.SetMinimum(1.)

    canvas = TCanvas('canvas', '', width, height)
    canvas.cd().SetLogz()

    hFrame = canvas.cd().DrawFrame(axisLimit[0], axisLimit[1], axisLimit[2], axisLimit[3], axisTitle)
    hFrame.GetYaxis().SetDecimals()
    hFrame.GetXaxis().SetDecimals()
    hFrame.GetZaxis().SetDecimals()

    obj.Draw('samecolz')

    latALICE = LatLabel(sampleLabel, latexLabel[0],
                        latexLabel[1], 0.05)
    run_label = 'Run 3 MC' if runLabel else 'Run 3'
    latSystem = LatLabel(f'{run_label}', latexLabel[0],
                             latexLabel[2], 0.04)
    latSystem = LatLabel('pp, #sqrt{#it{s}} = 13.6 TeV', latexLabel[0],
                         latexLabel[3], 0.04)
    canvas.Update()

    for outformat in outFormat:
        canvas.SaveAs(f'{foutPath}.{outformat}')

    input('Press enter to continue')


