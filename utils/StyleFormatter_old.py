'''
Script with helper methods for style settings
'''

import ctypes
from os import replace
import numpy as np
from ROOT import gStyle, TLine, TLatex, TGaxis, TPad, TColor, TGraphAsymmErrors, TGraphErrors, TH1 # pylint: disable=import-error,no-name-in-module
from ROOT import kBlack, kWhite, kGray, kRed, kBlue, kGreen # pylint: disable=import-error,no-name-in-module
from ROOT import kMagenta, kAzure, kCyan, kOrange, kYellow, kSpring, kTeal, kViolet, kPink # pylint: disable=import-error,no-name-in-module
from ROOT import kFullCircle, kFullSquare, kFullDiamond, kFullCross, kFullTriangleUp, kFullTriangleDown # pylint: disable=import-error,no-name-in-module
from ROOT import kOpenCircle, kOpenSquare, kOpenDiamond, kOpenCross, kOpenTriangleUp, kOpenTriangleDown # pylint: disable=import-error,no-name-in-module

kCR = TColor.GetFreeColorIndex()
cCR = TColor(kCR, 2./255, 172./255, 143./255, 'kCR', 1.0)
kAMPT = TColor.GetFreeColorIndex()
cAMPT = TColor(kAMPT, 4./255, 6./255, 143./255, 'kAMPT', 1.0)
kMonash = TColor.GetFreeColorIndex()
cMonash = TColor(kMonash, 6./255, 101./255, 247./255, 'kMonash', 1.0)
kGreenCool = TColor.GetFreeColorIndex()
cGreenCool = TColor(kGreenCool, 66./255, 155./255, 119./255, 'kGreenCool', 1.0)
kAzureCool = TColor.GetFreeColorIndex()
cAzureCool = TColor(kAzureCool, 85./255, 170./255, 216./255, 'kAzureCool', 1.0)
kOrangeCool = TColor.GetFreeColorIndex()
cOrangeCool = TColor(kOrangeCool, 248./255, 22./255, 36./255, 'kOrangeCool', 1.0)
kYellowCool = TColor.GetFreeColorIndex()
cYellowCool = TColor(kYellowCool, 204./255, 163./255, 0./255, 'kYellowCool', 1.0)
kDplusPrompt = TColor.GetFreeColorIndex()
cDplusPrompt = TColor(kDplusPrompt, 232/255, 202/255, 100/255, 'kDplusPrompt', 1.0)
kDplusFD = TColor.GetFreeColorIndex()
cDplusFD = TColor(kDplusFD, 67/255, 99/255, 114/255, 'kDplusFD', 1.0)
kDstarFD = TColor.GetFreeColorIndex()
cDstarFD = TColor(kDstarFD, 183/255, 150/255, 109/255, 'kDstarFD', 1.0)
kLHCb_2 = TColor.GetFreeColorIndex()
cLHCb_2 = TColor(kLHCb_2, 28/255, 177/255, 106/255, 'kLHCb_2', 1.0)
kLHCb_3 = TColor.GetFreeColorIndex()
cLHCb_3 = TColor(kLHCb_3, 235/255, 134/255, 41/255, 'kLHCb_3', 1.0)
kLHCb_4 = TColor.GetFreeColorIndex()
cLHCb_4 = TColor(kLHCb_4, 159/255, 7/255, 66/255, 'kLHCb_4', 1.0)
kDrays = TColor.GetFreeColorIndex()
cDrays = TColor(kDrays, 186/255, 76/255, 35/255, 'kDrays', 1.0)
kHInelastic = TColor.GetFreeColorIndex()
cHInelastic = TColor(kHInelastic, 235/255, 187/255, 50/255, 'kHInelastic', 1.0)
kPrimary = TColor.GetFreeColorIndex()
cPrimary = TColor(kPrimary, 132/255, 128/255, 168/255, 'kPrimary', 1.0)

# pylint: disable=too-many-branches, too-many-statements


def SetGlobalStyle(**kwargs):
    '''
    Method to set global style.

    Parameters
    ----------

    - padrightmargin (float), default = 0.035
    - padleftmargin (float), default = 0.12
    - padtopmargin (float), default = 0.035
    - padbottommargin (float), default = 0.12

    - titlesize (float), default = 0.050
    - titlesizex (float), default = 0.050
    - titlesizey (float), default = 0.050
    - titlesizez (float), default = 0.050

    - labelsize (float), default = 0.045
    - labelsizex (float), default = 0.045
    - labelsizey (float), default = 0.045
    - labelsizez (float), default = 0.045

    - titleoffset (float), default = 1.2
    - titleoffsetx (float), default = 1.2
    - titleoffsey (float), default = 1.2
    - titleoffsetz (float), default = 1.2

    - opttitle (int), default = 0
    - optstat (int), default = 0

    - padtickx (int), default = 1
    - padticky (int), default = 1

    - maxdigits (int), default no max value

    - palette (int), default kBird
    '''

    # pad margins
    if 'padrightmargin' in kwargs:
        gStyle.SetPadRightMargin(kwargs['padrightmargin'])
    else:
        gStyle.SetPadRightMargin(0.035)

    if 'padleftmargin' in kwargs:
        gStyle.SetPadLeftMargin(kwargs['padleftmargin'])
    else:
        gStyle.SetPadLeftMargin(0.12)

    if 'padtopmargin' in kwargs:
        gStyle.SetPadTopMargin(kwargs['padtopmargin'])
    else:
        gStyle.SetPadTopMargin(0.035)

    if 'padbottommargin' in kwargs:
        gStyle.SetPadBottomMargin(kwargs['padbottommargin'])
    else:
        gStyle.SetPadBottomMargin(0.1)

    # title sizes
    if 'titlesize' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesize'], 'xyz')
    else:
        gStyle.SetTitleSize(0.050, 'xyz')

    if 'titlesizex' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesizex'], 'x')
    if 'titlesizey' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesizex'], 'y')
    if 'titlesizez' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesizex'], 'z')

    # label sizes
    if 'labelsize' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsize'], 'xyz')
    else:
        gStyle.SetLabelSize(0.045, 'xyz')

    if 'labelsizex' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsizex'], 'x')
    if 'labelsizey' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsizex'], 'y')
    if 'labelsizez' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsizex'], 'z')
    if 'labeloffset' in kwargs:
        gStyle.SetLabelSize(kwargs['labeloffset'], 'xyz')

    # title offsets
    if 'titleoffset' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffset'], 'xyz')
    else:
        gStyle.SetTitleOffset(1.2, 'xyz')

    if 'titleoffsetx' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffsetx'], 'x')
    if 'titleoffsety' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffsety'], 'y')
    if 'titleoffsetz' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffsetz'], 'z')

    # other options
    if 'opttitle' in kwargs:
        gStyle.SetOptTitle(kwargs['opttitle'])
    else:
        gStyle.SetOptTitle(0)

    if 'optstat' in kwargs:
        gStyle.SetOptStat(kwargs['optstat'])
    else:
        gStyle.SetOptStat(0)

    if 'padtickx' in kwargs:
        gStyle.SetPadTickX(kwargs['padtickx'])
    else:
        gStyle.SetPadTickX(1)

    if 'padticky' in kwargs:
        gStyle.SetPadTickY(kwargs['padticky'])
    else:
        gStyle.SetPadTickY(1)

    gStyle.SetLegendBorderSize(0)

    if 'maxdigits' in kwargs:
        TGaxis.SetMaxDigits(kwargs['maxdigits'])

    if 'palette' in kwargs:
        gStyle.SetPalette(kwargs['palette'])


def SetObjectStyle(obj, **kwargs):
    '''
    Method to set root object style.

    Parameters
    ----------

    - obj: object to set style

    - linecolor (int) default 1 (black)
    - linealpha (float) default 1
    - linewitdh (int) default 2
    - linestyle (int) default 1

    - markercolor (int) default 1 (black)
    - markeralpha (float) default 1
    - markerstyle (int) default 20 (full circle)
    - markersize (int) default 20 (full circle)

    - fillcolor (int) default no filling
    - fillalpha (float) default 1
    - fillstyle (int) default 0 (no style)

    - color (int) sets same color for line, marker and fill
    - alpha (float) sets same alpha for line, marker and fill
    '''

    # alpha parameters
    lalpha = kwargs.get('linealpha', 1)
    malpha = kwargs.get('markeralpha', 1)
    falpha = kwargs.get('fillalpha', 1)
    if 'alpha' in kwargs:
        lalpha = kwargs['alpha']
        malpha = kwargs['alpha']
        falpha = kwargs['alpha']
    if 'linealpha' in kwargs:
        lalpha = kwargs['linealpha']
    if 'markeralpha' in kwargs:
        malpha = kwargs['markeralpha']
    if 'fillalpha' in kwargs:
        falpha = kwargs['fillalpha']

    # line styles
    if 'linecolor' in kwargs:
        if lalpha < 1:
            obj.SetLineColorAlpha(kwargs['linecolor'], lalpha)
        else:
            obj.SetLineColor(kwargs['linecolor'])
    else:
        if lalpha < 1:
            obj.SetLineColorAlpha(1, lalpha)
        else:
            obj.SetLineColor(1)

    if 'linewidth' in kwargs:
        obj.SetLineWidth(kwargs['linewidth'])
    else:
        obj.SetLineWidth(2)

    if 'linestyle' in kwargs:
        obj.SetLineStyle(kwargs['linestyle'])
    else:
        obj.SetLineStyle(1)

    # marker styles
    if 'markercolor' in kwargs:
        if malpha < 1:
            obj.SetMarkerColorAlpha(kwargs['markercolor'], malpha)
        else:
            obj.SetMarkerColor(kwargs['markercolor'])
    else:
        if malpha < 1:
            obj.SetMarkerColorAlpha(1, malpha)
        else:
            obj.SetMarkerColor(1)

    if 'markersize' in kwargs:
        obj.SetMarkerSize(kwargs['markersize'])
    else:
        obj.SetMarkerSize(1)

    if 'markerstyle' in kwargs:
        obj.SetMarkerStyle(kwargs['markerstyle'])
    else:
        obj.SetMarkerStyle(20)

    # fill styles
    if 'fillcolor' in kwargs:
        if falpha < 1:
            obj.SetFillColorAlpha(kwargs['fillcolor'], falpha)
        else:
            obj.SetFillColor(kwargs['fillcolor'])

    if 'fillstyle' in kwargs:
        obj.SetFillStyle(kwargs['fillstyle'])

    #global color
    if 'color' in kwargs:
        if lalpha < 1:
            obj.SetLineColorAlpha(kwargs['color'], lalpha)
        else:
            obj.SetLineColor(kwargs['color'])
        if malpha < 1:
            obj.SetMarkerColorAlpha(kwargs['color'], malpha)
        else:
            obj.SetMarkerColor(kwargs['color'])
        if falpha < 1:
            obj.SetFillColorAlpha(kwargs['color'], falpha)
        else:
            obj.SetFillColor(kwargs['color'])


def DivideCanvas(canv, nPads):
    '''
    Method to divide ROOT canvases

    Parameters
    ----------

    - canv: TCanvas to be divided
    - nPads: number (int) of pads in which divide the canvas

    '''
    if nPads < 2:
        canv.cd()
    elif nPads in [2, 3]:
        canv.Divide(int(nPads), 1)
    elif nPads in [4, 6, 8]:
        canv.Divide(int(nPads/2), 2)
    elif nPads in [5, 7]:
        canv.Divide(int((nPads+1)/2), 2)
    elif nPads in [12, 15]:
        canv.Divide(int(nPads/3), 3)
    elif nPads in [10, 11]:
        canv.Divide(4, 3)
    elif nPads in [13, 14]:
        canv.Divide(5, 3)
    elif 15 < nPads <= 20 and nPads % 4 == 0:
        canv.Divide(int(nPads/4), 4)
    elif 15 < nPads <= 20 and nPads % 4 != 0:
        canv.Divide(5, 4)
    elif nPads == 21:
        canv.Divide(7, 3)
    elif 21 < nPads <= 25:
        canv.Divide(5, 5)
    elif nPads > 25 and nPads % 2 == 0:
        canv.Divide(int(nPads/2), 2)
    else:
        canv.Divide(int((nPads+1)/2), 2)


def GetROOTColor(color='kBlack'):
    '''
    Method to retrieve a ROOT color

    Parameters
    ----------

    - color: color according to ROOT TColor convention

    Returns
    ----------

    - ROOT color corresponding to input color

    '''
    cMapROOT = {'kBlack': kBlack, 'kWhite': kWhite, 'kGrey': kGray,
                'kRed': kRed, 'kBlue': kBlue, 'kGreen': kGreen,
                'kTeal': kTeal, 'kAzure': kAzure, 'kCyan': kCyan,
                'kOrange': kOrange, 'kYellow': kYellow, 'kSpring': kSpring,
                'kMagenta': kMagenta, 'kViolet': kViolet, 'kPink': kPink}

    ROOTcolor = None
    for colorKey in cMapROOT:
        if colorKey in color:
            ROOTcolor = cMapROOT.get(colorKey)
            break
    if ROOTcolor:
        for shade in range(0, 11):
            if f' + {shade}' in color or f'+{shade}' in color:
                ROOTcolor += shade
                break
            elif f' - {shade}' in color or f'-{shade}' in color:
                ROOTcolor -= shade
                break

    return ROOTcolor


def GetROOTMarker(marker='kFullCircle'):
    '''
    Method to retrieve the ROOT marker map

    Parameters
    ----------

    - color: color according to ROOT TColor convention

    Returns
    ----------

    - ROOT color corresponding to input color

    '''
    mMapROOT = {'kFullCircle': kFullCircle, 'kFullSquare': kFullSquare, 'kFullDiamond': kFullDiamond,
                'kFullCross': kFullCross, 'kFullTriangleUp': kFullTriangleUp, 'kFullTriangleDown': kFullTriangleDown,
                'kOpenCircle': kOpenCircle, 'kOpenSquare': kOpenSquare, 'kOpenDiamond': kOpenDiamond,
                'kOpenCross': kOpenCross, 'kOpenTriangleUp': kOpenTriangleUp, 'kOpenTriangleDown': kOpenTriangleDown}

    if marker in mMapROOT:
        ROOTmarker = mMapROOT.get(marker)
    else:
        ROOTmarker = None

    return ROOTmarker


def ReturnAdjacentPads(nrows, ncols, leftmargin=0.14, rightmargin=186/25535, bottommargin=0.12,
                       topmargin=186/25535, corrfactwidth=0., corrfactheight=0.): # pylint: disable=too-many-statements
    '''
    Method that returns a numpy array of adjacent TPads

    Inputs
    ----------
    - number of rows in the canvas
    - number of columns in the canvas

    Optional
    - left margin (float), default = 0.14
    - right margin (float), default = 186/25535
    - bottom margin (float), default = 0.12
    - top margin (float), default = 186/25535
    - correction factor to fill remaining spaces in the horizontal direction
    - correction factor to fill remaining spaces in the vertical direction

    Returns
    ----------
    - numpy array of TPads with correct sizes and margins
    '''
    middleWidthMargin = (leftmargin+rightmargin)
    if ncols == 3:
        middleWidthMargin /= 2
    middleHeightMargin = (topmargin+bottommargin)
    padWidth = (1. / ncols)
    padHeight = (1. / nrows)

    x1Coord, x2Coord, y1Coord, y2Coord = ([] for iList in range(4))
    for iRow in range(nrows):
        x1Coord.append([])
        x2Coord.append([])
        y1Coord.append([])
        y2Coord.append([])
        for iCol in range(ncols):
            if ncols == 1:
                x1Coord[iRow].append(0)
                x2Coord[iRow].append(1)
            else:
                if iCol == 0:
                    x1Coord[iRow].append(0)
                    x2Coord[iRow].append(padWidth+middleWidthMargin/(2*ncols)+corrfactwidth)
                elif iCol == ncols-1:
                    x1Coord[iRow].append(1-padWidth-middleWidthMargin/(2*ncols)-corrfactwidth)
                    x2Coord[iRow].append(1)
                else:
                    x1Coord[iRow].append(x2Coord[iRow][iCol-1]-2*middleWidthMargin/(2*ncols)-corrfactwidth)
                    x2Coord[iRow].append(x1Coord[iRow][iCol]+padWidth+middleWidthMargin/(2*ncols)+corrfactwidth)

            if nrows == 1:
                y1Coord[iRow].append(0)
                y2Coord[iRow].append(1)
            else:
                if iRow == 0:
                    y1Coord[iRow].append(1-padHeight-middleHeightMargin/(2*nrows)-corrfactheight)
                    y2Coord[iRow].append(1)
                elif iRow == nrows-1:
                    y1Coord[iRow].append(0)
                    y2Coord[iRow].append(padHeight+middleHeightMargin/(2*nrows)+corrfactheight)
                else:
                    y2Coord[iRow].append(y1Coord[iRow-1][iCol]+middleHeightMargin/(2*nrows)+corrfactheight)
                    y1Coord[iRow].append(y2Coord[iRow][iCol]-padHeight-2*middleHeightMargin/(2*nrows)-corrfactheight)

    outPads = []
    for iRow in range(nrows):
        outPads.append([])
        for iCol in range(ncols):
            outPads[iRow].append(TPad(f'pad{iCol}{iRow}', '',
                                      x1Coord[iRow][iCol], y1Coord[iRow][iCol],
                                      x2Coord[iRow][iCol], y2Coord[iRow][iCol]))

            if nrows == 1:
                outPads[iRow][iCol].SetTopMargin(topmargin)
                outPads[iRow][iCol].SetBottomMargin(bottommargin)
            else:
                if iRow == 0:
                    outPads[iRow][iCol].SetTopMargin(topmargin)
                    if topmargin < bottommargin:
                        outPads[iRow][iCol].SetBottomMargin(bottommargin-topmargin)
                    else:
                        outPads[iRow][iCol].SetBottomMargin(0.)
                elif iRow == nrows-1:
                    outPads[iRow][iCol].SetBottomMargin(bottommargin)
                    if bottommargin < topmargin :
                        outPads[iRow][iCol].SetTopMargin(topmargin-bottommargin)
                    else:
                        outPads[iRow][iCol].SetTopMargin(0.)
                else:
                    outPads[iRow][iCol].SetBottomMargin(middleHeightMargin)
                    outPads[iRow][iCol].SetTopMargin(middleHeightMargin)

            if ncols == 1:
                outPads[iRow][iCol].SetLeftMargin(leftmargin)
                outPads[iRow][iCol].SetLeftMargin(rightmargin)
            else:
                if iCol == 0:
                    outPads[iRow][iCol].SetLeftMargin(leftmargin)
                    if leftmargin < rightmargin:
                        outPads[iRow][iCol].SetRightMargin(rightmargin-leftmargin)
                    else:
                        outPads[iRow][iCol].SetRightMargin(0.)
                elif iCol == ncols-1:
                    outPads[iRow][iCol].SetRightMargin(rightmargin)
                    if rightmargin < leftmargin:
                        outPads[iRow][iCol].SetLeftMargin(leftmargin-rightmargin)
                    else:
                        outPads[iRow][iCol].SetLeftMargin(0.)
                else:
                    outPads[iRow][iCol].SetRightMargin(0.)
                    outPads[iRow][iCol].SetLeftMargin(middleWidthMargin)

    return np.array(outPads)


def SetXsystForLogScale(objStat, graphSyst, perc=0.6):
    '''
    Method to set width of systematic uncertainty box for plots with x axis in log scale

    Inputs
    ----------
    - objStat: graph/histo with statistical uncertainties
    - graphSyst: graph with systematic uncertainties
    - perc: percentage of width compared to statistical uncertainties

    '''

    if isinstance(graphSyst, TGraphErrors):
        DummygraphSyst = TGraphAsymmErrors()
        for iPtGraph in range(graphSyst.GetN()):
            DummygraphSyst.SetPointError(iPtGraph,
                                         graphSyst.GetErrorXlow(iPtGraph),
                                         graphSyst.GetErrorXhigh(iPtGraph),
                                         graphSyst.GetErrorYlow(iPtGraph),
                                         graphSyst.GetErrorYhigh(iPtGraph)
                                        )
        graphSyst = DummygraphSyst
    
    if isinstance(objStat, TGraphAsymmErrors):
        for iPt in range(objStat.GetN()):
            statUncLow = objStat.GetErrorXlow(iPt)
            statUncHigh = objStat.GetErrorXhigh(iPt)
            systUncLow = statUncLow * perc
            systUncHigh = statUncHigh * perc
            graphSyst.SetPointEXlow(iPt, systUncLow)
            graphSyst.SetPointEXhigh(iPt, systUncHigh)

    elif isinstance(objStat, TH1):
        for iPt in range(objStat.GetNbinsX()):
            statUnc = objStat.GetBinWidth(iPt+1) / 2
            systUncLow = statUnc * perc
            systUncHigh = statUnc * perc
            for iPtGraph in range(graphSyst.GetN()):
                ptGraph, yGraph = ctypes.c_double(), ctypes.c_double()
                graphSyst.GetPoint(iPtGraph, ptGraph, yGraph)
                if abs(ptGraph.value - objStat.GetBinCenter(iPt+1)) < statUnc:
                    graphSyst.SetPointEXlow(iPtGraph, systUncLow)
                    graphSyst.SetPointEXhigh(iPtGraph, systUncHigh)
                    break
    else:
        print('WARNING: width of systematic uncertainty in x not set!')


def SetFixedXsyst(graphSyst, syst=0.2):
    '''
    Method to set fixed width of systematic uncertainty box for plots

    Inputs
    ----------
    - graphSyst: graph with systematic uncertainties
    - syst: fixed systematic uncertainty

    '''
    for iPt in range(graphSyst.GetN()):
        graphSyst.SetPointEXlow(iPt, syst)
        graphSyst.SetPointEXhigh(iPt, syst)

def SetStringColor(text, colour=kRed):
    '''
    Method to set color of output string
    
    Inputs
    ----------
    - text: string to be coloured
    -colour: selected color
     
    '''
    if colour=='kRed':
        print("\033[91m{}\033[00m" .format(text))
    else:
        print(f'Only kRed color defined.\n{text}')

def SetLegendStyle(leg, **kwargs):
    '''
    Method to set legend style.

    Parameters
    ----------

    - leg: object to set style

    - textsize (double) default 186/25535

    - bordersize (int) default 0 (none)

    - margin (double) default none

    - ncolumns (int) default 1

    - header (string) default none

    - fillstyle (int) default 0 (no style)
    '''

    # size parameter
    if 'textsize' in kwargs:
        leg.SetTextSize(kwargs['textsize'])
    else:
        leg.SetTextSize(186/25535)
    if 'bordersize' in kwargs:
        leg.SetBorderSize(kwargs['bordersize'])
    else:
        leg.SetBorderSize(0)
    if 'margin' in kwargs:
        leg.SetMargin(kwargs['margin'])
    # header
    if 'header' in kwargs:
        leg.SetHeader(kwargs['header'])
    # ncolumns
    if 'ncolumns' in kwargs:
        leg.SetNColumns(kwargs['ncolumns'])

    # fillstyle
    if 'filstyle' in kwargs:
        leg.SetFillStyle(kwargs['header'])
    else:
        leg.SetFillStyle(0)

def EmptyCloneProducer(histo):
    hEmptyClone = histo.Clone()
    fullmaker = hEmptyClone.GetMarkerStyle()
    hEmptyClone.SetName(f'{histo.GetName()}_Empty')
    lcolor = histo.GetMarkerColor()
    SetObjectStyle(hEmptyClone, fillstyle=0, linecolor=lcolor, markercolor=kBlack)
    return hEmptyClone

def LineAtOne(min, max, linecolor='', linewidth=1):
    lineAtOne = TLine(min, 1., max, 1.)
    lineAtOne.SetLineStyle(9)
    lineAtOne.SetLineWidth(linewidth)
    linecolor = linecolor if linecolor else kGray+2
    lineAtOne.SetLineColor(linecolor)
    return lineAtOne

def LatLabel(text, xtext, ytext, textsize=186/25555, textfont=42, textcolor=kBlack):
    latLabel = TLatex()
    latLabel.SetNDC()
    latLabel.SetTextSize(textsize)
    latLabel.SetTextFont(textfont)
    latLabel.SetTextColor(textcolor)
    latLabel.DrawLatex(xtext, ytext, f'{text}')

# define custom colors to mimic transparency
kAzureMy = TColor.GetFreeColorIndex()
cAzureMy = TColor(kAzureMy, 159./255, 191./255, 223./255, 'kAzureMy', 1.0)
kOrangeMy = TColor.GetFreeColorIndex()
cOrangeMy = TColor(kOrangeMy, 255./255, 204./255, 128./255, 'kOrangeMy', 1.0)
kRedMy = TColor.GetFreeColorIndex()
cRedMy = TColor(kRedMy, 250./255, 153./255, 153./255, 'kRedMy', 1.0)
kGreenMy = TColor.GetFreeColorIndex()
cGreenMy = TColor(kGreenMy, 179./255, 230./255, 179./255, 'kGreenMy', 1.0)