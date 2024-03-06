#
#

import os
from matplotlib.figure import Figure, Axes
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ROOT import TFile, TImage, TCanvas, gROOT

def saveMatplotlibToRootFile(pltPlot, outFile, pltName):

    if isinstance(pltPlot, Axes):           pltPlot.savefig('tmp.png')
    elif isinstance(pltPlot, Figure):       pltPlot.savefig('tmp.png')
    elif isinstance(pltPlot, go.Figure):    pltPlot.write_image('tmp.png')
    else:                                   raise ValueError(f"Unknown matplotlib object: {type(pltPlot)}")
    
    img = TImage.Open('tmp.png')
    img.SetConstRatio(0)
    
    canvas = TCanvas()
    canvas.SetName(pltName)
    img.Draw('')

    outFile.cd()
    canvas.Write()

    os.remove('tmp.png')

    
