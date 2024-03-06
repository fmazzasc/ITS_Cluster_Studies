'''
    Script to add a theorical line to the beta vs p scatter plots
'''

from ROOT import TFile, TF1, TCanvas, kRed, gPad, gStyle

#______________________________________
#   GLOBAL VARIABLES
#______________________________________

mass_Deu = 1.8756
mass_P =  0.93827200
mass_K = 0.4937
mass_Pi = 0.13957000
mass_E = 0.000511

names = ['E', 'Pi', 'K', 'P', 'Deu']
mass_dict = dict(zip(names, [mass_E, mass_Pi, mass_K, mass_P, mass_Deu]))

#______________________________________

def addTheoricalBetaLine(canvas, mass, xmin, xmax, lineName='betaLine'):
    '''
    Function to create a curve beta vs p for a particle of given mass. The curve is a TF1 that will be draw on a given TCanvas.

    Parameters
    ----------
        canvas (TCanvas): canvas where the line should be drawn
        mass (float): mass of the particle whose line should be drawn
        xmin (float): lower limit for the function
        xmax (float): upper limit for the function
        lineName (str): name of the line that will be drawn
    '''

    betaLine = TF1(lineName, f'x / sqrt(x^2 + {mass}^2)', xmin, xmax)
    canvas.cd()
    betaLine.SetLineColor(kRed)
    betaLine.Draw('same')

if __name__ == '__main__':

    inPath = '../output/TPC/application_beta_pflat.root'
    inFile = TFile.Open(inPath)
    hist = inFile.Get('beta_pred_vs_p')

    canvas = TCanvas('beta_pred_vs_p_line', '#beta vs #it{p}')
    gStyle.SetPalette(53)
    canvas.cd().SetLogz()
    hist.Draw('colz')
    hist.SetDirectory(0)
    inFile.Close()

    functions = []
    for part in names:  
        betaLine = TF1(f'betaLine{part}', f'x/(x*x + [0]*[0])^(1/2)', 0., 1.5)
        betaLine.SetParameter(0, mass_dict[part])
        betaLine.SetLineColor(kRed)
        betaLine.SetNpx(1000)
        functions.append(betaLine)
    
    for func in functions:  func.Draw('same')

    outPath = '../output/TPC/theoretical_line_beta_pflat.root'
    outFile = TFile(outPath, 'recreate')

    canvas.Write()
    canvas.SaveAs('betaML_vs_p_line.png')
    outFile.Close()

