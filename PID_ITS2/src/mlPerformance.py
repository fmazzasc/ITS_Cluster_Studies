#
#

from ROOT import TH1D, TCanvas

def hyperparameterScoreCanvas(hyperparameterImportance):

    hyperparameterImportance = sorted(hyperparameterImportance.items(), key=lambda x: x[1], reverse=True)
    h = TH1D('h', '', len(hyperparameterImportance), 0, len(hyperparameterImportance))
    for ientry, (hyperparameter, score) in enumerate(hyperparameterImportance.items()):
        h.SetBinContent(ientry+1, score)
        h.SetBinLabel(ientry+1, hyperparameter)

    c = TCanvas('hyperparameterImportance', 'Hyperparamter importance; Score; Hyperparameter', 800, 600)
    h.Draw('bar hist')
    
    return c
        