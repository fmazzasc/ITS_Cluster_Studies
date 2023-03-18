#
#   Classes and functions used in the machine learning process
#

import pandas as pd
import numpy as np
import sys

from sklearn.metrics import make_scorer

sys.path.append('..')
from utils.plotter import Plotter

#################################################################################################################
#
#   ML classes
#________________________________________________________________________________________________________________

class Scorer:
    """
    Attributes
    ----------
        model (): ML model
        df (pd.DataFrame): Entire dataset
        RegressionColumns (list[str]): Columns used for the regression
        nameYcol (str): Name of the target column
    """
    def __init__(self, model, df, RegressionColumns, nameYcol, tfile) -> None:
        self.model = model
        self.df = df
        self.RegressionColumns = RegressionColumns
        self.nameYcol = nameYcol
        self.predcol = f'{self.nameYcol}_pred'
        self.plot = Plotter(self.df, tfile)

    def Delta(self, absolute=False):
        """
        Evaluate a resolution (y - pred)/y for the model. It will be appended as a column of the df. 

        Parameters
        ----------
            absolute (bool, optional): Whether the resolution should be defined as an absolute value.

        Returns
        -------
            Dataset with appended Delta column.
        """
        self.df[self.predcol] = self.model.predict(self.df[self.RegressionColumns])
        if absolute:    self.df.eval(f'Delta = abs({self.nameYcol} - {self.predcol}) / {self.nameYcol}', inplace=True)
        else:           self.df.eval(f'Delta = ({self.nameYcol} - {self.predcol}) / {self.nameYcol}', inplace=True)
        return self.df

    def histScore(self, nbinsx=300, xlow=-1.5, xup=1.5):
        self.plot.plot1D(['Delta'], [['#Delta', nbinsx, xlow, xup]])

    def scatterPlotScore(self, xVariable, xLabel, nbinsx, xlow, xup, nbinsy=300, ylow=-1.5, yup=1.5):
        self.plot.plot2D([xVariable], ['Delta'], [[xLabel, '#Delta', nbinsx, xlow, xup, nbinsy, ylow, yup]])

#################################################################################################################
#
#   Specific ----- functions
#________________________________________________________________________________________________________________

EXIT_THRESHOLD = 0.5

def callback(study, trial):
    """
    Function to set an early stopping to optuna. It will be passed as a **kwarg to ModelHandler.optimize_params_optuna
    The function is written accordingly to https://github.com/optuna/optuna/issues/966
    """
    if trial.value < EXIT_THRESHOLD:
        raise optuna.StopStudy

def Delta(model, X, y, absolute=True):
    pred = model.predict(X)

    if absolute:    return abs(y - pred)/y
    else:           return (y - pred)/y

def Delta_alt(model, X, y, RegressionColumns, absolute=True):
    pred = model.predict(X[RegressionColumns])
    for n, (i, j) in enumerate(zip(pred, X['preds'])):
        if(i != j):  print(pred.head(n+5), X['preds'].head(n+5))
    if absolute:    return abs(y - pred)/y
    else:           return (y - pred)/y
    
    
#def Delta_score(model, X, y, weight:pd.Series):
def Delta_score(model, X, y):
    """
    Variable used to score a ML training process. The variable is a weighted average of abs(y_true - y_pred)/y_true
    """

    pred = model.predict(X)
    #Delta = weight * abs(y - pred)/y
    Delta = abs(y - pred)/y
    return Delta.sum()/len(y)
    #return Delta.sum()/weight.sum()

def _deltaScore(y_true, y_pred, **kwargs):
    delta = (y_true - y_pred)**2
    return delta.sum() / len(y_true)

delta_scorer = make_scorer(_deltaScore, greater_is_better=False)

def plot_score(X, y, RegressionColumns, model, x_label, plot_specifics, x=pd.Series(), filename='', absolute=False):
    """
    Plot a prediction scoring variable (defined as (true-predicted)/true) vs a chosen variable from X columns.

    Parameters:
    - model: model (or pipeline) used for the predictions
    - plot_specifics: list with the following entries [nbinsx, xlow, xup, nbinsy, ylow, yup]
    """

    if 'preds' not in X.columns:    X['preds'] = model.predict(X[RegressionColumns])
    if 'Delta' not in X.columns:
        print('ayaayyay')    
        if absolute:    X.eval('Delta = abs(beta - preds)/beta', inplace=True)
        else:           X.eval('Delta = (beta - preds)/beta', inplace=True)

    plot_spec = [x_label, '#Delta'] + plot_specifics
    if x.empty:     density_scatter(y, X['Delta'], f'{filename}_score_scatter', plot_spec)
    else:           density_scatter(x, X['Delta'], f'{filename}_score_scatter', plot_spec)

    plot_spec_hist = [f'#Delta'] + plot_specifics[3:]
    hist(X['Delta'], f'{filename}_score_hist', plot_spec_hist)

def plot_score_train(TrainTestData, RegressionColumns, model, x_label, plot_specifics, x_train=pd.Series(), x_test=pd.Series(), filename='', absolute=False):
    """
    Plot a prediction scoring variable (defined as (true-predicted)/true) vs a chosen variable from X columns.

    Parameters:
    - TrainTestData: list [X_train, y_train, X_test, y_test] that will be used for training and test (Xs still have the column passed to the ys)
    - model: model (or pipeline) used for the predictions
    - plot_specifics: list with the following entries [nbinsx, xlow, xup, nbinsy, ylow, yup]
    - abs: choose if the delta parameter is defined as the absolute value or not
    """

    X_train, y_train, X_test, y_test = TrainTestData

    if 'preds' not in X_train.columns:    X_train['preds'] = model.predict(X_train[RegressionColumns])
    if 'Delta' not in X_train.columns:    
        if absolute:    X_train.eval('Delta = abs(beta - preds)/beta', inplace=True)
        else:           X_train.eval('Delta = (beta - preds)/beta', inplace=True)

    if 'preds' not in X_test.columns:    X_test['preds'] = model.predict(X_test[RegressionColumns])
    if 'Delta' not in X_test.columns:    
        if absolute:    X_test.eval('Delta = abs(beta - preds)/beta', inplace=True)
        else:           X_test.eval('Delta = (beta - preds)/beta', inplace=True)

    plot_spec = [x_label, '#Delta'] + plot_specifics
    if type(x_train) == pd.Series and x_train.empty:       
        density_scatter(y_train, X_train['Delta'], f'{filename}_score_scatter_train', plot_spec, title='Score scatter train')
    else:                   density_scatter(x_train, X_train['Delta'], f'{filename}_score_scatter_train_x', plot_spec, title='Score scatter train')

    if type(x_test) == pd.Series and x_test.empty:        
        density_scatter(y_test, X_test['Delta'], f'{filename}_score_scatter_test', plot_spec, title='Score scatter test')
    else:                   density_scatter(x_test, X_test['Delta'], f'{filename}_score_scatter_test_p', plot_spec, title='Score scatter test')

    # no column will be used, since delta_train, delta_test are not dfs.
    multiple_hist([X_train['Delta'], X_test['Delta']], '', plot_specifics[3:], f'{filename}_score_hist', hist_names=['Train', 'Test'])


