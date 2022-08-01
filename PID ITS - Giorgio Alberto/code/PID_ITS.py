import pickle
import pandas as pd
import numpy as np
from math import floor, ceil

from alive_progress import alive_bar
from time import time

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

import matplotlib.pyplot as plt

from hipe4ml.model_handler import ModelHandler
import optuna

from ROOT import TH2F, TCanvas, TH1F, gStyle, kBird, gPad, gROOT, TLegend
from ROOT_graph import set_obj_style, fill_hist

gROOT.SetBatch()

# Tags
#_____________________________________
tag_Deu = 'nSigmaDeuAbs < 1 and nSigmaPAbs > 5 and nSigmaKAbs > 5 and nSigmaPiAbs > 5 and p <= 1.2'
tag_P = 'nSigmaPAbs < 1 and nSigmaKAbs > 3 and nSigmaDeuAbs > 3 and p <= 1.1'
tag_K = 'nSigmaKAbs < 1 and nSigmaPiAbs > 3 and nSigmaPAbs > 3 and p <= 0.5'
tag_Pi = 'nSigmaPiAbs < 1 and nSigmaKAbs > 3 and p <= 0.4'

# Masses
#_____________________________________
mass_Deu = 1.8756
mass_P =  0.93827200
mass_K = 0.4937
mass_Pi = 0.13957000

names = ['Deu', 'P', 'K', 'Pi']

tag_dict = dict(zip(names, [tag_Deu, tag_P, tag_K, tag_Pi]))
mass_dict = dict(zip(names, [mass_Deu, mass_P, mass_K, mass_Pi]))


# Functions
#_______________________________________
#_______________________________________

def filtering(df_full, part_name='all', label=True, beta=True):
    """
    From the full datatframe, creates a new one saving only data relative to a chosen particle (filtering  with instructions in its tag).
    The new dataframe will have a label column where its particle species is specified and a beta column where beta is defined.

    Parameters:
    - df_full: full dataframe
    - part_name: name of the particle to filter

    Returns:
    a list of reduced dataframes
    """
    
    if part_name == 'all':  part_name = names

    dfs = [df_full.query(tag_dict[part]).reset_index(drop=True) for part in part_name]
    for df, part in zip(dfs, part_name): 
        if label:   df['label'] = part
        if beta:    df.eval(f'beta = p / sqrt( {mass_dict[part]}**2 + p**2)', inplace=True)
    return dfs


# Plot functions
#________________________________________
    
def hist(x, filename, canvas, plot_specifics, pad=None, normalized=True, save=True):
    """
    
    Parameters:
    - plot_specific: list with the following content -> [x_label, nbinsx, xlow, xup] 
    """
    
    [x_label, nbinsx, xlow, xup] = plot_specifics
    if pad != None: canvas.cd(pad)
    else:           canvas.cd()     

    hist = TH1F('scatter_plot', '', nbinsx, xlow, xup)
    fill_hist(hist, x)
    
    if normalized:  set_obj_style(hist, x_label=x_label, y_label='Normaliazed counts', line_color=38)
    else:           set_obj_style(hist, x_label=x_label, y_label='Counts', line_color=38)

    if normalized:  hist.DrawNormalized('hist')
    else:           hist.Draw()
    
    if save:        canvas.SaveAs(f'{filename}.root')

def multiple_hist(dfs, columns, plot_specifics, filename, logx=False, logy=True, normalized=True, hist_names=None, x_label=None):
    """
    Draw multiple histigrams (different particle species, same variable) on the same canvas. You can reiterate for different variables as well.
    
    Parameters
    --------------------------------------------
    - dfs: list of dataframes whose columns will fill the histograms (it also works with pd.Series)
    - columns: list of variables you want to draw histograms of
    - hist_names: list of names for each hist
    - plot_specific: list with the following content -> [nbinsx, xlow, xup] 
    - logz: if True, the z-axis will be in log scale

    You could also pass a list of lists as dfs. If that is the case, please input hist_names and x_label. YOu should anyway pass a nonempty list for columns. Any single elements inside of it will be fine
    - hist_names: list of names for each of the histograms
    - x_label: label of the x axis 
    """
    
    [nbinsx, xlow, xup] = plot_specifics

    for column in columns:
        
        canvas = TCanvas('canvas', 'canvas', 700, 700)
        legend = TLegend(0.8,0.63,0.97,0.8)
        if logy:    canvas.SetLogy()
        if logx:    canvas.SetLogx()

        for i, df in enumerate(dfs):
            if df.dtype == pd.DataFrame:    
                if 'label' in df.columns:   hist_name = f"{df['label'][0]}"
            if hist_names != None:          hist_name = hist_names[i]
            else:                           hist_name = f'{i}'

            hist = TH1F(hist_name, hist_name, nbinsx, xlow, xup)

            if df.dtype == pd.DataFrame:    
                fill_hist(hist, df[column])
                if normalized:  set_obj_style(hist, x_label=f'{column}', y_label='Normalized counts', line_color=(i+1), y_title_offset=1.7)
                else:           set_obj_style(hist, x_label=f'{column}', y_label='Counts', line_color=(i+1), y_title_offset=1.7)
            else:                           
                fill_hist(hist, df)
                if normalized:  set_obj_style(hist, x_label=x_label, y_label='Normalized counts', line_color=(i+1), y_title_offset=1.7)  
                else:           set_obj_style(hist, x_label=x_label, y_label='Counts', line_color=(i+1), y_title_offset=1.7)
            
            legend.AddEntry(hist_name, hist_name, 'l')

            if normalized:      hist.DrawNormalized('hist same')
            else:               hist.Draw('hist same')

            del hist

        legend.Draw('same')
        canvas.SaveAs(f'{filename}{column}.root')
        del canvas, legend

def density_scatter(x, y, filename, canvas, plot_specifics, pad=None, logz=True, logy=False, normalized=True, save=True): 
    """
    
    Parameters:
    - plot_specific: list with the following content -> [x_label, y_label, nbinsx, xlow, xup, nbinsy, ylow, yup]
    - logz: if True, the z-axis will be in log scale
    """
    
    [x_label, y_label, nbinsx, xlow, xup, nbinsy, ylow, yup] = plot_specifics
    
    if pad != None: canvas.cd(pad)
    else:           canvas.cd()     

    if logz:        canvas.SetLogz()
    if logy:        canvas.SetLogy()

    scatter_plot = TH2F('scatter_plot', '', nbinsx, xlow, xup, nbinsy, ylow, yup)
    fill_hist(scatter_plot, x, y)
    set_obj_style(scatter_plot, x_label=x_label, y_label=y_label, y_title_offset=1.7)
    gStyle.SetPalette(kBird)

    if normalized:  scatter_plot.DrawNormalized('COLZ1')
    else:           scatter_plot.Draw('COLZ1')
    
    if save:        canvas.SaveAs(f'{filename}.root')

def multiplots(xs, ys, n_pads, filename, plot, plot_specifics):
    """
    Parameters:
    - xs: list of pd.Series (or lists) containing the x values
    - ys: list of pd.Series (or lists) containing the y values

    - n_pads: number of different pads the canvas will be split into
    - plot: plot type ('hist', 'scatter_density') 
    """

    canvas = TCanvas('canvas', 'canvas', 700, 700)
    nx_pads = 2
    ny_pads = ceil(n_pads/2)
    canvas.Divide(nx_pads, ny_pads)

    if plot == 'hist':
        for i, x in enumerate(xs):
            hist(x, '', canvas, plot_specifics, pad=i, save=False)
    
    if plot == 'scatter_density':
        for i, (x, y) in enumerate(zip(xs, ys)):
            density_scatter(x, y, '', canvas, plot_specifics, pad=i, save=False)
    
    canvas.SaveAs(f'{filename}.root')

        







# Data Augmentation Functions
#_______________________________________

def augmentation_raw():
    return 0

def augmentation_betaflat():
    return 0


# ML Related Functions
#_______________________________________

EXIT_THRESHOLD = 0.5

def callback(study, trial):
    """
    Function to set an early stopping to optuna. It will be passed as a **kwarg to ModelHandler.optimize_params_optuna
    The function is written accordingly to https://github.com/optuna/optuna/issues/966
    """
    if trial.value < EXIT_THRESHOLD:
        raise optuna.StopStudy

def Delta(model, X, y, abs=False):
    pred = model.predict(X)
    if abs:     return abs(y - pred)/y
    else:       return abs(y - pred)/y

#def Delta_score(model, X, y, weight:pd.Series):
def Delta_score(model, X, y):
    """
    Variable used to score a ML training process. The variable is a weighted average of abs(y_true - y_pred)/y_true
    """

    pred = model.predict(X)
    #Delta = weight * abs(y - pred)/y
    Delta = abs(y - pred)/y
    return Delta.sum()
    #return Delta.sum()/weight.sum()

def plot_score(X, y, model, x_label, plot_specifics, x=None, filename='', logy=False, abs=True):
    """
    Plot a prediction scoring variable (defined as (true-predicted)/true) vs a chosen variable from X columns.

    Parameters:
    - model: model (or pipeline) used for the predictions
    - plot_specifics: list with the following entries [nbinsx, xlow, xup, nbinsy, ylow, yup]
    """

    delta = Delta(model, X, y, abs=abs)

    canvas = TCanvas('canvas', 'canvas', 700, 700)

    plot_spec = [x_label, '#Delta'] + plot_specifics
    if x == None:   density_scatter(y, delta, f'../ML_output/score{filename}', canvas, plot_spec, logy=logy)
    else:           density_scatter(x, delta, f'../ML_output/score{filename}', canvas, plot_spec, logy=logy)

    del canvas

def plot_score_train(TrainTestData, RegressionColumns, model, x_label, plot_specifics, x_train=None, x_test=None, filename='', logx=False, logy=False, abs=True):
    """
    Plot a prediction scoring variable (defined as (true-predicted)/true) vs a chosen variable from X columns.

    Parameters:
    - TrainTestData: list [X_train, y_train, X_test, y_test] that will be used for training and test (Xs still have the column passed to the ys)
    - model: model (or pipeline) used for the predictions
    - plot_specifics: list with the following entries [nbinsx, xlow, xup, nbinsy, ylow, yup]
    - abs: choose if the delta parameter is defined as the absolute value or not
    """

    X_train, y_train, X_test, y_test = TrainTestData

    delta_train = Delta(model, X_train[RegressionColumns], y_train, abs=abs)
    delta_test = Delta(model, X_test[RegressionColumns], y_test, abs=abs)

    canvas = TCanvas('canvas', 'canvas', 700, 700)

    plot_spec = [x_label, '#Delta'] + plot_specifics
    if x_train == None:     density_scatter(y_train, delta_train, f'../ML_output/score{filename}_scatter_train', canvas, plot_spec, logy=logy)
    else:                   density_scatter(x_train, delta_train, f'../ML_output/score{filename}_scatter_train', canvas, plot_spec, logy=logy)

    if x_test == None:      density_scatter(y_test, delta_test, f'../ML_output/score{filename}_scatter_test', canvas, plot_spec, logy=logy)
    else:                   density_scatter(x_test, delta_test, f'../ML_output/score{filename}_scatter_test', canvas, plot_spec, logy=logy)

    multiple_hist([delta_train, delta_test], [filename], plot_specifics[:3], '../ML_output/score_hist', logx=logx, hist_names=['Train', 'Test'], x_label='#Delta')

    del canvas






def data_prep(data_path, test_frac, do_plots=False):
    """"
    Data preparation function. The full data will be uploaded and dataframes for single particle species will be created according to the tag rules. 
    Histograms for some of the column variables will be created. A data augmentation process will be added.

    Parameters
    --------------------------------------------------
    - data_path: path to the data storing file
    - do_plots: choose if you want to create histograms
    - data_augm: select a data augmantaion algorithm (options:)

    Returns
    --------------------------------------------------
    - TrainTestData: list [X_train, y_train, X_test, y_test] that will be used for training and test (Xs still have the column passed to the ys)
    """

    # Upload data
    df_full = pd.read_parquet(data_path)

    # define some new columns
    df_full.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
    df_full.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
    df_full.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)

    for part in names:
        df_full[f'nSigma{part}Abs'] = abs(df_full[f'nSigma{part}'])
    
    # Filtering (and evaluating beta)
    #__________________________________
    df_full.query('p <= 50', inplace=True)
    df_Deu, df_P, df_K, df_Pi = filtering(df_full)

    #df_Deu = pd.concat([df_Deu.copy(), df_full.query('nSigmaDeu < 1 and p >= 1')])
    #df_P = pd.concat([df_P, df_full.query('nSigmaP < 1 and p >= 1')])
    #df_K = pd.concat([df_K, df_full.query('nSigmaK < 1 and p >= 0.5')])
    #df_Pi = pd.concat([df_Pi, df_full.query('nSigmaPi < 1 and p >= 0.25')])

    filt_dfs = [df_Deu, df_P, df_K, df_Pi]
    total_df = pd.concat(filt_dfs)
    

    # Data Visualization
    #_________________________________
    if do_plots:
        print('Data Visualization...')

        hist_variables = ['meanClsize', 'meanPattID', 'ClSizeL0', 'ClSizeL1', 'ClSizeL2', 'ClSizeL3', 'ClSizeL4', 'ClSizeL5', 'ClSizeL6', 'L6_L0']
        angular_hist_variables = ['SnPhiL0', 'SnPhiL1', 'SnPhiL2', 'SnPhiL3', 'SnPhiL4', 'SnPhiL5', 'SnPhiL6', 'meanSnPhi']
        hist_specifics = [100, 0, 100]
        multiple_hist(filt_dfs, hist_variables, hist_specifics, 'data_visual/')
        multiple_hist(filt_dfs, angular_hist_variables, [100, -1, 1], 'data_visual/')
        multiple_hist(filt_dfs, ['L6_L0'], [100, 0, 10], 'data_visual/')
        multiple_hist(filt_dfs, ['p'], [100, 0, 2], 'data_visual/')
        multiple_hist(filt_dfs, ['tgL'], [100, -10, 10], 'data_visual/')


        canvas = TCanvas('canvas', 'canvas', 700, 700)

        dedx_scatter_specifics = ['p', '#frac{dE}{dx}', 10000, 0, 1.3, 10000, 0, 600]
        beta_scatter_specifics = ['p', '#beta', 10000, 0, 1.3, 10000, 0, 1]
        for df in filt_dfs:
            density_scatter(df['p'], df['dedx'], '../data_visual/dEdx_{}'.format(df['label'][0]), canvas, dedx_scatter_specifics)
            density_scatter(df['p'], df['beta'], '../data_visual/beta_{}'.format(df['label'][0]), canvas, beta_scatter_specifics)
        density_scatter(df_full['p'], df_full['dedx'], '../data_visual/dEdx_all', canvas, dedx_scatter_specifics)
        density_scatter(total_df['p'], total_df['beta'], '../data_visual/beta_all', canvas, beta_scatter_specifics)
        
        del canvas

    
    # Data Preprocessing (eliminate negative cl.sizes, apply beta flat, apply data augmentation, split dataframes)
    #_________________________________
    print('Data Preprocessing...')
    
    # negative values are intended to be nans
    for i in range(7):
        total_df[f'ClSizeL{i}'] = np.where(total_df[f'ClSizeL{i}'] < 0, 0, total_df[f'ClSizeL{i}'])

    TrainSet, a, b, c = train_test_split(total_df, total_df.beta, test_size=test_frac ,random_state=0)



    # Beta flat weights
    #_________________________________

    betamins = [0., 0.2, 0.4, 0.6, 0.8]
    betamaxs = [0.2, 0.4, 0.6, 0.8, 1.0]
    weights = []

    for betamin, betamax in zip(betamins, betamaxs):
        weights.append( len(TrainSet.query(f'{betamin} <= beta < {betamax}'))/len(TrainSet) )

    conditions = [(total_df['beta'] >= betamin) & (total_df['beta'] < betamax) for betamin, betamax in zip(betamins, betamaxs)]
    for weight in weights:  weight = 1./weight
    total_df['beta_weight'] = np.select(conditions, weights)


    
    # Dataframe splitting
    #____________________________________
    
    TrainTestData = train_test_split(total_df, total_df.beta, test_size=test_frac ,random_state=0) 
    TrainTestData[1], TrainTestData[2] = TrainTestData[2], TrainTestData[1]                         # TrainTestData = [X_train, y_train, X_test, y_test]


    return TrainTestData
    

    





def regression(TrainTestData, ModelParams, HyperParamsRange, optimization=True, model_choice='xgboost', beta_flat=False):
    """
    Parameters:
    ----------------------
    - TrainTestData: list [X_train, y_train, X_test, y_test] that will be used for training and test
    - ModelParams: dict of hyperparameters
    - HyperParamsRange: dict of hyperparameters range - {hyper_par: (hp_min, hp_max)}
    - model_choice: select a model for the regression (options: 'xgboost')
    """

    X_train, y_train, X_test, y_test = TrainTestData
    unwanted = 'beta', 'label', 'beta_weights' 
    RegressionColumns = [col for col in X_train.columns if col not in unwanted]
    

    # Model definition
    #__________________________________
    if model_choice=='xgboost':     model = xgb.XGBRegressor()

    # Optuna optimization
    #__________________________________
    
    if optimization:
        model_handler = ModelHandler(model, RegressionColumns)
        model_handler.set_model_params(ModelParams)

        print('Initialize Optuna hyper-parameters optimization...')
        with alive_bar(title='Hyper-Parameters optimization') as bar:
            study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, Delta_score, direction='minimize', callbacks=[callback], timeout=600)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image('../ML_output/plot_optimization_history.ong')

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image('../ML_output/plot_param_importances.ong')

        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image('../ML_output/plot_parallel_coordinate.ong')

        fig = optuna.visualization.plot_contour(study)
        fig.write_image('../ML_output/plot_contour.ong')

        HyperParams = study.best_trial






    # Model evaluation
    #__________________________________
    if model_choice=='xgboost':

        print('Global model...')
        model_reg = xgb.XGBRegressor(HyperParams)
        #pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline = Pipeline(steps=[('model', model_reg)])

        with alive_bar(title='Training all') as bar:
            if beta_flat:       pipeline.fit(X_train, y_train, sample_weight=X_train['beta_weights'])
            else:               pipeline.fit(X_train, y_train)
        
        plot_specifics = [1000, 0, 1.3, 10000, -0.5, 4]
        plot_score_train(TrainTestData, RegressionColumns, pipeline, x_label='#beta', plot_specifics=plot_specifics, logy=True)

        for name in names:
            print(f'Modeling {name}...')

            train = pd.concat([X_train, y_train])
            X_train_name = train.query(f"label == '{name}'")
            y_train_name = X_train_name['beta']

            test = pd.concat([X_test, y_test])
            X_test_name = test.query(f"label == '{name}'")
            y_test_name = X_test_name['beta']
            
            with alive_bar(title=f'Training {name}') as bar:
                if beta_flat:   pipeline.fit(X_train_name[RegressionColumns], y_train_name, sample_weight=X_train_name['beta_weights'])
                else:           pipeline.fit(X_train_name[RegressionColumns], y_train_name)

            TestTrainData_name = X_train_name, y_train_name, X_test_name, y_test_name

            plot_score_train(TestTrainData_name, pipeline, x_label='#beta', plot_specifics=plot_specifics, filename=f'_{name}', logy=True)

    
    # Save model in pickle
    #_______________________________

    print('Saving regressor model...')
    with open(f'RegressorModel_{model_choice}.pickle', 'wb') as output_file:
        pickle.dump(model_reg, output_file)
    print('Model saved.')


    return model_reg


def augmentation(data_augm=''):

    # Data Augmentation
    #__________________________________
    if data_augm == 'option1':
        pass

    elif data_augm == 'option2':
        pass 

    elif data_augm == 'option3':
        pass



    




def main():

    # Input file
    #__________________________________
    data_path = '/Users/giogi/Desktop/Stage INFN/PID ITS/data/Df_filtered_ITS2Cluster505673.parquet.gzip'

    # Data Preparation
    #__________________________________
    TrainTestData = data_prep(data_path=data_path, test_frac=0.2, do_plots=False)

    # Training
    #__________________________________
    ModelParams = {'n_jobs': 2, 'max_depth':5, 'learning_rate':0.023, 'n_estimators':500}
    HyperParamsRange = {'max_depth': (1, 20), 'learning_rate': (0.01, 0.1)}
    regression(TrainTestData, ModelParams, HyperParamsRange)



start_time = time()

main()

passed_time = time() - start_time
print(f'Time: {passed_time/60} min')