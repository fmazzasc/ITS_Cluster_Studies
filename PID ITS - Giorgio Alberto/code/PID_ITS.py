import pickle
from prompt_toolkit import Application
import yaml
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

from ROOT import TH2F, TCanvas, TH1F, gStyle, kBird, gPad, gROOT, TFile
from ROOT_graph import set_obj_style

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


# Filtering
#_______________________________________

def filtering(ApplicationDf, part_name='all', label=True, beta=True):
    """
    From the full datatframe, creates a new one saving only data relative to a chosen particle (filtering  with instructions in its tag).
    The new dataframe will have a label column where its particle species is specified and a beta column where beta is defined.

    Parameters:
    - ApplicationDf: full dataframe
    - part_name: name of the particle to filter

    Returns:
    a list of reduced dataframes
    """
    
    if part_name == 'all':  part_name = names

    dfs = [ApplicationDf.query(tag_dict[part]).reset_index(drop=True) for part in part_name]
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
    for xi in x:    hist.Fill(xi)
    
    if normalized:  set_obj_style(hist, x_label=x_label, y_label='Normaliazed counts', line_color=38)
    else:           set_obj_style(hist, x_label=x_label, y_label='Counts', line_color=38)

    if normalized:  hist.DrawNormalized('hist')
    else:           hist.Draw()
    
    if save:        canvas.SaveAs(f'{filename}.root')

def multiple_hist(dfs, column, plot_specifics, filename, hist_names=None, x_label=None):
    """
    Saves multiple histigrams (different particle species, same variable) on the same file. You can reiterate for different variables as well.
    
    Parameters
    --------------------------------------------
    - dfs: list of dataframes whose columns will fill the histograms (it also works with pd.Series)
    - column: variable you want to draw histograms of
    - hist_names: list of names for each hist
    - plot_specific: list with the following content -> [nbinsx, xlow, xup] 
    - logz: if True, the z-axis will be in log scale

    You could also pass a list of lists as dfs. If that is the case, please input hist_names and x_label. YOu should anyway pass a nonempty list for columns. Any single elements inside of it will be fine
    - hist_names: list of names for each of the histograms
    - x_label: label of the x axis 
    """
    
    [nbinsx, xlow, xup] = plot_specifics
    file = TFile(f'{filename}{column}.root', 'recreate')

    for i, df in enumerate(dfs):
        if type(df) == pd.DataFrame:    
            if 'label' in df.columns:   hist_name = f'{df.label[0]}'
        elif hist_names != None:        hist_name = hist_names[i]
        else:                           hist_name = f'{i}'

        hist = TH1F(hist_name, hist_name, nbinsx, xlow, xup)
        if type(df) == pd.DataFrame:    
            for x in df[column]:    hist.Fill(x)
        else:                           
            for x in df:            hist.Fill(x)

        set_obj_style(hist, title=hist_name, x_label=column, y_label="Counts")
        
        file.cd()
        hist.Write()
        del hist
    
    print(f"ROOT file {filename}{column}.root has been created")
    file.Write()
    file.Close()
    del file

def density_scatter(x, y, filename, plot_specifics, title=''): 
    """
    
    Parameters:
    - plot_specific: list with the following content -> [x_label, y_label, nbinsx, xlow, xup, nbinsy, ylow, yup]
    - logz: if True, the z-axis will be in log scale
    """
    
    [x_label, y_label, nbinsx, xlow, xup, nbinsy, ylow, yup] = plot_specifics
    file = TFile(f'{filename}.root', 'recreate')

    scatter_plot = TH2F('scatter_plot', '', nbinsx, xlow, xup, nbinsy, ylow, yup)
    for (xi, yi) in zip(x, y):  scatter_plot.Fill(xi, yi)
    set_obj_style(scatter_plot, title=title, x_label=x_label, y_label=y_label)
    gStyle.SetPalette(kBird)
    
    file.cd()
    scatter_plot.SetDrawOption('COLZ1')
    scatter_plot.Write()
    file.Write()
    file.Close()
    print(f"ROOT file {filename}.root has been created")

    del file, scatter_plot

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

def augmentation_fine(df, mother, daughter, pmin, pmax):
    """
    This function performs data augmentation, generating new data for the daughter species from the pre-existing data of the mother species.

    Parameters
    ----------------------------------
    - df: full dataframe of already identified particles
    - mother: label of the mother species
    - daughter: label of the daughter species
    - pmin, pmax: momentum range to perform the data augmentation in
    """

    mass_mother = mass_dict[mother]
    mass_daughter = mass_dict[daughter]

    betamin = pmin / mass_mother 
    betamax = pmax / mass_mother 
    mother_to_augm = df.query(f'label == "{mother}" and {betamin} <= beta < {betamax}')

    n_mother = len(df.query(f'label == "{mother}" and {pmin} <= p < {pmax}'))
    n_daughter = len(df.query(f'label == "{daughter}" and {pmin} <= p < {pmax}'))

    if n_mother < n_daughter:       return 0
    else:
        n_sample = min(n_mother-n_daughter, len(mother_to_augm))
        augm_daughter = mother_to_augm.sample(n_sample)

        augm_daughter['p'] = augm_daughter['p'] * mass_daughter / mass_mother
        augm_daughter['label'] = daughter
        augm_daughter['copy'] = 1

        return augm_daughter

    


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

def Delta(model, X, y, absolute=False):
    pred = model.predict(X)
    if absolute:    return abs(y - pred)/y
    else:           return abs(y - pred)/y

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

def plot_score(X, y, model, x_label, plot_specifics, x=None, filename='', logy=False, absolute=True):
    """
    Plot a prediction scoring variable (defined as (true-predicted)/true) vs a chosen variable from X columns.

    Parameters:
    - model: model (or pipeline) used for the predictions
    - plot_specifics: list with the following entries [nbinsx, xlow, xup, nbinsy, ylow, yup]
    """

    delta = Delta(model, X, y, absolute=absolute)

    plot_spec = [x_label, '#Delta'] + plot_specifics
    if x == None:   density_scatter(y, delta, f'{filename}_score', plot_spec)
    else:           density_scatter(x, delta, f'{filename}_score', plot_spec)


def plot_score_train(TrainTestData, RegressionColumns, model, x_label, plot_specifics, x_train=None, x_test=None, filename='', logx=False, logy=False, absolute=True):
    """
    Plot a prediction scoring variable (defined as (true-predicted)/true) vs a chosen variable from X columns.

    Parameters:
    - TrainTestData: list [X_train, y_train, X_test, y_test] that will be used for training and test (Xs still have the column passed to the ys)
    - model: model (or pipeline) used for the predictions
    - plot_specifics: list with the following entries [nbinsx, xlow, xup, nbinsy, ylow, yup]
    - abs: choose if the delta parameter is defined as the absolute value or not
    """

    X_train, y_train, X_test, y_test = TrainTestData

    delta_train = Delta(model, X_train[RegressionColumns], y_train, absolute=absolute)
    delta_test = Delta(model, X_test[RegressionColumns], y_test, absolute=absolute)

    plot_spec = [x_label, '#Delta'] + plot_specifics
    if x_train == None:     density_scatter(y_train, delta_train, f'{filename}_score_scatter_train', plot_spec, title='Score scatter train')
    else:                   density_scatter(x_train, delta_train, f'{filename}_score_scatter_train', plot_spec, title='Score scatter train')

    if x_test == None:      density_scatter(y_test, delta_test, f'{filename}_score_scatter_test', plot_spec, title='Score scatter test')
    else:                   density_scatter(x_test, delta_test, f'{filename}_score_scatter_test', plot_spec, title='Score scatter test')

    # no column will be used, since delta_train, delta_test are not dfs.
    multiple_hist([delta_train, delta_test], ' ', plot_specifics[:3], f'{filename}_score_hist', hist_names=['Train', 'Test'], x_label='#Delta')







def data_prep(config):
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
    - ApplicationDf: dataframe that will be used for the application. This is the original dataframe, with some new columns added. Will be created at the beginning and then saved in a file and returned.
    """

    # Upload from data file and config file
    ApplicationDf = pd.read_parquet(config['input']['data'])

    do_plots = config['plots']['do_plots']
    hist_spec = config['plots']['plot_spec_hist']
    scat_spec = config['plots']['plot_spec_scat']
    output_dir = config['output']['data_visual_dir']

    test_frac = config['data_prep']['test_frac']
    seed_split = config['data_prep']['seed_split']

    do_augm = config['data_prep']['do_augm']
    beta_flat = config['training']['beta_flat']
    save_data = config['data_prep']['save_data']
    save_data_dir = config['output']['save_data_dir']

    if do_augm and beta_flat:   sub_dir = '/augm_betaflat'
    elif do_augm:               sub_dir = '/augm'
    elif beta_flat:             sub_dir = '/betaflat'
    else:                       sub_dir = '/no_options'






    # define some new columns
    ApplicationDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
    ApplicationDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
    ApplicationDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)

    for part in names:
        ApplicationDf[f'nSigma{part}Abs'] = abs(ApplicationDf[f'nSigma{part}'])
    
    # Filtering (and evaluating beta)
    #__________________________________
    ApplicationDf.query('p <= 50', inplace=True)
    df_Deu, df_P, df_K, df_Pi = filtering(ApplicationDf)

    filt_dfs = [df_Deu, df_P, df_K, df_Pi]
    total_df = pd.concat(filt_dfs)
    pi_and_p_df = pd.concat([df_P, df_Pi])
    

    # Data Visualization
    #_________________________________
    if do_plots:
        print('\nData Visualization...')

        for var in hist_spec:   multiple_hist(filt_dfs, var, hist_spec[var], f'{output_dir}/')

        plot_x_scat = config['plots']['plot_x_scat']
        plot_y_scat = config['plots']['plot_y_scat']
        plot_spec_scat = config['plots']['plot_spec_scat']

        for x, y, scat_spec in zip(plot_x_scat, plot_y_scat, plot_spec_scat):
            for name, df in zip(names, filt_dfs): 
                density_scatter(df[x], df[y], f'{output_dir}/{y}_vs_{x}_{name}', scat_spec, title=f'{y}_{name}')
            density_scatter(total_df[x], total_df[y], f'{output_dir}/{y}_vs_{x}_total', scat_spec, title=f'{y}_total')

        density_scatter(ApplicationDf['p'], ApplicationDf['dedx'], f'{output_dir}/dedx_all', plot_spec_scat[0], title='dedx_all')
        density_scatter(ApplicationDf['p'], ApplicationDf['meanClsize'], f'{output_dir}/meanClsize_all', plot_spec_scat[2], title='meanClsize_all')
        density_scatter(pi_and_p_df['p'], pi_and_p_df['meanClsize'], f'{output_dir}/meanClsize_P_and_Pi', plot_spec_scat[2], title='meanClsize_P_and_Pi')

        # Check hypotesis 
        check_dfs = []
        for name in names:                          check_dfs.append( total_df.query(f"label == '{name}' and -0.1 <= SnPhiL0 <= 0.1 and -0.1 <= tgL <= 0.1") )
        for df, name in zip(check_dfs, names):      density_scatter(df['beta'], df['ClSizeL0'], f'{output_dir}/check_on_ClSizeL0_{name}', plot_spec_scat[3], title=f'check_on_ClSizeL0_{name}')

    
    # Data Preprocessing (eliminate negative cl.sizes, apply beta flat, apply data augmentation, split dataframes)
    #_________________________________
    print('\nData Preprocessing...')
    
    # negative values are intended to be nans
    for i in range(7):
        total_df[f'ClSizeL{i}'] = np.where(total_df[f'ClSizeL{i}'] < 0, 0, total_df[f'ClSizeL{i}'])

    TrainSet, TestSet, yTrain, yTest = train_test_split(total_df, total_df.beta, test_size=test_frac ,random_state=seed_split)


    # Data augmentation
    #_________________________________

    if do_augm:
        print('\nData Augmentation...')
        TrainSet['copy'] = 0

        to_augm = config['data_prep']['to_augm']
        mothers = config['data_prep']['mothers']
        p_ranges = config['data_prep']['p_ranges']

        augm = []
        for daughter, mother, p_range in zip(to_augm, mothers, p_ranges):
            pmin, pmax = p_range[0], p_range[1]
            augm_df = augmentation_fine(TrainSet, mother, daughter, pmin, pmax)
            if type(augm_df) != int:  augm.append(augm_df)
        augm.append(TrainSet)
        TrainSet = pd.concat(augm)
        yTrain = TrainSet['beta']

        for daughter in to_augm:    
            len_daughter = len(TrainSet.query(f"label == '{daughter}' and copy == 1"))
            print(f'Augmented {daughter}: {len_daughter}')

        augmented_dfs = []
        for name in names:
            augmented_df = TrainSet.query(f"label == '{name}'")  
            augmented_dfs.append(augmented_df)




        # Plots after augmentation
        if do_plots:
            plot_x_scat = config['plots']['plot_x_scat']
            plot_y_scat = config['plots']['plot_y_scat']
            plot_spec_scat = config['plots']['plot_spec_scat']
    
            for x, y, scat_spec in zip(plot_x_scat, plot_y_scat, plot_spec_scat):
                for name, df in zip(names, augmented_dfs): 
                    density_scatter(df[x], df[y], f'{output_dir}{sub_dir}/{y}_vs_{x}_{name}_augm', scat_spec, title=f'{y}_{name}_augm')
                density_scatter(TrainSet[x], TrainSet[y], f'{output_dir}{sub_dir}/{y}_vs_{x}_all_augm', scat_spec, title=f'{y}_all_augm')



    # Beta flat weights
    #_________________________________

    betamins = config['data_prep']['betamins']
    betamaxs = config['data_prep']['betamaxs']
    weights = []

    for betamin, betamax in zip(betamins, betamaxs):
        weights.append( len(TrainSet.query(f'{betamin} <= beta < {betamax}'))/len(TrainSet) )

    conditions = [(TrainSet['beta'] >= betamin) & (TrainSet['beta'] < betamax) for betamin, betamax in zip(betamins, betamaxs)]
    for weight in weights:  weight = 1./weight
    TrainSet['beta_weight'] = np.select(conditions, weights)


    # Save prepared data
    #___________________________________

    data_conf = ''
    if do_augm:     data_conf += '_augm'
    if beta_flat:   data_conf += '_betaflat'

    if save_data:
        dfTrainSet, dfyTrain, dfTestSet, dfyTest = pd.DataFrame(TrainSet), pd.DataFrame(yTrain), pd.DataFrame(TestSet), pd.DataFrame(yTest)

        dfTrainSet.to_parquet(f'{save_data_dir}/TrainSet{data_conf}.parquet.gzip')
        dfyTrain.to_parquet(f'{save_data_dir}/yTrain{data_conf}.parquet.gzip')
        dfTestSet.to_parquet(f'{save_data_dir}/TestSet{data_conf}.parquet.gzip')
        dfyTest.to_parquet(f'{save_data_dir}/yTest{data_conf}.parquet.gzip')

        ApplicationDf.to_parquet(f'{save_data_dir}/ApplicationDf{data_conf}.parquet.gzip')

    
    # Return complete dataframe ready for ML 
    #____________________________________
    
    TrainTestData = [TrainSet, yTrain, TestSet, yTest]
    return TrainTestData, ApplicationDf
    

    





def regression(TrainTestData, config):
    """
    Parameters:
    ----------------------
    - TrainTestData: list [X_train, y_train, X_test, y_test] that will be used for training and test
    - config: .yml configuration file
    """

    X_train, y_train, X_test, y_test = TrainTestData

    RegressionColumns = config['training']['RegressionColumns']
    model_choice = config['training']['model']
    ModelParams = config['training']['ModelParams']

    do_opt = config['training']['do_opt']
    HyperParamsRange = config['training']['HyperParamsRange']
    early_stop = config['training']['early_stop']

    do_augm = config['data_prep']['do_augm']
    beta_flat = config['training']['beta_flat']

    output_dir = config['output']['ml_dir']
    save_model = config['training']['save_model']

    options = ''
    if do_augm:     options += '_augm'
    if beta_flat:   options += '_betaflat_'
    
    if do_augm and beta_flat:   sub_dir = '/augm_betaflat'
    elif do_augm:               sub_dir = '/augm'
    elif beta_flat:             sub_dir = '/betaflat'
    else:                       sub_dir = '/no_options'
    

    # Model definition
    #__________________________________
    if model_choice=='xgboost':     model = xgb.XGBRegressor()

    # Optuna optimization
    #__________________________________
    
    if do_opt:
        model_handler = ModelHandler(model, RegressionColumns)
        model_handler.set_model_params(ModelParams)

        print('Initialize Optuna hyper-parameters optimization...')
        with alive_bar(title='Hyper-Parameters optimization') as bar:
            if early_stop:  study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, Delta_score, direction='minimize', callbacks=[callback], timeout=600)
            else:           study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, Delta_score, direction='minimize', timeout=600)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(f'{output_dir}{sub_dir}/plot_optimization_history{options}.png')

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(f'{output_dir}{sub_dir}/plot_param_importances{options}.png')

        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(f'{output_dir}{sub_dir}/plot_parallel_coordinate{options}.png')

        fig = optuna.visualization.plot_contour(study)
        fig.write_image(f'{output_dir}{sub_dir}/plot_contour{options}.png')

        HyperParams = study.best_trial.params

    else:   HyperParams = ModelParams

    



    # Model evaluation
    #__________________________________
    if model_choice=='xgboost':

        print('Global model...')
        model_reg = xgb.XGBRegressor(**HyperParams)
        #pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline = Pipeline(steps=[('model', model_reg)])

        with alive_bar(title='Training all') as bar:
            if beta_flat:       pipeline.fit(X_train[RegressionColumns], y_train, model__sample_weight=X_train['beta_weight'])
            else:               pipeline.fit(X_train[RegressionColumns], y_train)
        
        plot_specifics = [1000, 0, 1, 2000, -0.5, 1.5]
        plot_score_train(TrainTestData, RegressionColumns, pipeline, x_label='#beta', plot_specifics=plot_specifics, filename=f'{output_dir}{sub_dir}/{options}')

        for name in names:
            print(f'\nModeling {name}...')

            train = pd.concat([X_train, y_train])
            X_train_name = train.query(f"label == '{name}'")
            y_train_name = X_train_name['beta']

            test = pd.concat([X_test, y_test])
            X_test_name = test.query(f"label == '{name}'")
            y_test_name = X_test_name['beta']
            
            with alive_bar(title=f'Training {name}') as bar:
                if beta_flat:   pipeline.fit(X_train_name[RegressionColumns], y_train_name, model__sample_weight=X_train_name['beta_weight'])
                else:           pipeline.fit(X_train_name[RegressionColumns], y_train_name)

            TestTrainData_name = X_train_name, y_train_name, X_test_name, y_test_name

            plot_score_train(TestTrainData_name, RegressionColumns, pipeline, x_label='#beta', plot_specifics=plot_specifics, filename=f'{output_dir}{sub_dir}/{name}{options}')

    
    # Save model in pickle
    #_______________________________

    if save_model:

        print('\nSaving regressor model...')
        with open(f'{output_dir}/RegressorModel_{model_choice}{options}.pickle', 'wb') as output_file:
            pickle.dump(model_reg, output_file)
        print('Model saved.\n')


    return model_reg




def application(ApplicationDf, config, model):

    RegressionColumns = config['training']['RegressionColumns']
    output_dir = config['output']['final_dir']

    do_augm = config['data_prep']['do_augm']
    beta_flat = config['training']['beta_flat']

    X_application = ApplicationDf[RegressionColumns]
    preds = model.predict(X_application)               # beta

    output_file = f'{output_dir}/beta_vs_p'
    if do_augm:     output_file += '_augm'
    if beta_flat:   output_file += '_betaflat'

    density_scatter(ApplicationDf['p'], preds, output_file, title='beta_vs_p_ final')




def main():

    # Configuration File
    #_________________________________
    with open('/Users/giogi/Desktop/Stage INFN/PID ITS/ITS_Cluster_Studies/PID ITS - Giorgio Alberto/configs/config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    
    # Data Preparation
    #__________________________________
    skip_data_prep = config['data_prep']['skip_data_prep']

    if not skip_data_prep:   TrainTestData, ApplicationDf = data_prep(config)
    else:
        prep_data_loc = config['data_prep']['prep_data']
        TrainTestData = []
        for loc in prep_data_loc:   TrainTestData.append(pd.read_parquet(f'{loc}'))
        appl_loc = config['data_prep']['appl_data']
        ApplicationDf = pd.read_parquet(f'{appl_loc}')

    
    # Training
    #__________________________________
    skip_training = config['training']['skip_training']
    
    if not skip_training:   Model = regression(TrainTestData, config)
    

    # Application
    #__________________________________

    if skip_training:       
        model_loc = config['application']['model_loc']            
        Model = pickle.load(open(f'{model_loc}', "rb"))
        Model.training_columns = config['training']['RegressionColumns']
    application(ApplicationDf, config, Model)

    del TrainTestData



start_time = time()

main()

passed_time = time() - start_time
print(f'Time: {passed_time/60} min')