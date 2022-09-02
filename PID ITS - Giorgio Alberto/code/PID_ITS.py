import pickle
import yaml
import pandas as pd
import numpy as np
from math import floor, ceil, sqrt
import matplotlib.pyplot as plt

from alive_progress import alive_bar
from time import time

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from flaml import AutoML


from hipe4ml.model_handler import ModelHandler
from hipe4ml import plot_utils
import optuna

from ROOT import TH2F, TCanvas, TH1F, gStyle, kBird, gROOT, TFile
from ROOT_graph import set_obj_style


gROOT.SetBatch()

# Tags
#_____________________________________
tag_Deu = 'nSigmaDeuAbs < 1 and nSigmaPAbs > 5 and nSigmaKAbs > 5 and nSigmaPiAbs > 5 and p <= 1.2'
tag_P = 'nSigmaPAbs < 1 and nSigmaKAbs > 3 and nSigmaDeuAbs > 3 and p <= 0.7'
tag_K = 'nSigmaKAbs < 1 and nSigmaPiAbs > 3 and nSigmaPAbs > 3 and p <= 0.7'
tag_Pi = 'nSigmaPiAbs < 1 and nSigmaKAbs > 3 and p <= 0.7'
tag_E = 'nSigmaPi > 5'


# Masses
#_____________________________________
mass_Deu = 1.8756
mass_P =  0.93827200
mass_K = 0.4937
mass_Pi = 0.13957000
mass_e = 0.000511

names = ['Deu', 'P', 'K', 'Pi']

tag_dict = dict(zip(names, [tag_Deu, tag_P, tag_K, tag_Pi, tag_E]))
mass_dict = dict(zip(names, [mass_Deu, mass_P, mass_K, mass_Pi, mass_e]))




# Filtering
#_______________________________________



def filtering(full_df, part, tag=None, mass=None, label=True):
    """
    From the full datatframe, creates a new one saving only data relative to a chosen particle (filtering  with instructions in its tag).
    The new dataframe will have a label column where its particle species is specified and a beta column where beta is defined.

    Parameters:
    - full_df: full dataframe
    - part: name of the particle to filter
    - tag: tag used by pd.DataFrame.query for the selection
    - mass: mass of the particle
    - label: if true, a label column with the particle name will be appended to the dataframe

    Returns:
    the filtered dataframe
    """
    
    

    if tag != None:       df = full_df.query(tag, inplace=False).reset_index(drop=True)
    else:                 df = full_df.query(f"label == '{part}'").reset_index(drop=True)
 
    if label:             df['label'] = part
    if mass != None:      df.eval(f'beta = p / sqrt( {mass}**2 + p**2)', inplace=True)

    return df



# Plot functions
#________________________________________
def hist(x, filename, plot_specifics, normalized=True):
    """
    
    Parameters:
    - plot_specific: list with the following content -> [x_label, nbinsx, xlow, xup] 
    """
    
    x_label, nbinsx, xlow, xup = plot_specifics
    file = TFile(f'{filename}.root', 'recreate')    

    hist = TH1F('scatter_plot', '', nbinsx, xlow, xup)
    for xi in x:    hist.Fill(xi)
    
    if normalized:  set_obj_style(hist, x_label=x_label, y_label='Normaliazed counts', line_color=38)
    else:           set_obj_style(hist, x_label=x_label, y_label='Counts', line_color=38)

    file.cd()
    if normalized:  hist.DrawNormalized('hist')
    else:           hist.Draw()

    print(f"ROOT file {filename}.root has been created")
    file.Write()
    file.Close()
    del hist, file
    
def hist_canvas(x, filename, canvas, plot_specifics, pad=None, normalized=True, save=True):
    """
    
    Parameters:
    - plot_specific: list with the following content -> [x_label, nbinsx, xlow, xup] 
    """
    
    x_label, nbinsx, xlow, xup = plot_specifics
    if pad != None: canvas.cd(pad)
    else:           canvas.cd()     

    hist = TH1F('scatter_plot', '', nbinsx, xlow, xup)
    for xi in x:    hist.Fill(xi)
    
    if normalized:  set_obj_style(hist, x_label=x_label, y_label='Normaliazed counts', line_color=38)
    else:           set_obj_style(hist, x_label=x_label, y_label='Counts', line_color=38)

    if normalized:  hist.DrawNormalized('hist')
    else:           hist.Draw()
    
    if save:        canvas.SaveAs(f'{filename}.root')

def multiple_hist(dfs, column, plot_specifics, filename, hist_names=None):
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
    
    nbinsx, xlow, xup = plot_specifics
    file = TFile(f'{filename}{column}.root', 'recreate')

    for i, df in enumerate(dfs):
        if hist_names != None:        hist_name = hist_names[i]
        #elif type(df) == pd.DataFrame:    
        #    if 'particle' in df.columns:   hist_name = f'{df.particle.iloc[0]}'
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

def density_scatter(x, y, filename, plot_specifics, title='', weights=pd.Series([]), normalized=False): 
    """
    
    Parameters:
    - plot_specific: list with the following content -> [x_label, y_label, nbinsx, xlow, xup, nbinsy, ylow, yup]
    """
    
    [x_label, y_label, nbinsx, xlow, xup, nbinsy, ylow, yup] = plot_specifics
    file = TFile(f'{filename}.root', 'recreate')

    scatter_plot = TH2F('scatter_plot', '', nbinsx, xlow, xup, nbinsy, ylow, yup)
    if not weights.empty:   
        for (xi, yi, wi) in zip(x, y, weights):  scatter_plot.Fill(xi, yi, wi)
    else:                   
        for (xi, yi) in zip(x, y):  scatter_plot.Fill(xi, yi)
    set_obj_style(scatter_plot, title=title, x_label=x_label, y_label=y_label)
    gStyle.SetPalette(kBird)
    
    file.cd()
    scatter_plot.SetDrawOption('COLZ1')
    if normalized:  scatter_plot.Scale(1./ scatter_plot.Integral(), 'width')
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
            hist_canvas(x, '', canvas, plot_specifics, pad=i, save=False)
    
    if plot == 'scatter_density':
        for i, (x, y) in enumerate(zip(xs, ys)):
            density_scatter(x, y, '', canvas, plot_specifics, pad=i, save=False)
    
    canvas.SaveAs(f'{filename}.root')

        







# Data Augmentation Functions
#_______________________________________

def equal(df, column):
    """
    From a given dataframe, finds the minimum number of elements having unique values in a column. Discards elements
    having different unique values in that column until their size matches that minimum number.

    Parameters
    ----------------
    df: dataframe
    column: column where the unique values are stored

    Returns
    ----------------
    "Filtered" dataframe
    """
    
    sizes = []
    for item in df[column].unique():  sizes.append(len(df.query(f"{column} == '{item}'")))
    min_size = min(sizes)

    new_df = pd.DataFrame()
    for item in df[column].unique():  new_df = pd.concat([new_df, df.query(f"{column} == '{item}'").iloc[:min_size]], sort=False)

    return new_df

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

    betamin = pmin / sqrt(mass_mother**2 + pmin**2) 
    betamax = pmax / sqrt(mass_mother**2 + pmax**2) 
    mother_to_augm = df.query(f'label == "{mother}" and {betamin} <= beta < {betamax}')

    # Thic check should be included when working without weights
    #n_mother = len(df.query(f'label == "{mother}" and {pmin} <= p < {pmax}'))
    #n_daughter = len(df.query(f'label == "{daughter}" and {pmin} <= p < {pmax}'))
    #
    #
    #if n_mother < n_daughter:   return 0
    #else:   n_sample = min(n_mother-n_daughter, len(mother_to_augm))
    
    n_sample = len(mother_to_augm)
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

def Delta(model, X, y, absolute=True):
    pred = model.predict(X)
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

def plot_score(X, y, RegressionColumns, model, x_label, plot_specifics, x=pd.Series(), filename='', absolute=False):
    """
    Plot a prediction scoring variable (defined as (true-predicted)/true) vs a chosen variable from X columns.

    Parameters:
    - model: model (or pipeline) used for the predictions
    - plot_specifics: list with the following entries [nbinsx, xlow, xup, nbinsy, ylow, yup]
    """

    delta = Delta(model, X[RegressionColumns], y, absolute=absolute)

    plot_spec = [x_label, '#Delta'] + plot_specifics
    if x.empty:     density_scatter(y, delta, f'{filename}_score_scatter', plot_spec)
    else:           density_scatter(x, delta, f'{filename}_score_scatter', plot_spec)

    plot_spec_hist = [f'#Delta / {x_label}'] + plot_specifics[3:]
    hist(delta/y, f'{filename}_score_hist', plot_spec_hist)

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

    delta_train = Delta(model, X_train[RegressionColumns], y_train, absolute=absolute)
    delta_test = Delta(model, X_test[RegressionColumns], y_test, absolute=absolute)

    plot_spec = [x_label, '#Delta'] + plot_specifics
    if type(x_train) == pd.Series and x_train.empty:       
        density_scatter(y_train, delta_train, f'{filename}_score_scatter_train', plot_spec, title='Score scatter train')
    else:                   density_scatter(x_train, delta_train, f'{filename}_score_scatter_train_x', plot_spec, title='Score scatter train')

    if type(x_test) == pd.Series and x_test.empty:        
        density_scatter(y_test, delta_test, f'{filename}_score_scatter_test', plot_spec, title='Score scatter test')
    else:                   density_scatter(x_test, delta_test, f'{filename}_score_scatter_test_p', plot_spec, title='Score scatter test')

    # no column will be used, since delta_train, delta_test are not dfs.
    multiple_hist([delta_train/y_train, delta_test/y_test], '', plot_specifics[3:], f'{filename}_score_hist', hist_names=['Train', 'Test'])







def data_prep(config):
    """"
    Data preparation function. The full data will be uploaded and dataframes for single particle species will be created according to the tag rules. 
    Histograms for some of the column variables will be created. A data augmentation process will be added.

    Parameters
    --------------------------------------------------
    - config: .yml configuration file

    Returns
    --------------------------------------------------
    - TrainTestData: list [X_train, y_train, X_test, y_test] that will be used for training and test (Xs still have the column passed to the ys)
    - ApplicationDf: dataframe that will be used for the application. This is the original dataframe, with some new columns added. Will be created at the beginning and then saved in a file and returned.
    """

    # Upload data and settings from configuration file 
    #_________________________________________________

    isV0 = config['input']['isV0']
    ext_appl = config['input']['ext_appl']

    RegressionDf = pd.read_parquet(config['input']['data'])
    if ext_appl:    ApplicationDf = pd.read_parquet(config['input']['appl_data'])

    particle_dict = config['output']['particle'] 

    do_plots = config['plots']['do_plots']
    vars_to_plot = config['plots']['vars_to_plot']
    hist_spec = config['plots']['plot_spec_hist']
    scat_spec = config['plots']['plot_spec_scat']
    output_dir = config['output']['data_visual_dir']

    test_frac = config['data_prep']['test_frac']
    seed_split = config['data_prep']['seed_split']

    seven_hits = config['data_prep']['seven_hits']  # only consider candidates with hits on all the layers

    do_augm = config['data_prep']['do_augm']
    beta_flat = config['training']['beta_flat']
    beta_p_flat = config['training']['beta_p_flat']
    do_equal = config['data_prep']['do_equal']
    save_data = config['data_prep']['save_data']
    save_data_dir = config['output']['save_data_dir']

    if isV0:                    
        output_dir += '/V0'
        save_data_dir += '/V0'
    else:                       
        output_dir += '/TPC'
        save_data_dir += '/TPC'

    if do_augm and beta_p_flat: output_dir += '/augm'
    elif do_augm and beta_flat: output_dir += '/augm_betaflat'
    elif do_augm:               output_dir += '/augm'
    elif beta_flat:             output_dir += '/betaflat'
    elif beta_p_flat:           output_dir += '/beta_pflat'
    elif do_equal:              output_dir += '/equal'
    else:                       output_dir += '/no_options'

    options = ''
    if do_augm:                 options += '_augm'
    if beta_flat:               options += '_betaflat_'
    if beta_p_flat:             options += '_beta_pflat_'
    if do_equal:                options += '_equal_'






    # define some new columns
    RegressionDf.eval('meanClsize = (ClSizeL0+ClSizeL1+ClSizeL2+ClSizeL3+ClSizeL4+ClSizeL5+ClSizeL6)/7', inplace=True)
    RegressionDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
    RegressionDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
    RegressionDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)
    
    if isV0:    RegressionDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)  


    def get_mass(n): 
        return mass_dict[particle_dict[n]]

    if ext_appl:
        ApplicationDf.eval('meanClsize = (ClSizeL0+ClSizeL1+ClSizeL2+ClSizeL3+ClSizeL4+ClSizeL5+ClSizeL6)/7', inplace=True)
        ApplicationDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
        ApplicationDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
        ApplicationDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)  
        if isV0:    ApplicationDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
        
        
    for part in particle_dict.values():  RegressionDf[f'nSigma{part}Abs'] = abs(RegressionDf[f'nSigma{part}'])
    if ext_appl:    
        for part in particle_dict.values():  ApplicationDf[f'nSigma{part}Abs'] = abs(RegressionDf[f'nSigma{part}'])
    
    # Splitting, filtering (and evaluating beta)
    #__________________________________
    if isV0:
        RegressionDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)
        if ext_appl:    ApplicationDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)

        RegressionDf.eval('label = particle', inplace=True)
        ApplicationDf.eval('label = particle', inplace=True)
        for number, name in particle_dict.items():  
            RegressionDf['label'].mask(RegressionDf['particle'] == number, name, inplace=True)
            ApplicationDf['label'].mask(ApplicationDf['particle'] == number, name, inplace=True)

    else:   RegressionDf.query('p <= 50', inplace=True)

    # redefine K
    #RegressionDf['label'].mask(RegressionDf['nSigmaKAbs'] < 1, 'K', inplace=True)
    #if isV0:    RegressionDf['particle'].mask(RegressionDf['nSigmaKAbs'] < 1, 2, inplace=True)
#
    #if ext_appl:    
    #    ApplicationDf['label'].mask(ApplicationDf['nSigmaKAbs'] < 1, 'K', inplace=True)
    #    if isV0:    ApplicationDf['particle'].mask(ApplicationDf['nSigmaKAbs'] < 1, 2, inplace=True)

    
    TrainSet, TestSet, yTrain, yTest = train_test_split(RegressionDf, RegressionDf.p, test_size=test_frac, random_state=seed_split)

    if isV0:
        dfs_train = filtering(TrainSet, tag=False, label=False)
        dfs_test = filtering(TestSet, tag=False, label=False)

        # evaluate beta in ApplicationDf
        appl_list = filtering(ApplicationDf, tag=False, label=False)
        ApplicationDf = pd.concat(appl_list)

    else:
        dfs_train = [filtering(TrainSet, name, tag_dict[name], mass_dict[name]) for name in particle_dict.values()]
        dfs_test = [filtering(TestSet, name, tag_dict[name], mass_dict[name]) for name in particle_dict.values()]


    # Data Visualization
    #_________________________________
    if do_plots:
        print('\nData Visualization...')

        filt_dfs = [pd.concat([df_train, df_test], ignore_index=True) for df_train, df_test in zip(dfs_train, dfs_test)]
        total_df = pd.concat(filt_dfs)

        for var in hist_spec:   multiple_hist(filt_dfs, var, hist_spec[var], f'{output_dir}/')

        plot_x_scat = config['plots']['plot_x_scat']
        plot_y_scat = config['plots']['plot_y_scat']
        plot_spec_scat = config['plots']['plot_spec_scat']

        for x, y, scat_spec in zip(plot_x_scat, plot_y_scat, plot_spec_scat):
            for name, df in zip(names, filt_dfs): 
                density_scatter(df[x], df[y], f'{output_dir}/{y}_vs_{x}_{name}', scat_spec, title=f'{y}_{name}')
            density_scatter(total_df[x], total_df[y], f'{output_dir}/{y}_vs_{x}_total', scat_spec, title=f'{y}_total')

        density_scatter(RegressionDf['p'], RegressionDf['dedx'], f'{output_dir}/dedx_all', plot_spec_scat[0], title='dedx_all')
        density_scatter(RegressionDf['p'], RegressionDf['meanClsize'], f'{output_dir}/meanClsize_all', plot_spec_scat[2], title='meanClsize_all')
        
        # Check hypotesis 
        check_dfs = []
        for name in names:                          check_dfs.append( total_df.query(f"label == '{name}' and -0.1 <= SnPhiL0 <= 0.1 and -0.1 <= tgL <= 0.1") )
        for df, name in zip(check_dfs, names):      density_scatter(df['beta'], df['ClSizeL0'], f'{output_dir}/check_on_ClSizeL0_{name}', plot_spec_scat[3], title=f'check_on_ClSizeL0_{name}')

        # Correlation between dataframe features
        #___________________________________

        CorrMatrixFig = plot_utils.plot_corr(filt_dfs, vars_to_plot, names)
        for Fig, name in zip(CorrMatrixFig, names):
            plt.figure(Fig.number)
            plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
            Fig.savefig(f'{output_dir}/CorrMatrix_{name}{options}.png')
    

    # Data Preprocessing (eliminate negative cl.sizes, apply beta flat, apply data augmentation, split dataframes)
    #_________________________________
    print('\nData Preprocessing...')

    TrainSet = pd.concat(dfs_train)
    TrainSet = TrainSet.sample(frac=1).reset_index(drop=True)

    TestSet = pd.concat(dfs_test)
    TestSet = TestSet.sample(frac=1).reset_index(drop=True)
    if not ext_appl:    ApplicationDf = pd.DataFrame(TestSet)
    
    # negative values are intended to be nans
    if not seven_hits:
        for i in range(7):
            TrainSet[f'ClSizeL{i}'] = np.where(TrainSet[f'ClSizeL{i}'] < 0, np.nan, TrainSet[f'ClSizeL{i}'])
            TestSet[f'ClSizeL{i}'] = np.where(TestSet[f'ClSizeL{i}'] < 0, np.nan, TestSet[f'ClSizeL{i}'])
            ApplicationDf[f'ClSizeL{i}'] = np.where(ApplicationDf[f'ClSizeL{i}'] < 0, np.nan, ApplicationDf[f'ClSizeL{i}'])

    # consider only candidates with seven hits
    else:   
        for i in range(7):
            TrainSet.drop( TrainSet[TrainSet[f'ClSizeL{i}'] < 0].index, inplace=True )
            TestSet.drop( TestSet[TestSet[f'ClSizeL{i}'] < 0].index, inplace=True )
            ApplicationDf.drop( ApplicationDf[ApplicationDf[f'ClSizeL{i}'] < 0].index, inplace=True )

    yTrain = pd.Series(TrainSet['beta'])
    yTest = pd.Series(TestSet['beta'])
    

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
        print()

        augmented_dfs = [TrainSet.query(f"label == '{name}'") for name in names]

        # Plots after augmentation
        if do_plots:
            plot_x_scat = config['plots']['plot_x_scat']
            plot_y_scat = config['plots']['plot_y_scat']
            plot_spec_scat = config['plots']['plot_spec_scat']
    
            for x, y, scat_spec in zip(plot_x_scat, plot_y_scat, plot_spec_scat):
                for name, df in zip(names, augmented_dfs): 
                    density_scatter(df[x], df[y], f'{output_dir}/{y}_vs_{x}_{name}_augm', scat_spec, title=f'{y}_{name}_augm')
                density_scatter(TrainSet[x], TrainSet[y], f'{output_dir}/{y}_vs_{x}_all_augm', scat_spec, title=f'{y}_all_augm')

    # Equal number of candidates (without deuterons)
    #_________________________________

    if do_equal:
        TrainSet = TrainSet.query("label != 'Deu'")
        TrainSet = equal(TrainSet, 'label')
        yTrain = TrainSet['beta']

    # Beta flat weights
    #_________________________________

    if beta_flat:
        with alive_bar(title='Beta flat...') as bar:
        
            betamins = config['data_prep']['betamins']
            betamaxs = config['data_prep']['betamaxs']
            weights = [len(TrainSet.query(f'{betamin} <= beta < {betamax}'))/len(TrainSet) for betamin, betamax in zip(betamins, betamaxs)]

            conditions = [(TrainSet['beta'] >= betamin) & (TrainSet['beta'] < betamax) for betamin, betamax in zip(betamins, betamaxs)]
            n_weights = []
            for weight in weights:
                if weight == 0:     n_weights.append(0)
                else:               n_weights.append(1./weight) 
            TrainSet['beta_weight'] = np.select(conditions, n_weights)

    # Beta and momentum flat weights
    #_________________________________

    if beta_p_flat:
        with alive_bar(title='Beta and momentum flat...') as bar:

            pmins = config['data_prep']['pmins']
            pmaxs = config['data_prep']['pmaxs']
            weights_list = [[len(TrainSet.query(f'label == "{name}" and {pmin} <= p < {pmax}'))/len(TrainSet.query(f'label == "{name}"')) for pmin, pmax in zip(pmins, pmaxs)] for name in particle_dict.values()]
            weights = []
            for list in weights_list:   weights += list

            conditions = []
            for name in particle_dict.values(): conditions += [(TrainSet['label'] == name) & (TrainSet['p'] >= pmin) & (TrainSet['p'] < pmax) for pmin, pmax in zip(pmins, pmaxs)]
            n_weights = []
            for weight in weights:
                if weight == 0: n_weights.append(0)
                else:           n_weights.append(1./weight)
            TrainSet['beta_pweight'] = np.select(conditions, n_weights)

            plot_spec_flat = config['plots']['bp_flat_scat']
            density_scatter(TrainSet['p'], TrainSet['particle'], f'{output_dir}/beta_pflat_momentum', plot_specifics=plot_spec_flat['p'])
            density_scatter(TrainSet['beta_pweight'], TrainSet['particle'], f'{output_dir}/beta_pflat_weights', plot_specifics=plot_spec_flat['weights'])
            density_scatter(TrainSet['p'], TrainSet['particle'], f'{output_dir}/beta_pflat_weighted_momentum', plot_specifics=plot_spec_flat['p'], weights=TrainSet['beta_pweight'])

    
    

    # Save prepared data
    #___________________________________

    data_conf = ''
    if do_augm:     data_conf += '_augm'
    if beta_flat:   data_conf += '_betaflat'
    if beta_p_flat: data_conf += '_beta_pflat'
    if do_equal:    data_conf += '_equal'

    if save_data:
        with alive_bar(title="Saving data...") as bar:
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

    # Upload settings from configuration file 
    #_________________________________________________

    particle_dict = config['output']['particle']
    isV0 = config['input']['isV0']

    RegressionColumns = config['training']['RegressionColumns']
    random_state = config['training']['random_state']
    model_choice = config['training']['model']
    ModelParams = config['training']['ModelParams']

    do_opt = config['training']['do_opt']
    HyperParamsRange = config['training']['HyperParamsRange']
    early_stop = config['training']['early_stop']

    do_augm = config['data_prep']['do_augm']
    beta_flat = config['training']['beta_flat']
    beta_p_flat = config['training']['beta_p_flat']
    do_equal = config['data_prep']['do_equal']

    output_dir = config['output']['ml_dir']
    save_model = config['training']['save_model']

    options = ''
    if do_augm:                 options += '_augm'
    if beta_p_flat:             options += '_beta_pflat_'
    if beta_flat:               options += '_betaflat_'
    if do_equal:                options += '_equal'

    if isV0:                    output_dir += '/V0'
    else:                       output_dir += '/TPC'

    if do_augm and beta_p_flat: output_dir += '/augm'
    elif do_augm and beta_flat: output_dir += '/augm_betaflat'
    elif do_augm:               output_dir += '/augm'
    elif beta_flat:             output_dir += '/betaflat'
    elif beta_p_flat:           output_dir += '/beta_pflat'
    elif do_equal:              output_dir += '/equal'
    else:                       output_dir += '/no_options'

    if do_equal:    names.remove('Deu')
    

    # Model definition
    #__________________________________
    if model_choice=='xgboost':     model = xgb.XGBRegressor(random_state=random_state)
    if model_choice=='automl':      model = AutoML()

    # Optuna optimization
    #__________________________________
    
    if do_opt:
        model_handler = ModelHandler(model, RegressionColumns)
        model_handler.set_model_params(ModelParams)

        print('\nInitialize Optuna hyper-parameters optimization...')
        with alive_bar(title='Hyper-Parameters optimization') as bar:
            if early_stop:  study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, Delta_score, direction='minimize', callbacks=[callback])
            else:           study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, Delta_score, direction='minimize', timeout=300)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(f'{output_dir}/plot_optimization_history{options}.png')

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(f'{output_dir}/plot_param_importances{options}.png')

        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(f'{output_dir}/plot_parallel_coordinate{options}.png')

        fig = optuna.visualization.plot_contour(study)
        fig.write_image(f'{output_dir}/plot_contour{options}.png')

        HyperParams = study.best_trial.params

    else:   HyperParams = ModelParams

    



    # Model evaluation
    #__________________________________
    if model_choice=='xgboost':

        print('\nXGB model...')
        if beta_flat:   print('Beta weights selected...')
        if beta_p_flat: print('Beta and momentum weights selected...')

        model_reg = xgb.XGBRegressor(**HyperParams, tree_method="gpu_hist", random_state=random_state)
        plot_specifics = config['plots']['model_train']
        with alive_bar(title='Training...') as bar:     
            if beta_flat:       model_reg.fit(X_train[RegressionColumns], y_train, sample_weight=X_train['beta_weight'])
            elif beta_p_flat:   model_reg.fit(X_train[RegressionColumns], y_train, sample_weight=X_train['beta_pweight'])
            else:               model_reg.fit(X_train[RegressionColumns], y_train)
        
        TrainSet, TestSet = pd.DataFrame(TrainTestData[0]), pd.DataFrame(TrainTestData[2])
        
        plot_score_train(TrainTestData, RegressionColumns, model_reg, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{output_dir}/{options}', x_test=TestSet['p'], absolute=False)
        plot_score_train(TrainTestData, RegressionColumns, model_reg, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{output_dir}/{options}', absolute=False)
        for key, name in particle_dict.items():

            X_train_name = TrainSet.query(f"label == '{name}'")
            y_train_name = X_train_name['beta']
            X_test_name = TestSet.query(f"label == '{name}'")
            y_test_name = X_test_name['beta']

            TestTrainData_name = X_train_name, y_train_name, X_test_name, y_test_name
            plot_score_train(TestTrainData_name, RegressionColumns, model_reg, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{output_dir}/{name}{options}', x_test=X_test_name['p'], absolute=False)
            plot_score_train(TestTrainData_name, RegressionColumns, model_reg, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{output_dir}/{name}{options}', absolute=False)


    if model_choice=='automl':
        
        print('\nFLAML model...')
        model_reg = AutoML(**HyperParams)
        plot_specifics = config['plots']['model_train']
        with alive_bar(title='Training...') as bar:     model_reg.fit(X_train[RegressionColumns], y_train)

        plot_score_train(TrainTestData, RegressionColumns, model_reg, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{output_dir}/{options}', absolute=False)
        plot_score_train(TrainTestData, RegressionColumns, model_reg, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{output_dir}/{options}', absolute=False)
        for name in particle_dict.values():

            TrainSet, TestSet = pd.DataFrame(TrainTestData[0]), pd.DataFrame(TrainTestData[2])
            X_train_name = TrainSet.query(f"label == '{name}'")
            y_train_name = X_train_name['beta']
            X_test_name = TestSet.query(f"label == '{name}'")
            y_test_name = X_test_name['beta']

            TestTrainData_name = X_train_name, y_train_name, X_test_name, y_test_name
            plot_score_train(TestTrainData_name, RegressionColumns, model_reg, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{output_dir}/{name}{options}', absolute=False)
            plot_score_train(TestTrainData_name, RegressionColumns, model_reg, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{output_dir}/{name}{options}', absolute=False)


    
    # Save model in pickle
    #_______________________________

    if save_model:
        model_out = config['output']['model_out']

        print('\nSaving regressor model...')
        with open(f'{model_out}/RegressorModel_{model_choice}{options}.pickle', 'wb') as output_file:
            pickle.dump(model_reg, output_file)
        print('Model saved.')

    # Feature importance plot
    #________________________________

    if model_choice == 'xgboost':

        FeatureImportance = xgb.plot_importance(model_reg)
        plt.savefig(f'{output_dir}/FeatureImportance_{model_choice}{options}.png')
        plt.close('all')


    return model_reg




def application(ApplicationDf, config, model):

    # Upload settings from configuration file 
    #_________________________________________________

    isV0 = config['input']['isV0']

    RegressionColumns = config['training']['RegressionColumns']
    final_output_dir = config['output']['final_dir']
    delta_output_dir = config['output']['ml_dir']
    particle_dict = config['output']['particle']

    do_augm = config['data_prep']['do_augm']
    beta_flat = config['training']['beta_flat']
    beta_p_flat = config['training']['beta_p_flat']
    do_equal = config['data_prep']['do_equal']

    if isV0:        
        final_output_dir += '/V0'
        delta_output_dir += '/V0'
    else:           
        final_output_dir += '/TPC'
        delta_output_dir += '/TPC'

    if do_augm and beta_p_flat:  
        final_output_dir += '/augm' 
        delta_output_dir += '/augm'
    elif do_augm and beta_flat:  
        final_output_dir += '/augm_betaflat' 
        delta_output_dir += '/augm_betaflat'
    elif do_augm:               
        final_output_dir += '/augm'
        delta_output_dir += '/augm'
    elif beta_flat:
        final_output_dir += '/betaflat'             
        delta_output_dir += '/betaflat'
    elif beta_p_flat:
        final_output_dir += '/beta_pflat'            
        delta_output_dir += '/beta_pflat'
    elif do_equal:              
        final_output_dir += '/equal'
        delta_output_dir += '/equal'
    else:
        final_output_dir += '/no_options'
        delta_output_dir += '/no_options'


    output_file_end = '/beta_vs_p'
    if do_augm:         output_file_end += '_augm'
    if beta_flat:       output_file_end += '_betaflat'
    if beta_p_flat:     output_file_end += '_beta_pflat'
    if do_equal:        output_file_end += '_equal'

    output_file = f'{final_output_dir}' + output_file_end



    # Prediction and true beta vs p
    #________________________________________________

    with alive_bar(title='Application...') as bar:
        X_application = ApplicationDf[RegressionColumns]
        ApplicationDf['preds'] = model.predict(X_application)               # beta

    plot_specifics = config['plots']['appl_plot_spec']
    density_scatter(ApplicationDf['p'], ApplicationDf['preds'], output_file, plot_specifics['b_vs_p_final'], title='beta_vs_p_final')

    for name in particle_dict.values():   
        df = ApplicationDf.query(f'label == "{name}"', inplace=False)  
        

        density_scatter(df['p'], df['preds'], f'{output_file}_{name}', plot_specifics['b_vs_p_final'], title=f'beta_vs_p_final_{name}')
        density_scatter(df['p'], df['beta'], f'{final_output_dir}/beta_vs_p_{name}', plot_specifics=plot_specifics['b_vs_p_final'], title=f'beta_{name}')
    density_scatter(ApplicationDf['p'], ApplicationDf['beta'], f'{final_output_dir}/beta_vs_p_total', plot_specifics=plot_specifics['b_vs_p_final'], title=f'beta_true_total')  
    


    # Delta 
    #________________________________________________

    plot_score(ApplicationDf, ApplicationDf['p'], RegressionColumns, model, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{delta_output_dir}/Appl_', absolute=False)
    plot_score(ApplicationDf, ApplicationDf['beta'], RegressionColumns, model, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{delta_output_dir}/Appl_', absolute=False)

    for name in particle_dict.values():
        X_name = ApplicationDf.query(f"label == '{name}'", inplace=False)

        plot_score(X_name, X_name['p'], RegressionColumns, model, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{delta_output_dir}/Appl_{name}', absolute=False)
        plot_score(X_name, X_name['beta'], RegressionColumns, model, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{delta_output_dir}/Appl_{name}', absolute=False)




def main():

    # Configuration File
    #_________________________________
    with open('../configs/config.yml') as f:
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
    
    skip_appl = config['application']['skip_appl']
    if not skip_appl:   application(ApplicationDf, config, Model)      # yTest

    del TrainTestData, ApplicationDf



start_time = time()

main()

passed_time = time() - start_time
print(f'\nTime: {passed_time/60} min')