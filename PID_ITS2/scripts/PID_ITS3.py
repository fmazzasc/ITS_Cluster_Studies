#
#   Main script to perform PID using data collected by ITS2
#

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
import sys
import resource
from alive_progress import alive_bar

import xgboost as xgb
from hipe4ml.model_handler import ModelHandler
import optuna

from ROOT import TFile, TH3F

sys.path.append('..')
from src.preprocessing import *
from src.ml_utils import Scorer, delta_scorer
from src.weights_handler import DataAugmenter, SingleVariable_Flattener, DoubleVariable_Flattener
from utils.plotter import Plotter, TH2Handler

#################################################################################################################
#   GLOBAL  CONSTANTS

# Tags
#_____________________________________
tag_Deu = 'nSigmaDeuAbs < 1 and p <= 1.2'
#tag_P = 'nSigmaPAbs < 1 and nSigmaKAbs != 0 and nSigmaKAbs > 3 and p <= 1.5'
tag_P = 'nSigmaPAbs < 1 and p <= 1.5'
tag_K = 'nSigmaKAbs < 1 and nSigmaPiAbs > 3 and nSigmaPAbs > 3 and p <= 1.5'
tag_Pi = 'nSigmaPiAbs < 1  and p <= 1.5'
tag_E = 'nSigmaEAbs < 1 and nSigmaPiAbs > 3 and p <= 1.5'

# Masses
#_____________________________________
mass_Deu = 1.8756
mass_P =  0.93827200
mass_K = 0.4937
mass_Pi = 0.13957000
mass_E = 0.000511

names = ['E', 'Pi', 'K', 'P', 'Deu']

tag_dict = dict(zip(names, [tag_E, tag_Pi, tag_K, tag_P, tag_Deu]))
mass_dict = dict(zip(names, [mass_E, mass_Pi, mass_K, mass_P, mass_Deu]))

MC_dict = {2212: 1, 321: 2, 211: 3, 11: 4}

#################################################################################################################

class PID_config:
    """
    Class to load all necessary information to run PID

    Attributes
    ----------
        mode (str): Type of analysis to run (preprocessing steps depend on the dataset provided)
        skip_data_prep, skip_training, skip_appl (bool): Whether this steps should be skipped in the run. If 
            some of them are skipped, previous results are loaded to continue the analysis
        
        fimpPath (str): File with input data for training set
        ext_appl (bool): Whether an external dataset for application is provided or the training set should be 
            split.
        applPath (str): File with input data for application
        particle_dict (dict): Dictionary (index, particle_species)
        
        seven_hits (bool): Whether particles not detected by all seven silicon layers should be discarded
        test_size (float): Fraction of training set that will be used for test
        random_state (int): Seed initialization (garantees reproducibility)

        do_augm, beta_flat, beta_p_flat, MC_weights, do_equal (bool): Whether one of this weighting methods should
            be used in the ml regression
        
        do_plots (bool): Whether plots of preprocessed data should be produced
        Prep_output_dir, ML_output_dir, Application_output_dir, delta_output_dir (str): File to save plots to 
            (for data preparation, ML, application and score results)

        RegressionColumns (list[str]): Name of the columns of the dataset that should be considered for the regression
        model_choice (str): ML model to use for the regression
        ModelParams (dict): ML model hyperparams initialization values

        do_opt (bool): Whether an optuna hyperparameter optimization should be run
        HyperParamsRange (dict): Range in which the optuna optimization should be run
        early_stop (bool): Whether the optimization should automatically stop after given time
        save_model (bool): Whether the trained ML model should be saved to a file
        
    """
    def __init__(self, inputCfgFile):
        """
        Parameters
        ----------
            inputCfgFile (str): yaml file to load
        """
        with open(inputCfgFile) as f:   config = yaml.load(f, Loader=yaml.FullLoader)
    
        self.mode = config['mode']

        # process management
        self.skip_data_prep = config[self.mode]['data_prep']['skip_data_prep']
        self.skip_training = config[self.mode]['training']['skip_training']
        self.skip_appl = config[self.mode]['application']['skip_appl']

        # inputs
        self.fimpPath = config[self.mode]['input']['data']
        self.ext_appl = config[self.mode]['input']['ext_appl']
        if self.ext_appl:   self.applPath = config[self.mode]['input']['appl_data']
        else:               self.applPath = None
        self.particle_dict = config[self.mode]['output']['particle'] 
        
        # preprocessing config
        self.selection_tag = config[self.mode]['data_prep']['selection_tag']
        self.seven_hits = config[self.mode]['data_prep']['seven_hits']  # only consider candidates with hits on all the layers
        self.test_size = config[self.mode]['data_prep']['test_size']
        self.random_state = config[self.mode]['data_prep']['random_state']

        self.do_augm = config[self.mode]['data_prep']['do_augm']
        self.beta_flat = config[self.mode]['training']['beta_flat']
        self.beta_p_flat = config[self.mode]['training']['beta_p_flat']
        self.MCweights = config[self.mode]['training']['MCweights']
        self.do_equal = config[self.mode]['data_prep']['do_equal']

        # preprocessed data saving config
        self.save_data = config[self.mode]['data_prep']['save_data']
        self.save_data_dir = config[self.mode]['output']['save_data_dir']
        
        # output management
        self.do_plots = config[self.mode]['plots']['do_plots']
        self.Prep_output_dir = config[self.mode]['output']['data_visual_dir']
        self.ML_output_dir = config[self.mode]['output']['ml_dir']
        self.Application_output_dir = config[self.mode]['output']['final_dir']
        self.delta_output_dir = config[self.mode]['output']['delta_dir']

        output_specific = ''
        if self.do_augm and self.beta_p_flat:   output_specific = '_augm'
        elif self.do_augm and self.beta_flat:   output_specific = '_augm_betaflat'
        elif self.do_augm:                      output_specific = '_augm'
        elif self.MCweights:                    output_specific = '_MCweights'
        elif self.beta_flat:                    output_specific = '_betaflat'
        elif self.beta_p_flat:                  output_specific = '_beta_pflat'
        elif self.do_equal:                     output_specific = '_equal'
        else:                                   output_specific = '_no_options'
            
        self.ML_output_dir += output_specific
        self.Application_output_dir += output_specific
        self.delta_output_dir += output_specific
            
        # ML config
        self.RegressionColumns = config[self.mode]['training']['RegressionColumns']
        self.model_choice = config[self.mode]['training']['model']
        self.ModelParams = config[self.mode]['training']['ModelParams']

        self.do_opt = config[self.mode]['training']['do_opt']
        self.HyperParamsRange = config[self.mode]['training']['HyperParamsRange']
        self.early_stop = config[self.mode]['training']['early_stop']
        self.save_model = config[self.mode]['training']['save_model']

def timing_decorator(func):
    """
    Decorator to return time that a function takes to be run.
    To use this decorator for a function type '@time_function' before the function definition.
    """

    def wrapper(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        exec_time = end_time - start_time
        if exec_time > 60:  
            exec_time_min = int(exec_time / 60.)
            exec_time_s = exec_time % 60

        print(f'\n{func.__name__} process ended.')
        if exec_time > 60:  print(f'Execution time: {exec_time_min}:{exec_time_s:.2f} min')
        else:               print(f'Execution time: {end_time - start_time:.2f} s')
        print(f'Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} MB')
        
        return result 

    return wrapper

@timing_decorator
def perform_data_augmentation(config, opt, data, yData):
    """
    Perform all necessary steps for data augmentation on a given dataset

    Parameters
        config (yaml.load): Configurations from yaml file
        opt (PID_config): Easier access to configurations from yaml file
        data (pd.DataFrame): Dataset which data augmentation will be applied on
        yData (pd.Series): ML dataset target (must be updated after data augm)
    ----------

    """
    print('\nData Augmentation...')
    data['copy'] = 0

    dt_augm = DataAugmenter(data, daughters=config[opt.mode]['data_prep']['to_augm'], mothers=config[opt.mode]['data_prep']['mothers'], p_ranges=config[opt.mode]['data_prep']['p_ranges'], mass_dict=mass_dict)
    data = dt_augm.do_augm()
    dt_augm.print_augm()
    yData = data['beta']

    # Plots after augmentation
    if opt.do_plots:
        
        plotAxis2Ddict = config[opt.mode]['plots']['plotAxis2D']
        plotSpec2Ddict = config[opt.mode]['plots']['plotSpec2D']
        varsToPlot2D = [var for var in plotAxis2Ddict.values()]
        xsToPlot2D = [item[0] for item in varsToPlot2D]
        ysToPlot2D = [item[1] for item in varsToPlot2D]
        plotSpec2D = [spec for spec in plotSpec2Ddict.values()]

        outFile = TFile(f'{opt.Prep_output_dir}_AUGMENTED.root', 'recreate')
        print(f'ROOT file created in {opt.Prep_output_dir}_AUGMENTED.root')
        plot = Plotter(data, outFile)
        plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)

        plot.multi_df('label', opt.particle_dict.values())
        plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)
        outFile.Close()

@timing_decorator
def weights_application(config, opt, data):
    """
    Evaluate weights and append them to the dataset. They will be used in ML algorithms. This function acts as an 
    interface to run different modes.

    Parameters
    ----------
        config (yaml.load): Configurations from yaml file
        opt (PID_config): Easier access to configurations from yaml file
        data (pd.DataFrame): Dataset which weights will be applied on

    Returns
    -------
        Dataset with applied weights
    """

    if opt.do_equal:        data = perform_do_equal(data)
    elif opt.beta_flat:     data = perform_beta_flat(config, opt, data)
    elif opt.beta_p_flat:   data = perform_beta_p_flat(config, opt, data)
    elif opt.MCweights:     data = perform_MC_weights(config, opt, data)

    return data

#################################################################################################################
#
#   Specific weights functions
#________________________________________________________________________________________________________________

def perform_do_equal(data, use_e=False, use_d=False):
    """
    From a given dataframe, finds the minimum number of elements having unique values in the label column. Discards elements
    having different unique values in that column until their size matches that minimum number. 

    Parameters
    ----------------
        data (pd.DataFrame): Dataframe
        use_e, use_d (bool): Whether to use electrons and deuterons (typically you want to avoid that, since the 
            available statistics would drop significantly)

    Returns
    ----------------
        "Filtered" dataframe
    """

    if not use_e:   data = data.query("label != 'E")
    if not use_d:   data = data.query("label != 'Deu")

    sizes = []
    for item in data['label'].unique():  sizes.append(len(data.query(f"{'label'} == '{item}'")))
    min_size = min(sizes)

    equal_data = pd.DataFrame()
    for item in data['label'].unique():  equal_data = pd.concat([new_df, data.query(f"{'label'} == '{item}'").iloc[:min_size]], sort=False)

    return equal_data

def perform_beta_flat(config, opt, data):
    """
    Evaluate weights to flatten the beta distribution in the dataset. They will be used in ML algorithms.

    Parameters
    ----------
        config (yaml.load): Configurations from yaml file
        opt (PID_config): Easier access to configurations from yaml file
        data (pd.DataFrame): Dataset which weights will be applied on

    Returns
    -------
        Dataset with applied weights
    """

    print('\nBeta flat...')

    betaSpec = config[opt.mode]['data_prep']['betaSpec']
    weight_handler = SingleVariable_Flattener(data, 'beta_weight', 'beta')
    weight_handler.eval_weights(betaSpec[0], betaSpec[1], betaSpec[2])
    return weight_handler.assign_weights()

def perform_beta_p_flat(config, opt, data):
    """
    Evaluate weights to flatten the beta and momentum distribution in the dataset. They will be used in ML algorithms.
    Plots are created to visualize the results.

    Parameters
    ----------
        config (yaml.load): Configurations from yaml file
        opt (PID_config): Easier access to configurations from yaml file
        data (pd.DataFrame): Dataset which weights will be applied on

    Returns
    -------
        Dataset with applied weights
    """

    print('\nBeta and momentum flat...')

    betaPSpec = config[opt.mode]['data_prep']['betaPSpec']

    #weight_handler = DoubleVariable_Flattener(data, 'beta_pweight', 'p', 'particle')
    weight_handler = DoubleVariable_Flattener(data, 'beta_pweight', 'p', 'particle')
    weight_handler.eval_weights(betaPSpec[0], betaPSpec[1], betaPSpec[2], betaPSpec[3], betaPSpec[4], betaPSpec[5])
    data = weight_handler.assign_weights()

    plotAxisBetaPdict = config[opt.mode]['plots']['plotAxisBetaP']
    plotSpecBetaPdict = config[opt.mode]['plots']['plotSpecBetaP']
    varsToPlot2D = [var for var in plotAxisBetaPdict.values()]
    xsToPlot2D = [item[0] for item in varsToPlot2D]
    ysToPlot2D = [item[1] for item in varsToPlot2D]
    plotSpec2D = [spec for spec in plotSpecBetaPdict.values()]     #### WIP in the config file

    outFile = TFile(f'{opt.Prep_output_dir}_BETAPFLAT.root', 'recreate')
    print(f'ROOT file created in {opt.Prep_output_dir}_BETAPFLAT.root')
    plot = Plotter(data, outFile)

    plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)
    plot.plot2D([xsToPlot2D[0]], [ysToPlot2D[0]], [plotSpec2D[0]], 'beta_pweight')
    outFile.Close()

    return data

def perform_MC_weights(config, opt, data):
    """
    Evaluate tridimensional weights to flatten the beta and momentum distribution in the MC sample and match the cluster size distribution with 
    the ones from collected data. They will be used in ML algorithms. 
    Plots are created to visualize the results.
    Collected data will be loaded from a specific file

    THIS FUNCTION SHOULD BE UPDATED

    Parameters
    ----------
        config (yaml.load): Configurations from yaml file
        opt (PID_config): Easier access to configurations from yaml file
        data (pd.DataFrame): Dataset which weights will be applied on

    Returns
    -------
        Dataset with applied weights
    """

    print('\nMC weights...')

    df = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520143_itstpc.parquet')
    df.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
    df.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True) 

    #dfs = [df.query(f'particle == {num}', inplace=False) for num in particle_dict.keys()]
    #for dfi, name in zip(dfs, particle_dict.values()): 
    #density_scatter(df['p'], df['dedx'], f'{output_dir}/ITSTPC_dedx', ["p", "#frac{dE}{dx}", 150, 0, 1.5, 600, 0, 600])

    for name in opt.particle_dict.values():     df[f'nSigma{name}Abs'] = abs(df[f'nSigma{name}'])

    values = [1, 2, 3, 4]
    conditions = [(df['nSigmaPAbs'] < 1) & (df['nSigmaPAbs'] < df['nSigmaKAbs']) & (df['nSigmaPAbs'] < df['nSigmaPiAbs']) & (df['nSigmaPAbs'] < df['nSigmaEAbs']),
                    (df['nSigmaKAbs'] < 1) & (df['nSigmaKAbs'] < df['nSigmaPAbs']) & (df['nSigmaKAbs'] < df['nSigmaPiAbs']) & (df['nSigmaKAbs'] < df['nSigmaEAbs']),
                    (df['nSigmaPiAbs'] < 1) & (df['nSigmaPiAbs'] < df['nSigmaPAbs']) & (df['nSigmaPiAbs'] < df['nSigmaKAbs']) & (df['nSigmaPiAbs'] < df['nSigmaEAbs']),
                    (df['nSigmaEAbs'] < 1) & (df['nSigmaEAbs'] < df['nSigmaPAbs']) & (df['nSigmaEAbs'] < df['nSigmaKAbs']) & (df['nSigmaEAbs'] < df['nSigmaPiAbs']),]

    df['particle'] = np.select(conditions, values)

    h1 = TH3F('h1', 'h1', 250, 0, 25, 15, 0.0, 1.5, 5, 0, 5)

    N = len(df)
    pmins = np.linspace(0, 1.4, num=15)
    pmaxs = np.linspace(0.1, 1.5, num=15)

    for num in opt.particle_dict.keys():  
        for i, (pmin, pmax) in enumerate(zip(pmins, pmaxs)):
            temp_df = df.query(f'particle == {num} and {pmin} <= p < {pmax}', inplace=False)

            w = len(temp_df)/N
            if w != 0:  weight = 1./w
            else:       weight = 0.
        
            for x, y, z in zip(temp_df['clSizeCosLam'], temp_df['p'], temp_df['particle']):    h1.Fill(x, y, z, weight)

    h = h1.Project3D('zy')

    file = TFile('TH2.root', 'recreate')
    h.Write()
    file.Close()

    h2 = TH3F('h2', 'h2', 250, 0, 25, 15, 0.0, 1.5, 5, 0, 5)
    for x, y, z in zip(TrainSet['clSizeCosLam'], TrainSet['p'], TrainSet['particle']):    h2.Fill(x, y, z)

    h1.Divide(h2)

    weights = []
    conditions = []

    for xbin in range(1, h1.GetNbinsX()+1):
        xmin = h1.GetXaxis().GetBinLowEdge(xbin)
        xmax = xmin + h1.GetXaxis().GetBinWidth(xbin)

        for ybin in range(1, h1.GetNbinsY()+1):
            ymin = h1.GetYaxis().GetBinLowEdge(ybin)
            ymax = ymin + h1.GetYaxis().GetBinWidth(ybin)

            for zbin in range(1, h1.GetNbinsZ()+1):
                
                w = h1.GetBinContent(h1.GetBin(xbin, ybin, zbin))

                if zbin != 1:

                    #print('particle:', zbin-1,'| pmin:', ymin, '| pmax:', ymax, '|', w)
                    weights.append(w)
                    conditions.append((TrainSet['particle'] == zbin-1) & (TrainSet['clSizeCosLam'] >= xmin) & (TrainSet['clSizeCosLam'] < xmax)
                                    & (TrainSet['p'] >= ymin) & (TrainSet['p'] < ymax))

    TrainSet['3d_weight'] = np.select(conditions, weights)

    h3 = TH3F('h3', 'h3', 250, 0, 25, 15, 0.0, 1.5, 5, 0, 5)
    for x, y, z, w in zip(TrainSet['clSizeCosLam'], TrainSet['p'], TrainSet['particle'], TrainSet['3d_weight']):    h3.Fill(x, y, z, w)

    h4 = h3.Project3D('zy')

    file = TFile('TH2_2.root', 'recreate')
    h4.Write()
    file.Close()

    del h1, h2, h3, h, h4

    pmins = np.linspace(0, 1.4, num=15)
    pmaxs = np.linspace(0.1, 1.5, num=15)

    dfs: list[list[pd.DataFrame]] = [[] for i in range(4)]
    dfs2: list[list[pd.DataFrame]] = [[] for i in range(4)]
    for num, df_list, df_list2 in zip(range(1, 5), dfs, dfs2):
        for pmin, pmax in zip(pmins, pmaxs):
            df_list.append(pd.DataFrame(TrainSet.query(f'particle == {num} and {pmin} <= p < {pmax}', inplace=False)))
            df_list2.append(pd.DataFrame(df.query(f'particle == {num} and {pmin} <= p < {pmax}', inplace=False)))

    for name, df_list in zip(particle_dict.values(), dfs):  multiple_hist(df_list, 'clSizeCosLam', [250, 0, 25], f'{output_dir}/3d_{name}_', weights='3d_weight')
    for name, df_list in zip(particle_dict.values(), dfs2):  multiple_hist(df_list, 'clSizeCosLam', [250, 0, 25], f'{output_dir}/TOCOMPARE_{name}_')

    plot_spec_flat = config['plots']['bp_flat_scat']
    density_scatter(TrainSet['p'], TrainSet['particle'], f'{output_dir}/3d_weighted_momentum', plot_specifics=plot_spec_flat['p'], weights=TrainSet['3d_weight']) 

#################################################################################################################
#
#   Specific ML functions
#________________________________________________________________________________________________________________

def optuna_optimization(opt, model, TrainTestData):
    """
    Execute optuna optimization process for a given ML model. hipe4ml classes will be used.

    Parameters
    ----------
        opt (PID_config): Loaded yaml configuration (easier to access)
        model (): ML model to use.
        TrainTestData (list[pd.DataFrame]): [X_train, y_train, X_test, y_test]. It will be used for training 
            and testing the ML model (Xs still have the column passed to the ys).
    Returns
    -------
        HyperParams (study.best_trial.params): Hyperparameters from the trial with best results.
    """

    model_handler = ModelHandler(model, opt.RegressionColumns)
    model_handler.set_model_params(opt.ModelParams)

    print('\nInitialize Optuna hyper-parameters optimization...')
    with alive_bar(title='Hyper-Parameters optimization') as bar:
        if opt.early_stop:  study = model_handler.optimize_params_optuna(TrainTestData, opt.HyperParamsRange, delta_scorer, direction='maximize', timeout=300)
        #if opt.early_stop:  study = model_handler.optimize_params_optuna(TrainTestData, opt.HyperParamsRange, delta_scorer, direction='maximize', callbacks=[callback])
        else:               study = model_handler.optimize_params_optuna(TrainTestData, opt.HyperParamsRange, delta_scorer, direction='maximize', n_trials=50)
        #else:              study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='minimize', fit_params={'sample_weight': X_train['beta_pweight']}, timeout=300)
        ## VERIFY HOW CLASSES ASSIGN WEIGHTS TO REWRITE THE LAST LINE. IN THIS WAY THE ALGORITHM WILL TRAIN KNOWING ABOUT THE WEIGHTS
    
    # Save results
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f'{opt.ML_output_dir}plot_optimization_history.png')

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f'{opt.ML_output_dir}plot_param_importances.png')

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(f'{opt.ML_output_dir}plot_parallel_coordinate.png')

    fig = optuna.visualization.plot_contour(study)
    fig.write_image(f'{opt.ML_output_dir}plot_contour.png')

    return study.best_trial.params

def perform_xgboost_regression(config, opt, TrainSet, TestSet, HyperParams):
    """
    Performs a ML regression with a xgboost model and produces plots and graphs.

    Parameters
    ----------
        config (yaml object): .yml configurations
        opt (PID_config): Loaded yaml configuration (easier to access)
        TrainSet, TestSet (pd.DataFrame): Dataset used to train and test the ML model

    Returns
    -------
        model (xgb.XGBRegressor): xgboost ML model
    """

    print('\nXGB model...')
    if opt.beta_flat:   print('Beta weights selected...')
    if opt.beta_p_flat: print('Beta and momentum weights selected...')
    if opt.MCweights:   print('3D weights selected...')

    # Create and train the model
    model = xgb.XGBRegressor(**HyperParams, random_state=opt.random_state)
    
    with alive_bar(title='Training...') as bar:     
        if opt.beta_flat:       model.fit(TrainSet[opt.RegressionColumns], TrainSet['beta'], sample_weight=TrainSet['beta_weight'])
        elif opt.beta_p_flat:   model.fit(TrainSet[opt.RegressionColumns], TrainSet['beta'], sample_weight=TrainSet['beta_pweight'])
        elif opt.MCweights:     model.fit(TrainSet[opt.RegressionColumns], TrainSet['beta'], sample_weight=TrainSet['3d_weight'])
        else:                   model.fit(TrainSet[opt.RegressionColumns], TrainSet['beta'])

    # Hyperparameter importance 
    FeatureImportanceScores = model.get_booster().get_score(importance_type='weight')

    HyperparametersImportanceDict = {}
    for feature, importance in FeatureImportanceScores.items(): HyperparametersImportanceDict[feature] = importance
    HyperparametersImportanceDict = dict(sorted(HyperparametersImportanceDict.items(), key=lambda x: x[1], reverse=True))

    plt.bar(range(len(HyperparametersImportanceDict)), list(HyperparametersImportanceDict.values()), align='center')
    plt.xticks(range(len(HyperparametersImportanceDict)), list(HyperparametersImportanceDict.keys()), rotation='vertical')
    plt.title('Hyperparameter Importance')
    plt.savefig(f'{opt.ML_output_dir}_HyperparamImportances.png')
    
    # Results visualization
    dfs = {'train': TrainSet, 'test': TestSet}
    for name, df in dfs.items():

        outFile = TFile(f'{opt.ML_output_dir}_{name}.root', 'recreate')
        print(f'ROOT file created in {opt.ML_output_dir}_{name}.root')

        scorer = Scorer(model, df, opt.RegressionColumns, 'beta', outFile)
        scorer.Delta()

        pPltSpec = config[opt.mode]['plots']['scoreDeltaSpec']['p']
        betaPltSpec = config[opt.mode]['plots']['scoreDeltaSpec']['beta_pred']

        scorer.histScore()
        scorer.scatterPlotScore('p', pPltSpec[0], pPltSpec[1], pPltSpec[2], pPltSpec[3])
        scorer.scatterPlotScore('beta', betaPltSpec[0], betaPltSpec[1], betaPltSpec[2], betaPltSpec[3])

        scorer.plot.multi_df('label', opt.particle_dict.values())
        scorer.histScore()
        scorer.scatterPlotScore('p', pPltSpec[0], pPltSpec[1], pPltSpec[2], pPltSpec[3])
        scorer.scatterPlotScore('beta', betaPltSpec[0], betaPltSpec[1], betaPltSpec[2], betaPltSpec[3])  

        outFile.Close()
    
    # Feature importance plot
    fig, ax = plt.subplots(figsize=(20, 10))
    xgb.plot_importance(model, ax= ax)
    plt.savefig(f'{opt.ML_output_dir}_FeatureImportance.png')
    plt.close('all')
    
    return model

def perform_automl_regression(config, opt, TrainSet, TestSet, HyperParams):
    """
    Performs a ML regression with FLAML's autoML model and produces plots and graphs.

    Parameters
    ----------
        config (yaml object): .yml configurations
        opt (PID_config): Loaded yaml configuration (easier to access)
        TrainSet, TestSet (pd.DataFrame): Dataset used to train and test the ML model

    Returns
    -------
        model (AutoML): FLAML AutoML model
    """
    print('\nFLAML model...')
    model = AutoML(**HyperParams)
    plot_specifics = config['plots']['model_train']
    with alive_bar(title='Training...') as bar:     model.fit(TrainSet[opt.RegressionColumns], TrainSet['beta'])

    TrainTestData = [TrainSet, TrainSet['beta'], TestSet, TestSet['beta']]
    plot_score_train(TrainTestData, RegressionColumns, model, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{opt.ML_output_dir}', absolute=False)
    plot_score_train(TrainTestData, RegressionColumns, model, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{opt.ML_output_dir}', absolute=False)
    for name in particle_dict.values():

        X_train_name = TrainSet.query(f"label == '{name}'")
        y_train_name = X_train_name['beta']
        X_test_name = TestSet.query(f"label == '{name}'")
        y_test_name = X_test_name['beta']

        TestTrainData_name = X_train_name, y_train_name, X_test_name, y_test_name
        plot_score_train(TestTrainData_name, RegressionColumns, model, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{opt.ML_output_dir}_{name}', absolute=False)
        plot_score_train(TestTrainData_name, RegressionColumns, model, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{opt.ML_output_dir}_{name}', absolute=False)
    
    return model

#################################################################################################################
#
#   Functions
#________________________________________________________________________________________________________________

@timing_decorator
def data_prep(config, opt):
    """
    Data preparation function. The full data will be uploaded and dataframes for single particle species will be created according to the tag rules. 
    Data will be visualized in plots and histograms. Weight evaluation and data augmentation process will be perfomed

    Parameters
    ----------
        config (yaml object): .yml configurations
        opt (PID_config): loaded yaml configuration (easier to access)

    Returns
    -------
        TrainSet, TestSet (pd.DataFrame): Training and test set for the ML model. The target is still present in the dataset.
        ApplicationDf (pd.DataFrame): dataframe that will be used for the application.
    """

    print('\nPerforming data preparation...')

    # Data preprocessing
    prep_con = PrepConstructor()
    prep = prep_con.createPrepTool(opt.mode, opt.fimpPath, opt.applPath)
    prep.preprocess(opt.particle_dict, opt.selection_tag)

    #if opt.mode != 'TPC':   tag_dict = None
    TrainSet, TestSet, yTrain, yTest, ApplicationDf = prep.filter_and_split(opt.particle_dict, mass_dict, tag_dict, opt.test_size, opt.random_state)
    TotalSet = pd.concat([TrainSet, TestSet])

    # Data visualization
    if opt.do_plots:
        with alive_bar(title='Data Visualization...') as bar:
            
            file = TFile(f'{opt.Prep_output_dir}.root', 'recreate')
            print(f'\nROOT file created in {opt.Prep_output_dir}.root')
            plot = Plotter(TotalSet, file)

            plotSpec1Ddict = config[opt.mode]['plots']['plotSpec1D']
            varsToPlot1D = [var for var in plotSpec1Ddict.keys()]
            plotSpec1D = [spec for spec in plotSpec1Ddict.values()]
            plot.plot1D(varsToPlot1D, plotSpec1D)

            plotAxis2Ddict = config[opt.mode]['plots']['plotAxis2D']
            plotSpec2Ddict = config[opt.mode]['plots']['plotSpec2D']
            varsToPlot2D = [var for var in plotAxis2Ddict.values()]
            xsToPlot2D = [item[0] for item in varsToPlot2D]
            ysToPlot2D = [item[1] for item in varsToPlot2D]
            plotSpec2D = [spec for spec in plotSpec2Ddict.values()]
            plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)

            plot.multi_df('label', opt.particle_dict.values())
            plot.plot1D(varsToPlot1D, plotSpec1D)
            plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)
            file.Close()

            file = TFile(f'{opt.Prep_output_dir}_0304.root', 'recreate')
            plot = Plotter(TotalSet.query('0.3 < p < 0.4', inplace=False), file)

            plot.multi_df('label', opt.particle_dict.values())
            plot.plot1D(varsToPlot1D, plotSpec1D)
            file.Close()
    
    # End data preprocessing
    prep.preprocess_2(opt.seven_hits)
    TrainSet, TestSet, yTrain, yTest, ApplicationDf = prep.return_dfs()

    # Data augmentation
    if opt.do_augm: perform_data_augmentation(config, opt, TrainSet, yTrain)

    # Apply weights
    TrainSet = weights_application(config, opt, TrainSet)
    yTrain = TrainSet['beta']

    # Save data 
    if opt.save_data:
        with alive_bar(title='Save preprocessed data...') as bar:
            data_conf = ''
            if opt.do_augm:         data_conf += '_augm'
            if opt.beta_flat:       data_conf += '_betaflat'
            if opt.beta_p_flat:     data_conf += '_beta_pflat'
            if opt.do_equal:        data_conf += '_equal'

            TrainSet.to_parquet(f'{opt.save_data_dir}/TrainSet{data_conf}.parquet.gzip')
            TestSet.to_parquet(f'{opt.save_data_dir}/TestSet{data_conf}.parquet.gzip')

    return TrainSet, TestSet, ApplicationDf

@timing_decorator
def regression(config, opt, TrainSet, TestSet):
    """
    Regression function. Creates a machine learning model and trains it on a given sample. Training results will
    be visible as plots and histograms.

    Parameters
    ----------
        config (yaml object): .yml configurations
        opt (PID_config): Loaded yaml configuration (easier to access)
        TrainSet, TestSet (pd.DataFrame): Dataset used to train and test the ML model

    Returns
    -------
        TrainSet, TestSet (pd.DataFrame): Training and test set for the ML model. The target is still present in the dataset.
        ApplicationDf (pd.DataFrame): dataframe that will be used for the application.
    """

    TestTrainData = [TrainSet, TrainSet['beta'], TestSet, TestSet['beta']]

    # Model definition
    if opt.model_choice=='xgboost':     model = xgb.XGBRegressor(random_state=opt.random_state)
    if opt.model_choice=='automl':      model = AutoML()

    # Optuna optimization
    if opt.do_opt:      HyperParams = optuna_optimization(opt, model, TestTrainData)
    else:               HyperParams = opt.ModelParams

    # Model training and evaluation
    if opt.model_choice=='xgboost':     model = perform_xgboost_regression(config, opt, TrainSet, TestSet, HyperParams)
    if opt.model_choice=='automl':      model = perform_automl_regression(config, opt, TrainSet, TestSet, HyperParams)
    
    # Save model in pickle
    if opt.save_model:
        model_out = config[opt.mode]['output']['model_out']

        print('\nSaving regressor model...')
        with open(f'{model_out}_{opt.model_choice}.pickle', 'wb') as output_file:
            pickle.dump(model, output_file)
        print(f'Model saved in {model_out}_{opt.model_choice}.pickle')

    return model
    
@timing_decorator
def application(config, opt, ApplicationDf, model):
    """
    Function to apply an ML model to a dataset and visualize results in histograms and plots.
    Application set will be saved to a file after some columns will be appended (predicted beta and a resolution).

    Parameters
    ----------
        config (yaml object): .yml configurations
        opt (PID_config): Loaded yaml configuration (easier to access)
        ApplicationDf (pd.DatFrame): Dataset which the model will be applied on
        model (): ML model to apply
    """

    # Prediction and true beta vs p
    outDeltaFile = TFile(f'{opt.delta_output_dir}.root', 'recreate')
    print(f'ROOT file created in {opt.delta_output_dir}.root')
    with alive_bar(title='Application...') as bar:
        scorer = Scorer(model, ApplicationDf, opt.RegressionColumns, 'beta', outDeltaFile)   # pred column added
        scorer.evalMass()

    # Application plots
    outFile = TFile(f'{opt.Application_output_dir}.root', 'recreate')
    print(f'ROOT file created in {opt.Application_output_dir}.root')
    plot = Plotter(scorer.df, outFile)

    
    plotSpec1Ddict = config[opt.mode]['plots']['finalSpec1D']
    varsToPlot1D = [var for var in plotSpec1Ddict.keys()]
    plotSpec1D = [spec for spec in plotSpec1Ddict.values()]
    plot.plot1D(varsToPlot1D, plotSpec1D)

    plotAxis2Ddict = config[opt.mode]['plots']['finalAxis2D']
    plotSpec2Ddict = config[opt.mode]['plots']['finalSpec2D']
    varsToPlot2D = [var for var in plotAxis2Ddict.values()]
    xsToPlot2D = [item[0] for item in varsToPlot2D]
    ysToPlot2D = [item[1] for item in varsToPlot2D]
    plotSpec2D = [spec for spec in plotSpec2Ddict.values()]
    plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)

    scorer.df = process_application(scorer.df, opt.selection_tag, opt.particle_dict, mass_dict, tag_dict)
    plot.df = scorer.df
    plot.multi_df('label', opt.particle_dict.values())
    plot.plot1D(varsToPlot1D, plotSpec1D)
    plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)
    outFile.Close()

    # Delta plots
    ApplicationDf = scorer.Delta()                                                  # Delta column added

    pPltSpec = config[opt.mode]['plots']['scoreDeltaSpec']['p']
    betaPltSpec = config[opt.mode]['plots']['scoreDeltaSpec']['beta_pred']

    scorer.histScore()
    scorer.scatterPlotScore('p', pPltSpec[0], pPltSpec[1], pPltSpec[2], pPltSpec[3])
    scorer.scatterPlotScore('beta', betaPltSpec[0], betaPltSpec[1], betaPltSpec[2], betaPltSpec[3])

    scorer.plot.multi_df('label', opt.particle_dict.values())
    scorer.histScore()
    scorer.scatterPlotScore('p', pPltSpec[0], pPltSpec[1], pPltSpec[2], pPltSpec[3])
    scorer.scatterPlotScore('beta', betaPltSpec[0], betaPltSpec[1], betaPltSpec[2], betaPltSpec[3]) 

    outDeltaFile.Close()
    
    # Save data 
    if opt.save_data:
        with alive_bar(title='Save preprocessed data...') as bar:
            data_conf = ''
            if opt.do_augm:         data_conf += '_augm'
            if opt.beta_flat:       data_conf += '_betaflat'
            if opt.beta_p_flat:     data_conf += '_beta_pflat'
            if opt.do_equal:        data_conf += '_equal'

            ApplicationDf.to_parquet(f'{opt.save_data_dir}/ApplicationDf{data_conf}.parquet.gzip')

    # Resolution curves
    outFile = TFile(f'{opt.delta_output_dir}_LINE.root', 'recreate')
    print(f'ROOT file created in {opt.delta_output_dir}_LINE.root')
    th2_handler = TH2Handler(ApplicationDf, outFile, 'p', 'Delta')

    for name in opt.particle_dict.values():
        X_name = ApplicationDf.query(f"label == '{name}'", inplace=False)
        th2_handler.df = X_name
        
        th2_handler.build_th2(150, 0., 1.5, 300, -1.5, 1.5)
        th2_handler.TH2toLine(f'Delta_line_{name}', 'y', 1)
    outFile.Close()
     
#################################################################################################################
#
#   Main functions
#________________________________________________________________________________________________________________

@timing_decorator
def PID(inputCfgFile):
    """
    Execute full PID process

    Parameters
    ----------
        inputCfgFile (str): File to load configurations from 
    """

    # load configurations from yaml file
    print(f'\nLoading configurations from {inputCfgFile}')
    with open(inputCfgFile) as f:   config = yaml.load(f, Loader=yaml.FullLoader)
    opt = PID_config(inputCfgFile)

    TrainSet, TestSet, ApplicationDf = data_prep(config, opt)
    model = regression(config, opt, TrainSet, TestSet)
    application(config, opt, ApplicationDf, model)

if __name__ == '__main__':

    inputCfgFile = '../configs/config.yml'

    PID(inputCfgFile)
    
    print('\nHello world!')