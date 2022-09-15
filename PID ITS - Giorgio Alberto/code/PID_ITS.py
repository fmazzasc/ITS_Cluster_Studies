import pickle
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alive_progress import alive_bar
from time import time

from sklearn.model_selection import train_test_split
import xgboost as xgb
from flaml import AutoML


from hipe4ml.model_handler import ModelHandler
from hipe4ml import plot_utils
import optuna

from ROOT import TH2F, TCanvas, TH1F, gStyle, kBird, gROOT, TFile

from UsefulFunctions import filtering, multiple_hist, density_scatter, equal, augmentation_fine, callback, Delta_score, plot_score, plot_score_train

gROOT.SetBatch()

# Tags
#_____________________________________
tag_Deu = 'nSigmaDeuAbs < 1 and p <= 1.2'
tag_P = 'nSigmaPAbs < 1 and p <= 0.7'
tag_K = 'nSigmaKAbs < 1 and p <= 0.7'
tag_Pi = 'nSigmaPiAbs < 1 and p <= 0.7'
tag_E = 'nSigmaPi > 5'


# Masses
#_____________________________________
mass_Deu = 1.8756
mass_P =  0.93827200
mass_K = 0.4937
mass_Pi = 0.13957000
mass_E = 0.000511

names = ['Deu', 'P', 'K', 'Pi', 'E']

tag_dict = dict(zip(names, [tag_Deu, tag_P, tag_K, tag_Pi, tag_E]))
mass_dict = dict(zip(names, [mass_Deu, mass_P, mass_K, mass_Pi, mass_E]))







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

    if True:
        isV0 = config['input']['isV0']
        hybrid = config['input']['hybrid']
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

        if hybrid:
            output_dir += '/hybrid'
            save_data_dir += '/hybrid'
        elif isV0:                    
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



    # Define some new columns
    #_________________________________

    RegressionDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
    RegressionDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
    RegressionDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)
    
    RegressionDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)  

    if ext_appl:
        ApplicationDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
        ApplicationDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
        ApplicationDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)  
        ApplicationDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
        
    for part in particle_dict.values():  RegressionDf[f'nSigma{part}Abs'] = abs(RegressionDf[f'nSigma{part}'])
    if ext_appl:    
        for part in particle_dict.values():  ApplicationDf[f'nSigma{part}Abs'] = abs(RegressionDf[f'nSigma{part}'])



    
    # Splitting, filtering (and evaluating beta)
    #__________________________________
    
    RegressionDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)        
    if ext_appl:    ApplicationDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)

    if not isV0:
        RegressionDf.query('p <= 0.7', inplace=True)
        ApplicationDf.query('p <= 0.7', inplace=True)

    #RegressionDf.query('p <= 0.6', inplace=True)
    

    if isV0:
        RegressionDf.eval('label = particle', inplace=True)
        for number, name in particle_dict.items():  RegressionDf['label'].mask(RegressionDf['particle'] == number, name, inplace=True)     

    ApplicationDf.eval('label = particle', inplace=True)
    for number, name in particle_dict.items():  ApplicationDf['label'].mask(ApplicationDf['particle'] == number, name, inplace=True)


    # redefine K
    #RegressionDf['label'].mask(RegressionDf['nSigmaKAbs'] < 1, 'K', inplace=True)
    #if isV0:    RegressionDf['particle'].mask(RegressionDf['nSigmaKAbs'] < 1, 2, inplace=True)

    #if ext_appl:    
    #    ApplicationDf['label'].mask(ApplicationDf['nSigmaKAbs'] < 1, 'K', inplace=True)
    #    if isV0:    ApplicationDf['particle'].mask(ApplicationDf['nSigmaKAbs'] < 1, 2, inplace=True)

    
    TrainSet, TestSet, yTrain, yTest = train_test_split(RegressionDf, RegressionDf.p, test_size=test_frac, random_state=seed_split)

    if isV0:
        dfs_train = [filtering(TrainSet, name, mass=mass_dict[name], label=False)for name in particle_dict.values()]
        dfs_test = [filtering(TestSet, name, mass=mass_dict[name], label=False)for name in particle_dict.values()]

    else:
        dfs_train = [filtering(TrainSet, name, tag=tag_dict[name], mass=mass_dict[name]) for name in particle_dict.values()]
        dfs_test = [filtering(TestSet, name, tag=tag_dict[name], mass=mass_dict[name]) for name in particle_dict.values()]

   




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
            for name, df in zip(particle_dict.values(), filt_dfs): 
                density_scatter(df[x], df[y], f'{output_dir}/{y}_vs_{x}_{name}', scat_spec, title=f'{y}_{name}')
            density_scatter(total_df[x], total_df[y], f'{output_dir}/{y}_vs_{x}_total', scat_spec, title=f'{y}_total')

        density_scatter(RegressionDf['p'], RegressionDf['dedx'], f'{output_dir}/dedx_all', plot_spec_scat[0], title='dedx_all')
        density_scatter(RegressionDf['p'], RegressionDf['clSizeCosLam'], f'{output_dir}/clSizeCosLam_all', plot_spec_scat[2], title='clSizeCosLam_all')
        
        # Check hypotesis 
        check_dfs = []
        for name in particle_dict.values():         check_dfs.append( total_df.query(f"label == '{name}' and -0.1 <= SnPhiL0 <= 0.1 and -0.1 <= tgL <= 0.1") )
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
            augm_df = augmentation_fine(TrainSet, mother, daughter, mass_dict[mother], mass_dict[daughter], pmin, pmax)
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
            
            N = len(TrainSet)
            weights = []
            conditions = []

            for name in particle_dict.values():  
                for i, (pmin, pmax) in enumerate(zip(pmins, pmaxs)):    
                    weights.append(len(TrainSet.query(f'label == "{name}" and {pmin} <= p < {pmax}'))/N)
                    conditions.append((TrainSet['label'] == name) & (TrainSet['p'] >= pmin) & (TrainSet['p'] < pmax))
                
            
            n_weights = []
            for weight in weights:
                if weight == 0.:    n_weights.append(0.)
                else:               n_weights.append(1./weight)
            nn_weights = []
            for n_weight in n_weights:
                if n_weight == 0:   nn_weights.append(0.)
                else:               nn_weights.append(n_weight * 1000)

            TrainSet['beta_pweight'] = np.select(conditions, nn_weights)

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
            dfApplicationDf = pd.DataFrame(ApplicationDf)
    
            dfTrainSet.to_parquet(f'{save_data_dir}/TrainSet{data_conf}.parquet.gzip')
            dfyTrain.to_parquet(f'{save_data_dir}/yTrain{data_conf}.parquet.gzip')
            dfTestSet.to_parquet(f'{save_data_dir}/TestSet{data_conf}.parquet.gzip')
            dfyTest.to_parquet(f'{save_data_dir}/yTest{data_conf}.parquet.gzip')
    
            #dfApplicationDf.to_parquet(f'{save_data_dir}/ApplicationDf{data_conf}.parquet.gzip')

    
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

    if True:
        particle_dict = config['output']['particle']
        isV0 = config['input']['isV0']
        hybrid = config['input']['hybrid']

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

        if hybrid:                  output_dir += '/hybrid'
        elif isV0:                  output_dir += '/V0'
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
        for name in particle_dict.values():

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

    if True:
        isV0 = config['input']['isV0']
        hybrid = config['input']['hybrid']

        RegressionColumns = config['training']['RegressionColumns']
        final_output_dir = config['output']['final_dir']
        delta_output_dir = config['output']['ml_dir']
        particle_dict = config['output']['particle']

        do_augm = config['data_prep']['do_augm']
        beta_flat = config['training']['beta_flat']
        beta_p_flat = config['training']['beta_p_flat']
        do_equal = config['data_prep']['do_equal']

        if hybrid:        
            final_output_dir += '/hybrid'
            delta_output_dir += '/hybrid'
        elif isV0:        
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
    density_scatter(ApplicationDf['p'], ApplicationDf['dedx'], f'{final_output_dir}/dex_vs_p', plot_specifics=plot_specifics['dedx_vs_p'])

    # beta from TPC (some candidates will be erased)
    appl_list = [filtering(ApplicationDf, name, mass=mass_dict[name], label=False)for name in particle_dict.values()]
    ApplicationDf = pd.concat(appl_list)
    ApplicationDf.eval('Delta = (beta - preds)/beta', inplace=True)

    # redefine pi
    ApplicationDf['label'] = np.where(ApplicationDf['nSigmaPiAbs'] < 1, 'Pi', 
                                np.where(ApplicationDf['nSigmaKAbs'] > 3, 'Pi', 
                                    np.where(ApplicationDf['nSigmaPAbs'] > 3, 'Pi', 
                                        np.where(ApplicationDf['nSigmaEAbs'] > 3, 'Pi', ApplicationDf['label']) ) ) )
    ApplicationDf['particle'] = np.where(ApplicationDf['nSigmaPiAbs'] < 1, 3, 
                                np.where(ApplicationDf['nSigmaKAbs'] > 3, 3, 
                                    np.where(ApplicationDf['nSigmaPAbs'] > 3, 3, 
                                        np.where(ApplicationDf['nSigmaEAbs'] > 3, 3, ApplicationDf['particle']) ) ) )

    for name in particle_dict.values():   
        df = ApplicationDf.query(f'label == "{name}"', inplace=False)  
        
        df1 = df.query('0.4 <= p < 0.45')
        for i in range(7):  df1.drop( df1[df1[f'ClSizeL{i}'] < 0].index, inplace=True )
        
        density_scatter(df1['preds'], df1['clSizeCosLam'], f'{final_output_dir}/clSizeCosLam_vs_beta_ricostruita_{name}', ["#beta", "clSizeCosLam", 100, 0, 1, 120, 0, 12])
        density_scatter(df1['Delta'], df1['clSizeCosLam'], f'{final_output_dir}/clSizeCosLam_vs_delta_{name}', ["#Delta", "clSizeCosLam", 200, -1, 1, 120, 0, 12])

        density_scatter(df['p'], df['preds'], f'{output_file}_{name}', plot_specifics['b_vs_p_final'], title=f'beta_vs_p_final_{name}')
        density_scatter(df['p'], df['beta'], f'{final_output_dir}/betatrue_vs_p_{name}', plot_specifics=plot_specifics['b_vs_p_final'], title=f'beta_{name}')
    density_scatter(ApplicationDf['p'], ApplicationDf['beta'], f'{final_output_dir}/beta_vs_p_total', plot_specifics=plot_specifics['b_vs_p_final'], title=f'beta_true_total')  
    


    # Delta 
    #________________________________________________

    plot_score(ApplicationDf, ApplicationDf['p'], RegressionColumns, model, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{delta_output_dir}/Appl_p_', absolute=False)
    plot_score(ApplicationDf, ApplicationDf['beta'], RegressionColumns, model, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{delta_output_dir}/Appl_', absolute=False)

    for name in particle_dict.values():
        X_name = ApplicationDf.query(f"label == '{name}'", inplace=False)

        plot_score(X_name, X_name['p'], RegressionColumns, model, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{delta_output_dir}/Appl_p_{name}', absolute=False)
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