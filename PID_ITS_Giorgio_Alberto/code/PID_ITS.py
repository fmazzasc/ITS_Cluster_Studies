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

import uproot
from ROOT import TH2F, TCanvas, TH1F, gStyle, kBird, gROOT, TFile, TH3F

from UsefulFunctions import filtering, multiple_hist, density_scatter, equal, augmentation_fine, callback, Delta_score, plot_score, plot_score_train, hist, delta_scorer
from UsefulClasses import SingleVariable_Flattener, DoubleVariable_Flattener, TH2Handler

gROOT.SetBatch()

# Tags
#_____________________________________
tag_Deu = 'nSigmaDeuAbs < 1 and nSigmaPAbs > 3 and nSigmaKAbs > 3 and nSigmaPiAbs > 3 and p <= 1.2'
tag_P = 'nSigmaDeuAbs > 3 and nSigmaPAbs < 1 and nSigmaKAbs > 3 and nSigmaPiAbs > 3 and p <= 0.7'
tag_K = 'nSigmaDeuAbs > 3 and nSigmaPAbs > 3 and nSigmaKAbs < 1 and nSigmaPiAbs > 3  and p <= 0.7'
tag_Pi = 'nSigmaDeuAbs > 3 and nSigmaPAbs > 3 and nSigmaKAbs > 3 and nSigmaPiAbs < 1  and p <= 0.7'
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

MC_dict = {2212: 1, 321: 2, 211: 3, 11: 4}







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
        isMC = config['input']['isMC']
        MCtpc = config['input']['MCtpc']
        ext_appl = config['input']['ext_appl']
        RegressionColumns = config['training']['RegressionColumns']
        RegressionRanges = config['training']['RegressionRanges']

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
        MCweights = config['training']['MCweights']
        do_equal = config['data_prep']['do_equal']
        save_data = config['data_prep']['save_data']
        save_data_dir = config['output']['save_data_dir']

        if hybrid:
            output_dir += '/hybrid'
            save_data_dir += '/hybrid'
        elif MCtpc:
            output_dir += '/MCtpc'
            save_data_dir += '/MCtpc'
        elif isMC:
            output_dir += '/MC'
            save_data_dir += '/MC'
        elif isV0:                    
            output_dir += '/V0'
            save_data_dir += '/V0'
        else:                       
            output_dir += '/TPC'
            save_data_dir += '/TPC'

        if do_augm and beta_p_flat: output_dir += '/augm'
        elif do_augm and beta_flat: output_dir += '/augm_betaflat'
        elif do_augm:               output_dir += '/augm'
        elif MCweights:             output_dir += '/MCweights'
        elif beta_flat:             output_dir += '/betaflat'
        elif beta_p_flat:           output_dir += '/beta_pflat'
        elif do_equal:              output_dir += '/equal'
        else:                       output_dir += '/no_options'

        options = ''
        if do_augm:                 options += '_augm'
        if beta_flat:               options += '_betaflat_'
        if beta_p_flat:             options += '_beta_pflat_'
        if do_equal:                options += '_equal_'



    input_data = config['input']['data']
    print(f'input data: {input_data}')
    if isMC:        RegressionDf = uproot.open(input_data)['ITStreeML'].arrays(library='pd')
    else:           RegressionDf = pd.read_parquet(input_data)
    if ext_appl:    ApplicationDf = pd.read_parquet(config['input']['appl_data'])


    # Define some new columns
    #_________________________________

    if isMC:    RegressionDf.eval('p = pMC', inplace=True)
    RegressionDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
    RegressionDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
    RegressionDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)
    
    if not isMC:    RegressionDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)  

    if ext_appl:
        ApplicationDf.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
        ApplicationDf.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
        ApplicationDf.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)  
        if not isMC or MCtpc:    ApplicationDf.eval('delta_p = (p - pTPC)/pTPC', inplace=True)

    if isMC:    # PDG group label -> personal label
        temp = []
        for i, (key, value) in enumerate(MC_dict.items()):   
            temp.append(RegressionDf.query(f'abs(pdgMC) == {key}', inplace=False).reset_index(drop=True))
            temp[i]['particle'] = value
        RegressionDf = pd.concat(temp)

    if not isMC:    
        for part in particle_dict.values():  RegressionDf[f'nSigma{part}Abs'] = abs(RegressionDf[f'nSigma{part}'])
    
    if not isMC or MCtpc:
        if ext_appl:    
            for part in particle_dict.values():  ApplicationDf[f'nSigma{part}Abs'] = abs(ApplicationDf[f'nSigma{part}'])

    if isMC and not MCtpc:    RegressionDf, ApplicationDf = train_test_split(RegressionDf, test_size=0.5, random_state=seed_split)

    
    # Splitting, filtering (and evaluating beta)
    #__________________________________
    
    if not isMC:
        RegressionDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)  

    if not isMC or MCtpc:      
        if ext_appl:    ApplicationDf.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)
    else:
        RegressionDf.query('p <= 50', inplace=True)        
        if ext_appl:    ApplicationDf.query('p <= 50', inplace=True)

    if not isV0:
        RegressionDf.query('0.1 < p <= 0.7', inplace=True)
        ApplicationDf.query('0.1 < p <= 0.7', inplace=True)

    # Cut in momentum for training set
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
        dfs_train = [filtering(TrainSet, name, mass=mass_dict[name], label=False) for name in particle_dict.values()]
        dfs_test = [filtering(TestSet, name, mass=mass_dict[name], label=False) for name in particle_dict.values()]

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

        df1 = filt_dfs[2].query('0.4 < p < 0.5', inplace=False)
        for var in hist_spec:   multiple_hist([df1], var, hist_spec[var], f'{output_dir}/Pi_')
        #density_scatter(df1['p'], df1['dedx'], f'{output_dir}/dedx_all', plot_spec_scat[0], title='dedx_all') # -----------------

        plot_x_scat = config['plots']['plot_x_scat']
        plot_y_scat = config['plots']['plot_y_scat']
        plot_spec_scat = config['plots']['plot_spec_scat']

        for x, y, scat_spec in zip(plot_x_scat, plot_y_scat, plot_spec_scat):
            for name, df in zip(particle_dict.values(), filt_dfs): 
                density_scatter(df[x], df[y], f'{output_dir}/{y}_vs_{x}_{name}', scat_spec, title=f'{y}_{name}')
            density_scatter(total_df[x], total_df[y], f'{output_dir}/{y}_vs_{x}_total', scat_spec, title=f'{y}_total')

        if not isMC:    
            density_scatter(RegressionDf['p'], RegressionDf['dedx'], f'{output_dir}/dedx_all', plot_spec_scat[0], title='dedx_all')
            density_scatter(RegressionDf['p'], RegressionDf['clSizeCosLam'], f'{output_dir}/clSizeCosLam_all', plot_spec_scat[2], title='clSizeCosLam_all')
        else:   density_scatter(RegressionDf['p'], RegressionDf['clSizeCosLam'], f'{output_dir}/clSizeCosLam_all', plot_spec_scat[1], title='clSizeCosLam_all')
        
        if isMC:
            MC_vars = ['clSizeCosLam', 'ClSizeL0', 'ClSizeL1', 'ClSizeL2', 'ClSizeL3', 'ClSizeL4', 'ClSizeL5', 'ClSizeL6']    
            reduced_dfs = [df.query('0.3 < p < 0.4', inplace=False) for df in filt_dfs]
            for var in hist_spec:   multiple_hist(reduced_dfs, var, hist_spec[var], f'{output_dir}/0304_')


        # Check hypotesis 
        check_dfs = []
        for name in particle_dict.values():         check_dfs.append( total_df.query(f"label == '{name}' and -0.1 <= SnPhiL0 <= 0.1 and -0.1 <= tgL <= 0.1") )
        for df, name in zip(check_dfs, names):      
            if not isMC:    density_scatter(df['beta'], df['ClSizeL0'], f'{output_dir}/check_on_ClSizeL0_{name}', plot_spec_scat[3], title=f'check_on_ClSizeL0_{name}')
            else:           density_scatter(df['beta'], df['ClSizeL0'], f'{output_dir}/check_on_ClSizeL0_{name}', plot_spec_scat[2], title=f'check_on_ClSizeL0_{name}')

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
                else:               nn_weights.append(n_weight * 1000000)

            TrainSet['beta_pweight'] = np.select(conditions, nn_weights)

            plot_spec_flat = config['plots']['bp_flat_scat']
            density_scatter(TrainSet['p'], TrainSet['particle'], f'{output_dir}/beta_pflat_momentum', plot_specifics=plot_spec_flat['p'])
            density_scatter(TrainSet['beta_pweight'], TrainSet['particle'], f'{output_dir}/beta_pflat_weights', plot_specifics=plot_spec_flat['weights'])
            density_scatter(TrainSet['p'], TrainSet['particle'], f'{output_dir}/beta_pflat_weighted_momentum', plot_specifics=plot_spec_flat['p'], weights=TrainSet['beta_pweight'])

    if MCweights:
        with alive_bar(title='3D weights...') as bar:
            

            ################################    FRANCESCO
            if False:

                h1 = TH3F('h1', 'h1', 250, 0, 25, 15, 0.0, 1.5, 5, 0, 5)
                for x, y, z in zip(TrainSet['clSizeCosLam'], TrainSet['p'], TrainSet['particle']):  h1.Fill(x, y, z)

                h2 = TH3F('h2', 'h2', 250, 0, 25, 15, 0.0, 1.5, 5, 0, 5)
                df = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520143_itstpc.parquet')
                df.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
                df.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)  
                for x, y, z in zip(df['clSizeCosLam'], df['p'], df['particle']):  h2.Fill(x, y, z)

                h2.Divide(h1)

                #h2.Draw()
                #file = TFile("TH3.root", "recreate")
                #h2.Write()
                #file.Close()

                weights = []
                conditions = []

                for xbin in range(1, h2.GetNbinsX()+1):
                    xmin = h2.GetXaxis().GetBinLowEdge(xbin)
                    xmax = xmin + h2.GetXaxis().GetBinWidth(xbin)

                    for ybin in range(1, h2.GetNbinsY()+1):
                        ymin = h2.GetYaxis().GetBinLowEdge(ybin)
                        ymax = ymin + h2.GetYaxis().GetBinWidth(ybin)

                        for zbin in range(1, h2.GetNbinsZ()+1):

                            if zbin != 1:
                                weights.append(h2.GetBinContent(h2.GetBin(xbin, ybin, zbin)))
                                conditions.append((TrainSet['particle'] == zbin-1) & (TrainSet['clSizeCosLam'] >= xmin) & (TrainSet['clSizeCosLam'] < xmax)
                                                & (TrainSet['p'] >= ymin) & (TrainSet['p'] < ymax))

                TrainSet['3d_weight'] = np.select(conditions, weights)
                del h1, h2

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

            ################################    FRANCESCO 2
            if False:

                h1 = TH2F('h1', 'h1', 75, 0.0, 1.5, 5, 0, 5)
                for x, y in zip(TrainSet['p'], TrainSet['particle']):   h1.Fill(x, y)

                h2 = TH3F('h2', 'h2', 250, 0, 25, 75, 0.0, 1.5, 5, 0, 5)
                for x, y, z in zip(TrainSet['clSizeCosLam'], TrainSet['p'], TrainSet['particle']):  h2.Fill(x, y, z)

                h3 = TH3F('h3', 'h3', 250, 0, 25, 75, 0.0, 1.5, 5, 0, 5)
                df = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520143_itstpc.parquet')
                df.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
                df.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True)  
                for x, y, z in zip(df['clSizeCosLam'], df['p'], df['particle']):  h3.Fill(x, y, z)

                h3.Divide(h2)

                #h2.Draw()
                #file = TFile("TH3.root", "recreate")
                #h2.Write()
                #file.Close()

                weights = []
                conditions = []

                for xbin in range(1, h3.GetNbinsX()+1):
                    xmin = h3.GetXaxis().GetBinLowEdge(xbin)
                    xmax = xmin + h3.GetXaxis().GetBinWidth(xbin)

                    for ybin in range(1, h3.GetNbinsY()+1):
                        ymin = h3.GetYaxis().GetBinLowEdge(ybin)
                        ymax = ymin + h3.GetYaxis().GetBinWidth(ybin)

                        for zbin in range(1, h3.GetNbinsZ()+1):
                            
                            if h1.GetBinContent(h1.GetBin(ybin, zbin)) == 0:    h1_inv_val = 0.
                            else:                                               h1_inv_val = 1000./h1.GetBinContent(h1.GetBin(ybin, zbin))
                            w = h3.GetBinContent(h3.GetBin(xbin, ybin, zbin)) * h1_inv_val

                            if zbin != 1:
                                weights.append(w)
                                conditions.append((TrainSet['particle'] == zbin-1) & (TrainSet['clSizeCosLam'] >= xmin) & (TrainSet['clSizeCosLam'] < xmax)
                                                & (TrainSet['p'] >= ymin) & (TrainSet['p'] < ymax))

                TrainSet['3d_weight'] = np.select(conditions, weights)
                del h1, h2, h3

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

            ##################################  STEFANO
            if False:
                
                #beta_p_flat preliminare

                df = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520143_itstpc.parquet')
                df.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
                df.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True) 

                pmins = np.linspace(0, 1.4, num=15)
                pmaxs = np.linspace(0.1, 1.5, num=15)

                N = len(df)
                weights = []
                conditions = []

                for num in particle_dict.keys():  
                    for i, (pmin, pmax) in enumerate(zip(pmins, pmaxs)):    
                        weights.append(len(df.query(f'particle == {num} and {pmin} <= p < {pmax}'))/N)
                        conditions.append((df['particle'] == num) & (df['p'] >= pmin) & (df['p'] < pmax))


                n_weights = []
                for weight in weights:
                    if weight == 0.:    n_weights.append(0.)
                    else:               n_weights.append(1./weight)

                df['beta_pweight'] = np.select(conditions, n_weights)

                weights.clear()
                conditions.clear()

                

                for num in particle_dict.keys():  
                    h1 = TH1F('h1', 'h1', 250, 0, 25)
                    for x in TrainSet.query(f'particle=={num}', inplace=False)['clSizeCosLam']:  h1.Fill(x)

                    h2 = TH1F('h2', 'h2', 250, 0, 25) 
                    for x, w in zip(df.query(f'particle=={num}', inplace=False)['clSizeCosLam'], df.query(f'particle=={num}', inplace=False)['beta_pweight']):  h2.Fill(x, w)

                    h2.Divide(h1)

                    for xbin in range(1, h2.GetNbinsX()+1):
                        xmin = h2.GetXaxis().GetBinLowEdge(xbin)
                        xmax = xmin + h2.GetXaxis().GetBinWidth(xbin)

                        weights.append(h2.GetBinContent(xbin))
                        conditions.append((TrainSet['particle'] == num) & (TrainSet['clSizeCosLam'] >= xmin) & (TrainSet['clSizeCosLam'] < xmax))
                    
                TrainSet['3d_weight'] = np.select(conditions, weights)

                pmins = np.linspace(0, 1.4, num=15)
                pmaxs = np.linspace(0.1, 1.5, num=15)

                dfs = [TrainSet.query(f'particle == {num}', inplace=False) for num in particle_dict.keys()]
                dfs2 = [df.query(f'particle == {num}', inplace=False) for num in particle_dict.keys()]

                multiple_hist(dfs, 'clSizeCosLam', [250, 0, 25], f'{output_dir}/3d_INT_CLSIZE_', weights='3d_weight')
                multiple_hist(dfs2, 'clSizeCosLam', [250, 0, 25], f'{output_dir}/TOCOMPARE_INT_CLSIZE_', weights='beta_pweight')


                #dfs: list[list[pd.DataFrame]] = [[] for i in range(4)]
                #dfs2: list[list[pd.DataFrame]] = [[] for i in range(4)]
                #for num, df_list, df_list2 in zip(range(1, 5), dfs, dfs2):
                #    for pmin, pmax in zip(pmins, pmaxs):
                #        df_list.append(pd.DataFrame(TrainSet.query(f'particle == {num} and {pmin} <= p < {pmax}', inplace=False)))
                #        df_list2.append(pd.DataFrame(df.query(f'particle == {num} and {pmin} <= p < {pmax}', inplace=False)))
#
                #for name, df_list in zip(particle_dict.values(), dfs):  multiple_hist(df_list, 'clSizeCosLam', [250, 0, 25], f'{output_dir}/3d_{name}_', weights='3d_weight')
                #for name, df_list in zip(particle_dict.values(), dfs2):  multiple_hist(df_list, 'clSizeCosLam', [250, 0, 25], f'{output_dir}/TOCOMPARE_{name}_', weights='beta_pweight')

                #plot_spec_flat = config['plots']['bp_flat_scat']
                #density_scatter(TrainSet['p'], TrainSet['particle'], f'{output_dir}/3d_weighted_momentum', plot_specifics=plot_spec_flat['p'], weights=TrainSet['3d_weight'])
                    
            ################## GIORGIO

            if True:

                df = pd.read_parquet('/data/shared/ITS/ML/particles_pid_520143_itstpc.parquet')
                df.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
                df.query('p <= 50 and 20 < rofBC < 500 and tpcITSchi2 < 5 and nClusTPC > 100 and -0.2 < delta_p < 0.2', inplace=True) 

                #dfs = [df.query(f'particle == {num}', inplace=False) for num in particle_dict.keys()]
                #for dfi, name in zip(dfs, particle_dict.values()): 
                #density_scatter(df['p'], df['dedx'], f'{output_dir}/ITSTPC_dedx', ["p", "#frac{dE}{dx}", 150, 0, 1.5, 600, 0, 600])

                for name in particle_dict.values():     df[f'nSigma{name}Abs'] = abs(df[f'nSigma{name}'])

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

                for num in particle_dict.keys():  
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
        isMC = config['input']['isMC']
        MCtpc = config['input']['MCtpc']

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
        MCweights = config['training']['MCweights']
        do_equal = config['data_prep']['do_equal']

        output_dir = config['output']['ml_dir']
        save_model = config['training']['save_model']

        options = ''
        if do_augm:                 options += '_augm'
        if beta_p_flat:             options += '_beta_pflat_'
        if beta_flat:               options += '_betaflat_'
        if do_equal:                options += '_equal'

        if hybrid:                  output_dir += '/hybrid'
        elif MCtpc:                 output_dir += '/MCtpc'
        elif isMC:                  output_dir += '/MC'
        elif isV0:                  output_dir += '/V0'
        else:                       output_dir += '/TPC'

        if do_augm and beta_p_flat: output_dir += '/augm'
        elif do_augm and beta_flat: output_dir += '/augm_betaflat'
        elif do_augm:               output_dir += '/augm'
        elif MCweights:             output_dir += '/MCweights'
        elif beta_flat:             output_dir += '/betaflat'
        elif beta_p_flat:           output_dir += '/beta_pflat'
        elif do_equal:              output_dir += '/equal'
        else:                       output_dir += '/no_options'

        if do_equal:    names.remove('Deu')
    



    # Model definition
    #__________________________________
    if model_choice=='xgboost':     model = xgb.XGBRegressor(random_state=random_state, n_jobs=40)
    if model_choice=='automl':      model = AutoML()




    # Optuna optimization
    #__________________________________
    
    if do_opt:
        model_handler = ModelHandler(model, RegressionColumns)
        model_handler.set_model_params(ModelParams)

        print('\nInitialize Optuna hyper-parameters optimization...')
        with alive_bar(title='Hyper-Parameters optimization') as bar:
            if early_stop:      study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='minimize', fit_params={'sample_weight': X_train['beta_pweight']}, callbacks=[callback])
            elif beta_flat:     study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='minimize', fit_params={'sample_weight': X_train['beta_weight']}, timeout=300)    
            elif beta_p_flat:   study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='minimize', fit_params={'sample_weight': X_train['beta_pweight']}, timeout=300)
            elif MCweights:     study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='minimize', fit_params={'sample_weight': X_train['3d_weight']}, timeout=300)
            #else:               study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='minimize', fit_params={'sample_weight': X_train['beta_pweight']}, n_trials=50)
            else:               study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='minimize', timeout=300)
            #if early_stop:  study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, Delta_score, direction='minimize', callbacks=[callback])
            #else:           study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='minimize', timeout=300)


        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        #fig = optuna.visualization.plot_optimization_history(study)
        #fig.write_image(f'{output_dir}/plot_optimization_history{options}.png')
#
        #fig = optuna.visualization.plot_param_importances(study)
        #fig.write_image(f'{output_dir}/plot_param_importances{options}.png')
#
        #fig = optuna.visualization.plot_parallel_coordinate(study)
        #fig.write_image(f'{output_dir}/plot_parallel_coordinate{options}.png')
#
        #fig = optuna.visualization.plot_contour(study)
        #fig.write_image(f'{output_dir}/plot_contour{options}.png')

        HyperParams = study.best_trial.params

    else:   HyperParams = ModelParams

    



    # Model evaluation
    #__________________________________
    
    if model_choice=='xgboost':

        print('\nXGB model...')
        if beta_flat:   print('Beta weights selected...')
        if beta_p_flat: print('Beta and momentum weights selected...')

        if do_opt:  model_reg = xgb.XGBRegressor(**HyperParams, 
                                                n_jobs=40,
                                                #tree_method="gpu_hist", 
                                                random_state=random_state)
        else:       model_reg = xgb.XGBRegressor(n_jobs=40,
                                                #tree_method="gpu_hist", 
                                                random_state=random_state)
        plot_specifics = config['plots']['model_train']
        with alive_bar(title='Training...') as bar:     
            if beta_flat:       model_reg.fit(X_train[RegressionColumns], y_train, sample_weight=X_train['beta_weight'])
            elif beta_p_flat:   model_reg.fit(X_train[RegressionColumns], y_train, sample_weight=X_train['beta_pweight'])
            elif MCweights:     model_reg.fit(X_train[RegressionColumns], y_train, sample_weight=X_train['3d_weight'])
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
        print(f'Model saved in {model_out}/RegressorModel_{model_choice}{options}.pickle.')

    # Feature importance plot
    #________________________________

    if model_choice == 'xgboost':
        pass
        #FeatureImportance = xgb.plot_importance(model_reg)
        #plt.savefig(f'{output_dir}/FeatureImportance_{model_choice}{options}.png')
        #plt.close('all')


    return model_reg




def application(ApplicationDf, config, model):

    # Upload settings from configuration file 
    #_________________________________________________

    if True:
        isV0 = config['input']['isV0']
        hybrid = config['input']['hybrid']
        isMC = config['input']['isMC']
        MCtpc = config['input']['MCtpc']

        RegressionColumns = config['training']['RegressionColumns']
        final_output_dir = config['output']['final_dir']
        delta_output_dir = config['output']['ml_dir']
        particle_dict = config['output']['particle']

        do_augm = config['data_prep']['do_augm']
        beta_flat = config['training']['beta_flat']
        beta_p_flat = config['training']['beta_p_flat']
        MCweights = config['training']['MCweights']
        do_equal = config['data_prep']['do_equal']

        if hybrid:        
            final_output_dir += '/hybrid'
            delta_output_dir += '/hybrid'
        elif MCtpc:        
            final_output_dir += '/MCtpc'
            delta_output_dir += '/MCtpc'
        elif isMC:        
            final_output_dir += '/MC'
            delta_output_dir += '/MC'
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
        elif MCweights:
            final_output_dir += '/MCweights'            
            delta_output_dir += '/MCweights'
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
    if not isMC:    density_scatter(ApplicationDf['p'], ApplicationDf['dedx'], f'{final_output_dir}/dex_vs_p', plot_specifics=plot_specifics['dedx_vs_p'])

    # beta from TPC (some candidates will be erased)
    new_tag_Pi = 'nSigmaPiAbs < 1 and nSigmaKAbs > 3 and nSigmaPAbs > 3 and nSigmaEAbs > 3'     # redefine pi
    appl_list = []
    for name in particle_dict.values():
        #if name == "Pi":    appl_list.append(filtering(ApplicationDf, name, tag=new_tag_Pi, mass=mass_dict[name], label=False)) 
        #else:               
        appl_list.append(filtering(ApplicationDf, name, mass=mass_dict[name], label=False))

    ApplicationDf = pd.concat(appl_list)
    ApplicationDf.eval('Delta = (beta - preds)/beta', inplace=True)


    # Beta vs p
    #________________________________________________

    
    for name in particle_dict.values():   
        df = ApplicationDf.query(f'label == "{name}"', inplace=False)  
        #if name == "Pi":    print(df.nSigmaPiAbs.describe(), df.nSigmaKAbs.describe(), df.nSigmaPAbs.describe(), df.nSigmaEAbs.describe())
        if not isMC:
            if name == "Pi":
                plot_score(df, df['beta'], RegressionColumns, model, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{delta_output_dir}/Appl_UNCUT_{name}', absolute=False)
                density_scatter(df['p'], df['preds'], f'{output_file}_UNCUT_{name}', plot_specifics['b_vs_p_final'], title=f'beta_vs_p_final_{name}')
    
                df_pi = df.query('nSigmaPAbs > 3 and nSigmaKAbs > 3', inplace=False)
                density_scatter(df_pi['p'], df_pi['preds'], f'{output_file}_more_select_{name}', plot_specifics['b_vs_p_final'], title=f'beta_vs_p_final_{name}')
    
                # Study on πs
                df_alone = df.query('Delta > 0.18', inplace=False)
                df_alone.query('p < 0.55', inplace=True)
    
                hist_spec = config['plots']['plot_spec_hist']
                for var in hist_spec:   multiple_hist([df_alone], var, hist_spec[var], f'{final_output_dir}/Pi_alone_')
                density_scatter(df_alone['p'], df_alone['clSizeCosLam'], f'{final_output_dir}/Pi_alone_clSizeCosLam_vs_p_ricostruita_{name}', ["p", "clSizeCosLam", 55, 0, 0.55, 120, 0, 12])
    
                df_not_alone = df.query('Delta <= 0.18', inplace=False)
                df_not_alone.query('p < 0.55', inplace=True)
    
                hist_spec = config['plots']['plot_spec_hist']
                for var in hist_spec:   multiple_hist([df_not_alone], var, hist_spec[var], f'{final_output_dir}/Pi_not_alone_')
                density_scatter(df_not_alone['p'], df_not_alone['clSizeCosLam'], f'{final_output_dir}/Pi_not_alone_clSizeCosLam_vs_p_ricostruita_{name}', ["p", "clSizeCosLam", 55, 0, 0.55, 120, 0, 12])
    
                #--------------
    
                df2_alone = df_alone.query('0.4 < p < 0.5', inplace=False)
                for var in hist_spec:   multiple_hist([df2_alone], var, hist_spec[var], f'{final_output_dir}/Pi_alone_0405_')
                df2_not_alone = df_not_alone.query('0.4 < p < 0.5', inplace=False)
                for var in hist_spec:   multiple_hist([df2_not_alone], var, hist_spec[var], f'{final_output_dir}/Pi_not_alone_0405_')
                hist(df2_alone['clSizeCosLam'], f'{final_output_dir}/clSizeCosLam_0405_alone', ["clSizeCosLam", 120, 0, 12])
        
        
            

        df1 = df.query('0.4 <= p < 0.45')
        for i in range(7):  df1.drop( df1[df1[f'ClSizeL{i}'] < 0].index, inplace=True )
        
        density_scatter(df1['preds'], df1['clSizeCosLam'], f'{final_output_dir}/clSizeCosLam_vs_beta_ricostruita_{name}', ["#beta", "clSizeCosLam", 100, 0, 1, 120, 0, 12])
        density_scatter(df1['Delta'], df1['clSizeCosLam'], f'{final_output_dir}/clSizeCosLam_vs_delta_{name}', ["#Delta", "clSizeCosLam", 200, -1, 1, 120, 0, 12])

        density_scatter(df['p'], df['clSizeCosLam'], f'{final_output_dir}/clSizeCosLam_vs_p_total_{name}', plot_specifics=config['plots']['plot_spec_scat'][2])  
        density_scatter(df['p'], df['preds'], f'{output_file}_{name}', plot_specifics['b_vs_p_final'], title=f'beta_vs_p_final_{name}')
        density_scatter(df['p'], df['beta'], f'{final_output_dir}/betatrue_vs_p_{name}', plot_specifics=plot_specifics['b_vs_p_final'], title=f'beta_{name}')
    
    density_scatter(ApplicationDf['p'], ApplicationDf['clSizeCosLam'], f'{final_output_dir}/clSizeCosLam_vs_p_total', plot_specifics=config['plots']['plot_spec_scat'][2])  
    density_scatter(ApplicationDf['p'], ApplicationDf['beta'], f'{final_output_dir}/beta_vs_p_total', plot_specifics=plot_specifics['b_vs_p_final'], title=f'beta_true_total')  
    


    # Delta 
    #________________________________________________

    plot_score(ApplicationDf, ApplicationDf['p'], RegressionColumns, model, x_label='p', plot_specifics=plot_specifics['p'], filename=f'{delta_output_dir}/Appl_p_', absolute=False)
    plot_score(ApplicationDf, ApplicationDf['beta'], RegressionColumns, model, x_label='#beta', plot_specifics=plot_specifics['beta'], filename=f'{delta_output_dir}/Appl_', absolute=False)

    th2_handler = TH2Handler(ApplicationDf, 'p', 'Delta')
    th2_handler.build_th2(150, 0., 1.5, 300, -1.5, 1.5)
    th2_handler.TH2toLine(f'{delta_output_dir}/Delta_line_P', 'y', 1)

    
    for name in particle_dict.values():
        X_name = ApplicationDf.query(f"label == '{name}' and p <= 1.5", inplace=False)

        th2_handler = TH2Handler(X_name, 'p', 'Delta')
        th2_handler.build_th2(150, 0., 1.5, 300, -1.5, 1.5)
        th2_handler.TH2toLine(f'{delta_output_dir}/Delta_line_{name}', 'y', 1)

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
    if not skip_appl:   application(ApplicationDf, config, Model)     

    del TrainTestData, ApplicationDf



start_time = time()

main()

passed_time = time() - start_time
print(f'\nTime: {passed_time/60} min')