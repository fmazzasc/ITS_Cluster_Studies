#
#   First implementation. OUTDATED. Left for reference
#

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

import uproot
from ROOT import TH2F, gROOT, TFile, TH3F

from UsefulFunctions import filtering, multiple_hist, density_scatter, equal, callback, delta_scorer, plot_score_train
from UsefulClasses import DataAugmenter, SingleVariable_Flattener, DoubleVariable_Flattener, TH2Handler, Plotter, Scorer, PrepConstructor

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

MC_dict = {2212: 1, 321: 2, 211: 3, 11: 4}

#########################

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
        mode = config['mode']
        ext_appl = config[mode]['input']['ext_appl']

        particle_dict = config[mode]['output']['particle'] 

        do_plots = config[mode]['plots']['do_plots']
        output_dir = config[mode]['output']['data_visual_dir']
        test_size = config[mode]['data_prep']['test_size']
        random_state = config[mode]['data_prep']['random_state']

        seven_hits = config[mode]['data_prep']['seven_hits']  # only consider candidates with hits on all the layers

        do_augm = config[mode]['data_prep']['do_augm']
        beta_flat = config[mode]['training']['beta_flat']
        beta_p_flat = config[mode]['training']['beta_p_flat']
        MCweights = config[mode]['training']['MCweights']
        do_equal = config[mode]['data_prep']['do_equal']
        save_data = config[mode]['data_prep']['save_data']
        save_data_dir = config[mode]['output']['save_data_dir']

        if do_augm and beta_p_flat: output_dir += f'/{mode}_augm'
        elif do_augm and beta_flat: output_dir += f'/{mode}_augm_betaflat'
        elif do_augm:               output_dir += f'/{mode}_augm'
        elif MCweights:             output_dir += f'/{mode}_MCweights'
        elif beta_flat:             output_dir += f'/{mode}_betaflat'
        elif beta_p_flat:           output_dir += f'/{mode}_beta_pflat'
        elif do_equal:              output_dir += f'/{mode}_equal'
        else:                       output_dir += f'/{mode}_no_options'

    fimpPath = config[mode]['input']['data']
    applPath = None
    if ext_appl:    applPath = config[mode]['input']['appl_data']

    print(f'input data: {fimpPath}')

    prep_con = PrepConstructor()
    prep = prep_con.CreatePrepTool(mode, fimpPath, applPath)
    prep.Preprocess(particle_dict, mass_dict, tag_dict, test_size, random_state)

    TrainSet, TestSet, ApplicationDf = prep.returnDfs()
    TotalSet = pd.concat([TrainSet, TestSet])


    # Data Visualization
    #_________________________________
    if do_plots:
        with alive_bar(title='\nData Visualization...') as bar:
            
            file = TFile(f'{output_dir}.root', 'recreate')
            print(f'ROOT file created in {output_dir}.root')
            plot = Plotter(TotalSet, file)

            plotSpec1Ddict = config[mode]['plots']['plotSpec1D']
            varsToPlot1D = [var for var in plotSpec1Ddict.keys()]
            plotSpec1D = [spec for spec in plotSpec1Ddict.values()]
            plot.plot1D(varsToPlot1D, plotSpec1D)

            plotAxis2Ddict = config[mode]['plots']['plotAxis2D']
            plotSpec2Ddict = config[mode]['plots']['plotSpec2D']
            varsToPlot2D = [var for var in plotAxis2Ddict.values()]
            xsToPlot2D = [item[0] for item in varsToPlot2D]
            ysToPlot2D = [item[1] for item in varsToPlot2D]
            plotSpec2D = [spec for spec in plotSpec2Ddict.values()]
            plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)

            plot.multi_df('label', particle_dict.values())
            plot.plot1D(varsToPlot1D, plotSpec1D)
            plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)
            file.Close()

            file = TFile(f'{output_dir}_0304.root', 'recreate')
            plot = Plotter(TotalSet.query('0.3 < p < 0.4', inplace=False), file)

            plot.multi_df('label', particle_dict.values())
            plot.plot1D(varsToPlot1D, plotSpec1D)
            file.Close()


        # Correlation between dataframe features
        #___________________________________

        
        if mode == 'TPC':   filt_dfs = [filtering(TotalSet, name, tag=tag_dict[name], mass=mass_dict[name]) for name in particle_dict.values()]
        else:               filt_dfs = [filtering(TotalSet, name, mass=mass_dict[name], label=False) for name in particle_dict.values()]

        CorrMatrixFig = plot_utils.plot_corr(filt_dfs, varsToPlot1D, names)
        for Fig, name in zip(CorrMatrixFig, names):
            plt.figure(Fig.number)
            plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
            Fig.savefig(f'{output_dir}_CorrMatrix_{name}.png')
    
    TrainSet, yTrain, TestSet, yTest = prep.Preprocess2(seven_hits)
    ApplicationDf = prep.ApplicationDf
    

    # Data augmentation
    #_________________________________

    if do_augm:
        print('\nData Augmentation...')
        TrainSet['copy'] = 0

        dt_augm = DataAugmenter(TrainSet, daughters=config[mode]['data_prep']['to_augm'], mothers=config[mode]['data_prep']['mothers'], p_ranges=config[mode]['data_prep']['p_ranges'], mass_dict=mass_dict)
        TrainSet = dt_augm.do_augm()
        dt_augm.print_augm()
        yTrain = TrainSet['beta']

        # Plots after augmentation
        if do_plots:
            
            plotAxis2Ddict = config[mode]['plots']['plotAxis2D']
            plotSpec2Ddict = config[mode]['plots']['plotSpec2D']
            varsToPlot2D = [var for var in plotAxis2Ddict.values()]
            xsToPlot2D = [item[0] for item in varsToPlot2D]
            ysToPlot2D = [item[1] for item in varsToPlot2D]
            plotSpec2D = [spec for spec in plotSpec2Ddict.values()]

            file = TFile(f'{output_dir}_AUGMENTED.root', 'recreate')
            print(f'ROOT file created in {output_dir}_AUGMENTED.root')
            plot = Plotter(TrainSet, file)
            plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)

            plot.multi_df('label', particle_dict.values())
            plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)
            file.Close()

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

            betaSpec = config[mode]['data_prep']['betaSpec']
            weight_handler = SingleVariable_Flattener(TrainSet, 'beta_weight', 'beta')
            weight_handler.eval_weights(betaSpec[0], betaSpec[1], betaSpec[2])
            TrainSet = weight_handler.assign_weights()

    # Beta and momentum flat weights
    #_________________________________

    if beta_p_flat:
        with alive_bar(title='Beta and momentum flat...') as bar:

            betaPSpec = config[mode]['data_prep']['betaPSpec']

            weight_handler = DoubleVariable_Flattener(TrainSet, 'beta_pweight', 'p', 'particle')
            weight_handler.eval_weights(betaPSpec[0], betaPSpec[1], betaPSpec[2], betaPSpec[3], betaPSpec[4], betaPSpec[5])
            TrainSet = weight_handler.assign_weights()

            plotAxisBetaPdict = config[mode]['plots']['plotAxisBetaP']
            plotSpecBetaPdict = config[mode]['plots']['plotSpecBetaP']
            varsToPlot2D = [var for var in plotAxisBetaPdict.values()]
            xsToPlot2D = [item[0] for item in varsToPlot2D]
            ysToPlot2D = [item[1] for item in varsToPlot2D]
            plotSpec2D = [spec for spec in plotSpecBetaPdict.values()]     #### WIP in the config file

            file = TFile(f'{output_dir}_BETAPFLAT.root', 'recreate')
            print(f'ROOT file created in {output_dir}_BETAPFLAT.root')
            plot = Plotter(TrainSet, file)

            plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)
            plot.plot2D([xsToPlot2D[0]], [ysToPlot2D[0]], [plotSpec2D[0]], 'beta_pweight')
            file.Close()

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

    X_train, y_train, __, __ = TrainTestData

    # Upload settings from configuration file 
    #_________________________________________________

    if True:

        mode = config['mode']

        particle_dict = config[mode]['output']['particle'] 

        output_dir = config[mode]['output']['ml_dir']
        random_state = config[mode]['data_prep']['random_state']
        RegressionColumns = config[mode]['training']['RegressionColumns']
        model_choice = config[mode]['training']['model']
        ModelParams = config[mode]['training']['ModelParams']

        do_opt = config[mode]['training']['do_opt']
        HyperParamsRange = config[mode]['training']['HyperParamsRange']
        early_stop = config[mode]['training']['early_stop']
        save_model = config[mode]['training']['save_model']

        do_augm = config[mode]['data_prep']['do_augm']
        beta_flat = config[mode]['training']['beta_flat']
        beta_p_flat = config[mode]['training']['beta_p_flat']
        MCweights = config[mode]['training']['MCweights']
        do_equal = config[mode]['data_prep']['do_equal']

        if do_augm and beta_p_flat: output_dir += f'/{mode}_augm'
        elif do_augm and beta_flat: output_dir += f'/{mode}_augm_betaflat'
        elif do_augm:               output_dir += f'/{mode}_augm'
        elif MCweights:             output_dir += f'/{mode}_MCweights'
        elif beta_flat:             output_dir += f'/{mode}_betaflat'
        elif beta_p_flat:           output_dir += f'/{mode}_beta_pflat'
        elif do_equal:              output_dir += f'/{mode}_equal'
        else:                       output_dir += f'/{mode}_no_options'

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
            if early_stop:  study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='maximize', callbacks=[callback])
            else:           study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='maximize', n_trials=50)
            #else:           study = model_handler.optimize_params_optuna(TrainTestData, HyperParamsRange, delta_scorer, direction='minimize', fit_params={'sample_weight': X_train['beta_pweight']}, timeout=300)
            ## VERIFY HOW CLASSES ASSIGN WEIGHTS TO REWRITE THE LAST LINE. IN THIS WAY THE ALGORITHM WILL TRAIN KNOWING ABOUT THE WEIGHTS
        

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        #fig = optuna.visualization.plot_optimization_history(study)
        #fig.write_image(f'{output_dir}/plot_optimization_history{options}.png')

        #fig = optuna.visualization.plot_param_importances(study)
        #fig.write_image(f'{output_dir}/plot_param_importances{options}.png')

        #fig = optuna.visualization.plot_parallel_coordinate(study)
        #fig.write_image(f'{output_dir}/plot_parallel_coordinate{options}.png')

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
        if MCweights:   print('3D weights selected...')

        if do_opt:  model_reg = xgb.XGBRegressor(**HyperParams, tree_method="gpu_hist", random_state=random_state)
        else:       model_reg = xgb.XGBRegressor(tree_method="gpu_hist", random_state=random_state)
        
        with alive_bar(title='Training...') as bar:     
            if beta_flat:       model_reg.fit(X_train[RegressionColumns], y_train, sample_weight=X_train['beta_weight'])
            elif beta_p_flat:   model_reg.fit(X_train[RegressionColumns], y_train, sample_weight=X_train['beta_pweight'])
            elif MCweights:     model_reg.fit(X_train[RegressionColumns], y_train, sample_weight=X_train['3d_weight'])
            else:               model_reg.fit(X_train[RegressionColumns], y_train)
        
        TrainSet, TestSet = pd.DataFrame(TrainTestData[0]), pd.DataFrame(TrainTestData[2])

        dfs = {'train': TrainSet, 'test': TestSet}
        for name, df in dfs.items():

            file = TFile(f'{output_dir}_{name}.root', 'recreate')
            print(f'ROOT file created in {output_dir}_{name}.root')

            scorer = Scorer(model_reg, df, RegressionColumns, 'beta', file)
            scorer.Delta()

            pPltSpec = config[mode]['plots']['scoreDeltaSpec']['p']
            betaPltSpec = config[mode]['plots']['scoreDeltaSpec']['beta_pred']

            scorer.histScore()
            scorer.scatterPlotScore('p', pPltSpec[0], pPltSpec[1], pPltSpec[2], pPltSpec[3])
            scorer.scatterPlotScore('beta', betaPltSpec[0], betaPltSpec[1], betaPltSpec[2], betaPltSpec[3])

            scorer.plot.multi_df('label', particle_dict.values())
            scorer.histScore()
            scorer.scatterPlotScore('p', pPltSpec[0], pPltSpec[1], pPltSpec[2], pPltSpec[3])
            scorer.scatterPlotScore('beta', betaPltSpec[0], betaPltSpec[1], betaPltSpec[2], betaPltSpec[3])  

            file.Close()      

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
        model_out = config[mode]['output']['model_out']

        print('\nSaving regressor model...')
        with open(f'{model_out}_RegressorModel_{model_choice}.pickle', 'wb') as output_file:
            pickle.dump(model_reg, output_file)
        print(f'Model saved in {model_out}_RegressorModel_{model_choice}.pickle')

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

        mode = config['mode']

        particle_dict = config[mode]['output']['particle'] 

        final_dir = config[mode]['output']['final_dir']
        delta_dir = config[mode]['output']['delta_dir']

        RegressionColumns = config[mode]['training']['RegressionColumns']

        do_augm = config[mode]['data_prep']['do_augm']
        beta_flat = config[mode]['training']['beta_flat']
        beta_p_flat = config[mode]['training']['beta_p_flat']
        MCweights = config[mode]['training']['MCweights']
        do_equal = config[mode]['data_prep']['do_equal']

        if do_augm and beta_p_flat:  
            final_dir += f'_APPLICATION_augm' 
            delta_dir += f'_APPLICATION_augm'
        elif do_augm and beta_flat:  
            final_dir += f'_APPLICATION_augm_betaflat' 
            delta_dir += f'_APPLICATION_augm_betaflat'
        elif do_augm:               
            final_dir += f'_APPLICATION_augm'
            delta_dir += f'_APPLICATION_augm'
        elif beta_flat:
            final_dir += f'_APPLICATION_betaflat'             
            delta_dir += f'_APPLICATION_betaflat'
        elif MCweights:
            final_dir += f'_APPLICATION_MCweights'            
            delta_dir += f'_APPLICATION_MCweights'
        elif beta_p_flat:
            final_dir += f'_APPLICATION_beta_pflat'            
            delta_dir += f'_APPLICATION_beta_pflat'
        elif do_equal:              
            final_dir += f'_APPLICATION_equal'
            delta_dir += f'_APPLICATION_equal'
        else:
            final_dir += f'_APPLICATION_no_options'
            delta_dir += f'_APPLICATION_no_options'

        if do_equal:    names.remove('Deu')

    # Prediction and true beta vs p
    #________________________________________________

    with alive_bar(title='Application...') as bar:
        file = TFile(f'{delta_dir}.root', 'recreate')
        print(f'ROOT file created in {delta_dir}.root')
        scorer = Scorer(model, ApplicationDf, RegressionColumns, 'beta', file)
        ApplicationDf = scorer.Delta()                                      # pred and Delta column are added

        # Delta plots
        pPltSpec = config[mode]['plots']['scoreDeltaSpec']['p']
        betaPltSpec = config[mode]['plots']['scoreDeltaSpec']['beta_pred']

        scorer.histScore()
        scorer.scatterPlotScore('p', pPltSpec[0], pPltSpec[1], pPltSpec[2], pPltSpec[3])
        scorer.scatterPlotScore('beta', betaPltSpec[0], betaPltSpec[1], betaPltSpec[2], betaPltSpec[3])

        scorer.plot.multi_df('label', particle_dict.values())
        scorer.histScore()
        scorer.scatterPlotScore('p', pPltSpec[0], pPltSpec[1], pPltSpec[2], pPltSpec[3])
        scorer.scatterPlotScore('beta', betaPltSpec[0], betaPltSpec[1], betaPltSpec[2], betaPltSpec[3]) 

        file.Close()

    # Risolution curves
    file = TFile(f'{delta_dir}_LINE.root', 'recreate')
    print(f'ROOT file created in {delta_dir}_LINE.root')
    th2_handler = TH2Handler(ApplicationDf, file, 'p', 'Delta')

    for name in particle_dict.values():
        X_name = ApplicationDf.query(f"label == '{name}'", inplace=False)
        th2_handler.df = X_name
        
        th2_handler.build_th2(150, 0., 1.5, 300, -1.5, 1.5)
        th2_handler.TH2toLine(f'Delta_line_{name}', 'y', 1)
    file.Close()
    
    # Final plots
    file = TFile(f'{final_dir}.root', 'recreate')
    print(f'ROOT file created in {final_dir}_LINE.root')
    plot = Plotter(ApplicationDf, file)

    plotAxis2Ddict = config[mode]['plots']['finalAxis2D']
    plotSpec2Ddict = config[mode]['plots']['finalSpec2D']
    varsToPlot2D = [var for var in plotAxis2Ddict.values()]
    xsToPlot2D = [item[0] for item in varsToPlot2D]
    ysToPlot2D = [item[1] for item in varsToPlot2D]
    plotSpec2D = [spec for spec in plotSpec2Ddict.values()]
    plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)

    plot.multi_df('label', particle_dict.values())
    plot.plot2D(xsToPlot2D, ysToPlot2D, plotSpec2D)
    file.Close()

def main():

    # Configuration File
    #_________________________________
    with open('../configs/config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    mode = config['mode']

    
    # Data Preparation
    #__________________________________
    skip_data_prep = config[mode]['data_prep']['skip_data_prep']

    if not skip_data_prep:   TrainTestData, ApplicationDf = data_prep(config)
    else:
        prep_data_loc = config[mode]['data_prep']['prep_data']
        TrainTestData = []
        for loc in prep_data_loc:   TrainTestData.append(pd.read_parquet(f'{loc}'))
        appl_loc = config[mode]['data_prep']['appl_data']
        ApplicationDf = pd.read_parquet(f'{appl_loc}')

    
    # Training
    #__________________________________
    skip_training = config[mode]['training']['skip_training']
    
    if not skip_training:   Model = regression(TrainTestData, config)


    # Application
    #__________________________________

    if skip_training:       
        model_loc = config[mode]['application']['model_loc']            
        Model = pickle.load(open(f'{model_loc}', "rb"))
        Model.training_columns = config[mode]['training']['RegressionColumns']
    
    skip_appl = config[mode]['application']['skip_appl']
    if not skip_appl:   application(ApplicationDf, config, Model)     

    del TrainTestData, ApplicationDf

#############################

start_time = time()

main()

passed_time = time() - start_time
print(f'\nTime: {passed_time/60} min')