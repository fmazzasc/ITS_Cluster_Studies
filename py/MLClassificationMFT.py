'''
python scri to run basic training and application using the hipe4ml package
run: python MLClassification.py cfgFileNameML.yml [--train, --apply]
'''
import os
import sys
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot

from hipe4ml import plot_utils
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler


import optuna 

def data_prep(inputCfg, OutPutDir): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for data preparation
    '''
    print('Data preparation: ...', end='\r')
    traintest_data = inputCfg["input"]["traintest_data"] 
    appl_data = inputCfg["input"]["appl_data"]
    seed_split = inputCfg['data_prep']['seed_split']
    test_frac = inputCfg['data_prep']['test_fraction']
    presel = inputCfg['data_prep']['presel']
    LabelsArray = inputCfg['output']['out_labels']
    OutFileLabels = inputCfg['output']['outfile_label']
    
    traintest_df = uproot.open(traintest_data)['DauTree'].arrays(library='pd')
    if presel:
        traintest_df.query(f'{presel}', inplace=True)

    for i in range(7):
        traintest_df[f'ClSizeL{i}'] = np.where(traintest_df[f'ClSizeL{i}'] < 0, float("nan"), traintest_df[f'ClSizeL{i}'])
    traintest_df.eval('mean_patt_ID = (PattIDL0 + PattIDL1 + PattIDL2 + PattIDL3 + PattIDL4 + PattIDL5 + PattIDL6)/7', inplace=True)
    traintest_df.eval('isMIP = 1', inplace=True)
    traintest_df['isMIP'] = np.where(traintest_df['p'] < 1, 0, traintest_df['isMIP'])

    if appl_data:
        appl_df = uproot.open(appl_data)['ITStreeML'].arrays(library='pd')
        appl_df.query(f'{presel}', inplace=True) 
        ApplData = appl_df
    TrainSet, TestSet, yTrain, yTest = train_test_split(traintest_df, traintest_df['isMIP'], test_size=test_frac, random_state=seed_split)
    TrainTestData = [TrainSet, yTrain, TestSet, yTest]

    # plots
    LegLabels = [inputCfg['output']['out_labels']['MIP'], 
                 inputCfg['output']['out_labels']['BKG']]
    VarsToDraw = inputCfg['plots']['plotting_columns']
    df_mip = TrainSet[TrainSet['isMIP'] == 1]
    df_bkg = TrainSet[TrainSet['isMIP'] == 0]
    list_df = [df_mip, df_bkg]
    plot_utils.plot_distr(list_df, VarsToDraw, 100, LegLabels, figsize=(24, 14),
                          alpha=0.3, log=True, grid=False, density=True)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(f'{OutPutDir}/DistributionsAll_{OutFileLabels}.png')
    plt.close('all')

    for _, (df, lab) in enumerate(zip(list_df, LegLabels)):
        plt.hist2d(df['p'], df['dedx'], norm=mpl.colors.LogNorm(), range=[[0, 2], [0, 800]],  bins=(100, 100), cmap=plt.cm.RdBu)
        plt.xlabel('p [GeV]')
        plt.ylabel('dE/dx')
        plt.colorbar()
        plt.savefig(f'{OutPutDir}/dedx_{lab}.png')
        plt.close('all')
    
    CorrMatrixFig = plot_utils.plot_corr(list_df, VarsToDraw, LegLabels)
    if len(list_df) > 1:
        for Fig, Lab in zip(CorrMatrixFig, LabelsArray):
            plt.figure(Fig.number)
            plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
            Fig.savefig(f'{OutPutDir}/CorrMatrix_{OutFileLabels}.png')
            plt.close(Fig)
    else:
        plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
        CorrMatrixFig.savefig(f'{OutPutDir}/CorrMatrix_{OutFileLabels}.png')
        plt.close(CorrMatrixFig)

    del df_mip, df_bkg, list_df
    return TrainTestData #, ApplData


def train_test(inputCfg, OutPutDir, TrainTestData): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for model training and testing
    '''
    print('Training and testing: ...', end='\r')
    n_classes = len(TrainTestData[1].unique())
    if n_classes != len(inputCfg['output']['out_labels']):
        print('\033[91mERROR: Number of classes and labels must be equal!\033[0m')
        sys.exit()
    model = xgb.XGBClassifier(use_label_encoder=False)
    TrainCols = inputCfg['ml']['training_columns']
    HyperPars = inputCfg['ml']['hyper_par']
    ModelHandl = ModelHandler(model, TrainCols, HyperPars)

    # hyperparams optimization
    if inputCfg['ml']['hyper_par_opt']['do_hyp_opt']:
        print('Performing optuna optimization: ...\r')
        #OptConfig = inputCfg['ml']['hyper_par_opt']['optuna_opt_config']
        OptConfig = {'max_depth': (2, 6), 
                     'learning_rate': (0.01, 0.1),
                     'n_estimators': (200, 1200),
                     'min_child_weight': (1, 10),
                     'subsample': (0.8, 1.), 
                     'colsample_bytree': (0.8, 1.),
                     'max_bin': (200, 400),
                     'tree_method': 'gpu_hist'
                    }
        if not isinstance(OptConfig, dict):
            print('\033[91mERROR: optuna_opt_config must be defined!\033[0m')
            sys.exit()
        metric = 'roc_auc'
        timeout = inputCfg['ml']['hyper_par_opt']['timeout']
        early_stopping = inputCfg['ml']['hyper_par_opt']['early_stopping']

        OutFileHypPars = open(f'{OutPutDir}/OptunaHyperParOpt.txt', 'wt')
        sys.stdout = OutFileHypPars

        ModelHandl.optimize_params_optuna(TrainTestData, OptConfig, cross_val_scoring='roc_auc_ovo', save_study=f'{OutPutDir}/OptunaStudy.pkl', timeout=timeout, early_stopping=early_stopping)
        OutFileHypPars.close()
        sys.stdout = sys.__stdout__
        print(f'Best hyper-parameters:\n{ModelHandl.get_model_params()}')
        print(f'Optuna output saved in {OutPutDir}/HyperParOpt.txt')

        with open(f'{OutPutDir}/OptunaStudy.pkl', 'rb') as resume_study_file:
            study = pickle.load(resume_study_file)
        optuna_figs, titles = plot_utils.plot_optuna_results(study)
        for i, (fig, title) in enumerate(zip(optuna_figs, titles)):
            fig.write_image(f'{OutPutDir}/{title}.png')

        print('\033[92mOptuna optimization: Done!\033[0m')

    else:
        ModelHandl.set_model_params(HyperPars)

    # train and test the model with the updated hyper-parameters
    yPredTest = ModelHandl.train_test_model(TrainTestData, True, output_margin=inputCfg['ml']['raw_output'],
                                            average=inputCfg['ml']['roc_auc_average'],
                                            multi_class_opt='ovo')
    yPredTrain = ModelHandl.predict(TrainTestData[0], inputCfg['ml']['raw_output'])

    # save model handler in pickle
    ModelHandl.dump_model_handler(f'{OutPutDir}/ModelHandler.pickle')
    ModelHandl.dump_original_model(f'{OutPutDir}/XGBoostModel.model', True)

    #plots
    LegLabels = [inputCfg['output']['leg_labels']]

    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 7)
    MLOutputFig = plot_utils.plot_output_train_test(ModelHandl, TrainTestData, 100, inputCfg['ml']['raw_output'],
                                                    LegLabels, inputCfg['plots']['train_test_log'], density=False)
    MLOutputFig.savefig(f'{OutPutDir}/MLOutputDistr.png')

    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 9)
    ROCCurveFig = plot_utils.plot_roc(TrainTestData[3], yPredTest, None, LegLabels, inputCfg['ml']['roc_auc_average'],
                                      inputCfg['ml']['roc_auc_approach'])
    ROCCurveFig.savefig(f'{OutPutDir}/ROCCurveAll.png')
    plt.close(ROCCurveFig)
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 9)
    ROCCurveTTFig = plot_utils.plot_roc_train_test(TrainTestData[3], yPredTest, TrainTestData[1], yPredTrain, None,
                                                   LegLabels, inputCfg['ml']['roc_auc_average'],
                                                   inputCfg['ml']['roc_auc_approach'])
    ROCCurveTTFig.savefig(f'{OutPutDir}/ROCCurveTrainTest.png')
    plt.close(ROCCurveTTFig)
    #_____________________________________________
    PrecisionRecallFig = plot_utils.plot_precision_recall(TrainTestData[3], yPredTest, LegLabels)
    PrecisionRecallFig.savefig(f'{OutPutDir}/PrecisionRecallAll.png')
    plt.close(PrecisionRecallFig)
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (12, 7)
    FeaturesImportanceFig = plot_utils.plot_feature_imp(TrainTestData[2][TrainCols], TrainTestData[3], ModelHandl,
                                                        LegLabels)
    n_plot = 1
    for iFig, Fig in enumerate(FeaturesImportanceFig):
        Fig.savefig(f'{OutPutDir}/FeatureImportanceAll.png')
        plt.close(Fig)

    return ModelHandl


def appl(inputCfg, OutPutDir, ModelHandl, ApplDf):
    """
    Apply the model to the final dataset
    """
    print('Applying ML model: ...', end='\r')
    OutputLabels = [inputCfg['output']['out_labels']['BKG'],
                    inputCfg['output']['out_labels']['MIP']]
    Pred = ModelHandl.predict(ApplDf, inputCfg['ml']['raw_output'])
    df_column_to_save_list = inputCfg['appl']['column_to_save_list']
    if not isinstance(df_column_to_save_list, list):
        print('\033[91mERROR: df_column_to_save_list must be defined!\033[0m')
        sys.exit()
    ApplDf['ML_output'] = Pred
    ApplDf.to_parquet(f'{OutPutDir}/ModelApplied.parquet.gzip')
    ApplDf.write_df_to_root_files('MLTree', f'{OutPutDir}/ModelApplied.root')


def main(): #pylint: disable=too-many-statements
    # read config file
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    args = parser.parse_args()

    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)

    print(f'\n\033[94mStarting ML analysis\033[0m')
    OutPutDir = inputCfg['output']['dir']
    if os.path.isdir(OutPutDir):
        print((f'\033[93mWARNING: Output directory \'{OutPutDir}\' already exists,'
                ' overwrites possibly ongoing!\033[0m'))
    else:
        os.makedirs(OutPutDir)

    if not inputCfg['appl']['standalone']:
        # data preparation
        #_____________________________________________
        TrainTestData = data_prep(inputCfg, OutPutDir)
        print('\033[92mData preparation: completed!\033[0m')

        # training, testing
        #_____________________________________________
        ModelHandl = train_test(inputCfg, OutPutDir, TrainTestData)
        print('\033[92mTraining and testing: completed!\033[0m')

    else:
        print(f'Standalone mode se,ected: loading model handler from {inputCfg["appl"]["saved_model"]} and applying it to {inputCfg["input"]["appl_data"]}!')
        ModelHandl = pickle.load(open(inputCfg['appl']['saved_model'], "rb"))
        ModelHandl.training_columns = inputCfg['ml']['training_columns']
        hypPar = ModelHandl.get_params()
        ModelHandl.set_params(**hypPar)
        ApplDf = pd.read_parquet(inputCfg['input']['appl_data']) 
    
    # model application
    #_____________________________________________
    appl(inputCfg, OutPutDir, ModelHandl, ApplDf)
    print('\033[92mApplication: completed!\033[0m')

    # delete dataframes to release memory
    del TrainTestData, ApplDf

    print(f'\n\033[94mML analysis completed successfully!\033[0m')

main()
