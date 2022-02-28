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
import matplotlib.pyplot as plt

from hipe4ml import plot_utils
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler

def data_prep(inputCfg, OutPutDir, Df): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for data preparation
    '''

    seed_split = inputCfg['data_prep']['seed_split']
    test_frac = inputCfg['data_prep']['test_fraction']

    LabelsArray = np.array(Df['isProton'])

    TrainSet, TestSet, yTrain, yTest = train_test_split(Df.get_data_frame(), LabelsArray, test_size=test_frac, random_state=seed_split)
    TrainTestData = [TrainSet, yTrain, TestSet, yTest]
    ApplDf = TestSet

    if (inputCfg["data_prep"]["presel"]):
        Df.apply_preselections(f'{inputCfg["data_prep"]["presel"]}', inplace=True)


    # plots
    VarsToDraw = inputCfg['plots']['plotting_columns']
    LegLabels = [inputCfg['output']['leg_labels']['Bkg'],
                 inputCfg['output']['leg_labels']['Proton']]
    OutputLabels = [inputCfg['output']['out_labels']]

    #_____________________________________________
    list_df = [Df.apply_preselections("isProton==0", inplace=False), Df.apply_preselections("isProton==1", inplace=False)]
    plot_utils.plot_distr(list_df, VarsToDraw, 100, LegLabels, figsize=(12, 7),
                          alpha=0.3, log=True, grid=False, density=True)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(f'{OutPutDir}/DistributionsAll.pdf')
    plt.close('all')
    #_____________________________________________
    CorrMatrixFig = plot_utils.plot_corr(list_df, VarsToDraw, LegLabels)
    for Fig, Lab in zip(CorrMatrixFig, OutputLabels):
        plt.figure(Fig.number)
        plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
        Fig.savefig(f'{OutPutDir}/CorrMatrix.pdf')

    return TrainTestData, ApplDf


def train_test(inputCfg, OutPutDir, TrainTestData): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for model training and testing
    '''
    n_classes = 2
    modelClf = xgb.XGBClassifier(use_label_encoder=False)
    TrainCols = inputCfg['ml']['training_columns']
    if ('isProton' in TrainCols):
        print('\033[91mERROR: issProton in training columns!\033[0m')
    HyperPars = inputCfg['ml']['hyper_par']
    if not isinstance(TrainCols, list):
        print('\033[91mERROR: training columns must be defined!\033[0m')
        sys.exit()
    if not isinstance(HyperPars, dict):
        print('\033[91mERROR: hyper-parameters must be defined or be an emy dict!\033[0m')
        sys.exit()
    ModelHandl = ModelHandler(modelClf, TrainCols, HyperPars)

    # hyperparams optimization
    if inputCfg['ml']['hyper_par_opt']['do_hyp_opt']:
        print('Perform bayesian optimization')
        BayesOptConfig = inputCfg['ml']['hyper_par_opt']['bayes_opt_config']
        if not isinstance(BayesOptConfig, dict):
            print('\033[91mERROR: bayes_opt_config must be defined!\033[0m')
            sys.exit()

        if n_classes > 2:
            average_method = inputCfg['ml']['roc_auc_average']
            roc_method = inputCfg['ml']['roc_auc_approach']
            if not (average_method in ['macro', 'weighted'] and roc_method in ['ovo', 'ovr']):
                print('\033[91mERROR: selected ROC configuration is not valid!\033[0m')
                sys.exit()

            if average_method == 'weighted':
                metric = f'roc_auc_{roc_method}_{average_method}'
            else:
                metric = f'roc_auc_{roc_method}'
        else:
            metric = 'roc_auc'

        print('Performing hyper-parameters optimisation: ...', end='\r')
        OutFileHypPars = open(f'{OutPutDir}/HyperParO.txt', 'wt')
        sys.stdout = OutFileHypPars
        ModelHandl.optimize_params_bayes(TrainTestData, BayesOptConfig, metric,
                                         nfold=inputCfg['ml']['hyper_par_opt']['nfolds'],
                                         init_points=inputCfg['ml']['hyper_par_opt']['initpoints'],
                                         n_iter=inputCfg['ml']['hyper_par_opt']['niter'],
                                         njobs=inputCfg['ml']['hyper_par_opt']['njobs'])
        OutFileHypPars.close()
        sys.stdout = sys.__stdout__
        print('Performing hyper-parameters optimisation: Done!')
        print(f'Output saved in {OutPutDir}/HyperParOpt.txt')
        print(f'Best hyper-parameters:\n{ModelHandl.get_model_params()}')
    else:
        ModelHandl.set_model_params(HyperPars)

    # train and test the model with the updated hyper-parameters
    yPredTest = ModelHandl.train_test_model(TrainTestData, True, output_margin=inputCfg['ml']['raw_output'],
                                            average=inputCfg['ml']['roc_auc_average'],
                                            multi_class_opt=inputCfg['ml']['roc_auc_approach'])
    yPredTrain = ModelHandl.predict(TrainTestData[0], inputCfg['ml']['raw_output'])

    # save model handler in pickle
    ModelHandl.dump_model_handler(f'{OutPutDir}/ModelHandler.pickle')
    ModelHandl.dump_original_model(f'{OutPutDir}/XGBoostModel.model', True)

    #plots
    LegLabels = [inputCfg['output']['leg_labels']['Bkg'],
                 inputCfg['output']['leg_labels']['Proton']]

    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 7)
    MLOutputFig = plot_utils.plot_output_train_test(ModelHandl, TrainTestData, 80, inputCfg['ml']['raw_output'],
                                                    LegLabels, inputCfg['plots']['train_test_log'], density=False)
    if n_classes > 2:
        for Fig, Lab in zip(MLOutputFig, OutputLabels):
            Fig.savefig(f'{OutPutDir}/MLOutputDistr{Lab}.pdf')
    else:
        MLOutputFig.savefig(f'{OutPutDir}/MLOutputDistr.pdf')
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 9)
    ROCCurveFig = plot_utils.plot_roc(TrainTestData[3], yPredTest, None, LegLabels, inputCfg['ml']['roc_auc_average'],
                                      inputCfg['ml']['roc_auc_approach'])
    ROCCurveFig.savefig(f'{OutPutDir}/ROCCurveAll.pdf')
    pickle.dump(ROCCurveFig, open(f'{OutPutDir}/ROCCurveAll.pkl', 'wb'))
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 9)
    ROCCurveTTFig = plot_utils.plot_roc_train_test(TrainTestData[3], yPredTest, TrainTestData[1], yPredTrain, None,
                                                   LegLabels, inputCfg['ml']['roc_auc_average'],
                                                   inputCfg['ml']['roc_auc_approach'])
    ROCCurveTTFig.savefig(f'{OutPutDir}/ROCCurveTrainTest.pdf')
    #_____________________________________________
    PrecisionRecallFig = plot_utils.plot_precision_recall(TrainTestData[3], yPredTest, LegLabels)
    PrecisionRecallFig.savefig(f'{OutPutDir}/PrecisionRecallAll.pdf')
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (12, 7)
    FeaturesImportanceFig = plot_utils.plot_feature_imp(TrainTestData[2][TrainCols], TrainTestData[3], ModelHandl,
                                                        LegLabels)
    n_plot = n_classes if n_classes > 2 else 1
    for iFig, Fig in enumerate(FeaturesImportanceFig):
        if iFig < n_plot:
            label = OutputLabels[iFig] if n_classes > 2 else ''
            Fig.savefig(f'{OutPutDir}/FeatureImportance{label}.pdf')
        else:
            Fig.savefig(f'{OutPutDir}/FeatureImportanceAll.pdf')

    return ModelHandl


def appl(inputCfg, OutPutDir, ModelHandl, ApplDf):
    OutputLabels = [inputCfg['output']['out_labels']['Bkg'],
                    inputCfg['output']['out_labels']['Proton']]
    print('Applying ML model: ...', end='\r')
    Pred = ModelHandl.predict(ApplDf, inputCfg['ml']['raw_output'])
    df_column_to_save_list = inputCfg['appl']['column_to_save_list']
    if not isinstance(df_column_to_save_list, list):
        print('\033[91mERROR: df_column_to_save_list must be defined!\033[0m')
        sys.exit()
    ApplDf['ML_output'] = Pred
    ApplDf.to_parquet(f'{OutPutDir}/ModelApplied.parquet.gzip')
    #DataDfSel.write_df_to_root_files('MLTree', f'{OutPutDir}/ModelApplied.root')
    print('ML model application: Done!')


def main(): #pylint: disable=too-many-statements
    # read config file
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)

    print('Loading and preparing data files: ...', end='\r')
    DataHandler = TreeHandler(inputCfg['input']['data'])

    print(f'\n\033[94mStarting ML analysis[0m')
    OutPutDir = inputCfg['output']['dir']
    if os.path.isdir(OutPutDir):
        print((f'\033[93mWARNING: Output directory \'{OutPutDir}\' already exists,'
                ' overwrites possibly ongoing!\033[0m'))
    else:
        os.makedirs(OutPutDir)
    # data preparation
    #_____________________________________________
    TrainTestData, ApplDf = data_prep(inputCfg, OutPutDir, DataHandler)

    # training, testing
    #_____________________________________________
    ModelHandl = train_test(inputCfg, OutPutDir, TrainTestData)

    # model application
    #_____________________________________________
    appl(inputCfg, OutPutDir, ModelHandl, ApplDf)

    # delete dataframes to release memory
    del TrainTestData, ApplDf

main()
