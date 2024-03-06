#
#   Data preparation classes and functions
#

import numpy as np
import polars as pl
import yaml

from abc import ABC, abstractmethod
import polars as pl

from utils.particles import particleMasses, particlePDG
from src.weightsDefinition import weightsPandSpecies

class DataPreprocessor(ABC):

    def __init__(self, inData, cfg):
        self.data = inData
        self.cfg = cfg

    @abstractmethod
    def Preprocess(self):                               pass

    @abstractmethod
    def ParticleID(self):                               pass

    @abstractmethod
    def DefineWeights(self):                            pass

    @abstractmethod
    def ApplyCuts(self):                                pass

    @abstractmethod
    def CleanData(self):                                pass

    @classmethod
    def CreatePreprocessor(cls, inData, cfg, opt):
        
        if opt == 'ITS-TPC':    return ItsTpcDataPreprocessor(inData, cfg)
        #elif opt == 'MC':       return MCDataPreprocessor(inData, cfgFile)
        else:                   raise ValueError('Unsupported option')



class ItsTpcDataPreprocessor(DataPreprocessor):

    def __init__(self, inData, cfgFile):
        super().__init__(inData, cfgFile)

    def Preprocess(self):

        # require cl size > 0 for all layers
        for layer in range(7):  self.data = self.data.filter(pl.col(f'clSizeL{layer}') > 0) 
        
        # create new columns
        self.data = self.data.with_columns(cosL=(1 / (pl.col('tgL')**2 + 1)))
        self.data = self.data.with_columns(meanClSize=((pl.col('clSizeL0') + pl.col('clSizeL1') + pl.col('clSizeL2') + pl.col('clSizeL3') + pl.col('clSizeL4') + pl.col('clSizeL5') + pl.col('clSizeL6')) / 7))
        self.data = self.data.with_columns(clSizeCosL=(pl.col('meanClSize') * pl.col('cosL')))
        self.data = self.data.with_columns(
            nSigmaAbsDeu=abs(pl.col('nSigmaDeu')),
            nSigmaAbsP=abs(pl.col('nSigmaP')),
            nSigmaAbsK=abs(pl.col('nSigmaK')),
            nSigmaAbsPi=abs(pl.col('nSigmaPi')),
            nSigmaAbsE=abs(pl.col('nSigmaE'))
            )
        self.data = self.data.with_columns(deltaP = (pl.col('pTPC') - pl.col('pITS')) / pl.col('pTPC'))


    def ParticleID(self):
        
        self.data = self.data.with_columns(
            partID=np.nan,
            mass=np.nan,
            beta=np.nan
            )
        
        for part in self.cfg['species']:    
            
            cfgTags = self.cfg['selTags'][part]
            self.data = self.data.with_columns(
                partID=pl.when((pl.col(f'nSigmaAbs{part}') < cfgTags['selfSel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part1"]}') > cfgTags['part1Sel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part2"]}') > cfgTags['part2Sel']),
                                (pl.col('p') < cfgTags['pmax'])).then(particlePDG[part]).otherwise(pl.col('partID')),
                mass=pl.when((pl.col(f'nSigmaAbs{part}') < cfgTags['selfSel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part1"]}') > cfgTags['part1Sel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part2"]}') > cfgTags['part2Sel']),
                                (pl.col('p') < cfgTags['pmax'])).then(particleMasses[part]).otherwise(pl.col('mass')),
                beta=pl.when((pl.col(f'nSigmaAbs{part}') < cfgTags['selfSel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part1"]}') > cfgTags['part1Sel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part2"]}') > cfgTags['part2Sel']),
                                (pl.col('p') < cfgTags['pmax'])).then(pl.col('p') / np.sqrt(pl.col('p')**2 + particleMasses[part]**2)).otherwise(pl.col('beta'))
                )

    def DefineWeights(self):

        if self.cfg['weights']['type'] == 'PandSpecies':    self.data = weightsPandSpecies(self.data, self.cfg)

    def ApplyCuts(self):

        self.data = self.data.filter((pl.col('nClusTPC') > self.cfg['cuts']['nClusTPCmin']) & 
                                     (pl.col('chi2ITSTPC') < self.cfg['cuts']['chi2ITSTPCmax']) &
                                     (abs(pl.col('eta')) > self.cfg['cuts']['etamin'])
                                     )

    def CleanData(self):
        '''
        Convert all negative values to NaNs
        '''

        self.data = self.data.with_columns([
            pl.when(pl.col(f'clSizeL{layer}') < 0).then(np.nan).otherwise(pl.col(f'clSizeL{layer}')).alias(f'clSizeL{layer}')
            for layer in range(7)
            ])

    def DropUnidentified(self):
        '''
        Drop all particles that were not identified
        '''

        self.data = self.data.filter(pl.col('partID') != np.nan)





'''

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod

import uproot

sys.path.append('..')
from utils.data_reader import readFile

########################################
# 
#   Data preparation classes
#_______________________________________

class PrepTool(ABC):
    """
    Abstract class for data preparation

    Attributes:
        data (pd.DataFrame): dataset that will be used for training and test
        application (pd.DataFrame): dataset for application (if given)
        columns (list[str]): columns to add to existing dataframes (different for each child)
    """
    def __init__(self, data, application=None) -> None:

        self.data = data.copy()
        if application is not None: self.application = application.copy()
        else:                       self.data, self.application = train_test_split(self.data, test_size=0.5, train_size=0.5, random_state=0)
        
        self.columns = []

    @abstractmethod
    def preprocess(self, particle_dict, selection_tag, selection_tag_appl=None):
        """
        Preprocess the input data

        Parameters
        ----------
            particle_dict (dict): Dictionary (index, particle) 
            selection_tag (str): String used to filter the dataset
            selection_tag_appl (str): String to filter the application dataset
        """

        pass

    def preprocess_2(self, seven_hits=False):
        """
        Second step of preprocessing (to be done after data visualization)

        Parameters
        ----------
            seven_hits (bool): Whether to discard particles that are not detected by all ITS layers
        """

        if seven_hits:  self.require_seven_hits()
        else:           self.drop_negative_values()

    def filter_and_split(self, particle_dict, mass_dict, tag_dict=None, test_size=0.2, random_state=0):
        """
        Add beta information to the dataset and split it into train, test and application.
        trainSet, testSet, yTrain, yTest are added as class members.

        Parameters
        ----------
            particle_dict (dict): dictionary (index, particle_name)
            mass_dict (dict): dictionary (particle_name, mass)
            tag_dict (dict): dictionary (particle_name, selection_tag)
            test_size (float): proportion of data to use for the test set
            random_state (int): random seed to use for reproducibility

        Returns
        -------
            TrainSet (pd.DataFrame): training data
            TestSet (pd.DataFrame): test data
            yTrain (pd.DataFrame): training target
            yTest (pd.DataFrame): test target
            application (pd.DataFrame): application set
        """

        # filter data and application
        dataList = []
        for name in particle_dict.values():     
            
            dataFilt = pd.DataFrame()

            if tag_dict is not None:    dataFilt = filtering(self.data, name, mass=mass_dict[name], tag=tag_dict[name], label=True)
            else:                       dataFilt = filtering(self.data, name, mass=mass_dict[name], label=True)

            dataList.append(dataFilt)
            dataFiltToDrop = dataFilt.drop(columns=['label', 'beta'])
            self.data = pd.concat([self.data, dataFiltToDrop, dataFiltToDrop]).drop_duplicates(keep=False)
            print(f'selected {name}, dataset size: {len(self.data.index)}')

        self.data = pd.concat(dataList)

        # create the 'particle' column using a reversed dictionary
        inv_particle_dict = {v: k for k, v in particle_dict.items()}
        if not 'particle' in self.data.columns: self.data['particle'] = self.data['label'].map(inv_particle_dict)
        y = self.data['beta']
        self.trainSet, self.testSet, self.yTrain, self.yTest = train_test_split(self.data, y, test_size=test_size, random_state=random_state)

        return self.trainSet, self.testSet, self.yTrain, self.yTest, self.application

    def drop_negative_values(self):
        """
        Replace negative values with NaNs (they can be handled by xgboost models)
        """
        
        for i in range(7):
            self.data[f'ClSizeL{i}'] = np.where(self.data[f'ClSizeL{i}'] < 0, np.nan, self.data[f'ClSizeL{i}']) 
            self.application[f'ClSizeL{i}'] = np.where(self.application[f'ClSizeL{i}'] < 0, np.nan, self.application[f'ClSizeL{i}'])

    def require_seven_hits(self):
        """
        Drop all particles that were not detected by all seven layers
        """

        for i in range(7):  self.data.drop(self.data[self.data[f'ClSizeL{i}'] < 0].index, inplace=True)

    def add_columns(self, columns):
        """
        Adds columns to data and application dataframes

        Parameters
        ----------
            columns (list[str]): name of the columns to create
        """

        if 'meanPattID' in columns:     self.data.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
        if 'meanSnPhi' in columns:      self.data.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
        if 'L6_L0' in columns:          self.data.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)
        if 'delta_p' in columns:        self.data.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
        if 'p' in columns:              self.data.eval('p = pMC', inplace=True)
        if 'label' in columns:          self.data.eval('label = particle', inplace=True)

        if 'meanPattID' in columns:     self.application.eval('meanPattID = (PattIDL0+PattIDL1+PattIDL2+PattIDL3+PattIDL4+PattIDL5+PattIDL6)/7', inplace=True)
        if 'meanSnPhi' in columns:      self.application.eval('meanSnPhi = (SnPhiL0+SnPhiL1+SnPhiL2+SnPhiL3+SnPhiL4+SnPhiL5+SnPhiL6)/7', inplace=True)
        if 'L6_L0' in columns:          self.application.eval('L6_L0 = ClSizeL6/ClSizeL0', inplace=True)
        if 'delta_p' in columns:        self.application.eval('delta_p = (p - pTPC)/pTPC', inplace=True)
        if 'label' in columns:          self.application.eval('label = particle', inplace=True)

    def return_dfs(self):
        """
        Return dataframes

        Returns
        -------
            trainSet, testSet, yTrain, yTest, application (pd.DataFrame)
        """
        return self.trainSet, self.testSet, self.yTrain, self.yTest, self.application

class TPC_prep(PrepTool):
    """
    Concrete class for data preparation using TPC particle identification

    Attributes:
        data (pd.DataFrame): dataset that will be used for training and test
        application (pd.DataFrame): dataset for application (if given)
        columns (list[str]): columns to add to existing dataframes (different for each child)
    """
    def __init__(self, data, application=None) -> None:
        
        super().__init__(data, application)
        self.columns = ['meanPattID', 'meanSnPhi', 'L6_L0', 'delta_p']

    def preprocess(self, particle_dict, selection_tag):
        """
        Preprocess the input data

        Parameters
        ----------
            particle_dict (dict): dictionary (index, particle) 
            selection_tag (str): String used to filter the dataset
        """

        self.add_columns(self.columns)
        
        self.data.query(selection_tag, inplace=True)  
        for part in particle_dict.values():  self.data[f'nSigma{part}Abs'] = abs(self.data[f'nSigma{part}'])    

        for part in particle_dict.values():  self.application[f'nSigma{part}Abs'] = abs(self.application[f'nSigma{part}'])

    def add_columns(self, columns):
        super().add_columns(columns)

class Hybrid_prep(PrepTool):
    """
    Concrete class for data preparation using TPC and ITS particle identification

    Attributes:
        data (pd.DataFrame): dataset that will be used for training and test
        application (pd.DataFrame): dataset for application (if given)
        columns (list[str]): columns to add to existing dataframes (different for each child)
    """
    def __init__(self, data, application=None) -> None:
        
        super().__init__(data, application)
        self.columns = ['meanPattID', 'meanSnPhi', 'L6_L0', 'delta_p', 'label']

    def preprocess(self, particle_dict, selection_tag, selection_tag_appl=None):
        """
        Preprocess the input data

        Parameters
        ----------
            particle_dict (dict): dictionary (index, particle) 
            selection_tag (str): String used to filter the dataset
            selection_tag_appl (str, not used): String to filter the application dataset
        """

        self.add_columns(self.columns)
        
        self.data.query(selection_tag, inplace=True)  
        for part in particle_dict.values():  self.data[f'nSigma{part}Abs'] = abs(self.data[f'nSigma{part}'])
        for number, name in particle_dict.items():  self.data['label'].mask(self.data['particle'] == number, name, inplace=True)
            
        for part in particle_dict.values():  self.application[f'nSigma{part}Abs'] = abs(self.application[f'nSigma{part}']) 
        for number, name in particle_dict.items():  self.application['label'].mask(self.application['particle'] == number, name, inplace=True)
   
    def add_columns(self, columns):
        super().add_columns(columns)

class MC_prep(PrepTool):
    """
    Concrete class for data preparation using MC simulation samples

    Attributes:
        data (pd.DataFrame): dataset that will be used for training and test
        application (pd.DataFrame): dataset for application (if given)
        columns (list[str]): columns to add to existing dataframes (different for each child)
    """

    def __init__(self, data, application=None) -> None:
        
        super().__init__(data, application)
        self.columns = ['meanPattID', 'meanSnPhi', 'L6_L0', 'p', 'label']

        # translates PDG labels to those used in this script
        self.MC_dict = {2212: 1, 321: 2, 211: 3, 11: 4}

    def preprocess(self, particle_dict, selection_tag, selection_tag_appl=None):
        """
        Preprocess the input data

        Parameters
        ----------
            particle_dict (dict): dictionary (index, particle) 
            selection_tag (str): String used to filter the dataset
            selection_tag_appl (str, not used): String to filter the application dataset
        """

        # translate from PDG indeces
        temp = []
        for i, (key, value) in enumerate(self.MC_dict.items()):   
            temp.append(self.data.query(f'abs(pdgMC) == {key}', inplace=False).reset_index(drop=True))
            temp[i]['particle'] = value
        self.data = pd.concat(temp)
        
        self.add_columns(self.columns)

        self.data.query(selection_tag, inplace=True)  
        for part in particle_dict.values():  self.data[f'nSigma{part}Abs'] = abs(self.data[f'nSigma{part}'])
        for number, name in particle_dict.items():  self.data['label'].mask(self.data['particle'] == number, name, inplace=True)
        
        for number, name in particle_dict.items():  self.application['label'].mask(self.application['particle'] == number, name, inplace=True)
   
    def add_columns(self, columns):
        super().add_columns(columns)

class MCtpc_prep(PrepTool):
    """
    Concrete class for data preparation using MC simulation samples

    Attributes:
        data (pd.DataFrame): dataset that will be used for training and test
        application (pd.DataFrame): dataset for application (if given)
        columns (list[str]): columns to add to existing dataframes (different for each child)
    """

    def __init__(self, data, application=None) -> None:
        
        super().__init__(data, application)
        self.columns = ['meanPattID', 'meanSnPhi', 'L6_L0', 'p', 'label']

        # translates PDG labels to those used in this script
        self.MC_dict = {2212: 1, 321: 2, 211: 3, 11: 4}
    
    def preprocess(self, particle_dict, selection_tag, selection_tag_appl=None):
        """
        Preprocess the input data

        Parameters
        ----------
            particle_dict (dict): dictionary (index, particle) 
            selection_tag (str): String used to filter the dataset
            selection_tag_appl (str): String used to filter the application dataset
        """

        # translate from PDG indeces
        temp = []
        for i, (key, value) in enumerate(self.MC_dict.items()):   
            temp.append(self.data.query(f'abs(pdgMC) == {key}', inplace=False).reset_index(drop=True))
            temp[i]['particle'] = value
        self.data = pd.concat(temp)

        self.add_columns(self.columns)

        self.data.query(selection_tag, inplace=True)  
        for part in particle_dict.values():  self.data[f'nSigma{part}Abs'] = abs(self.data[f'nSigma{part}'])
        for number, name in particle_dict.items():  self.data['label'].mask(self.data['particle'] == number, name, inplace=True)
        
        for number, name in particle_dict.items():  self.application['label'].mask(self.application['particle'] == number, name, inplace=True)
   
    def add_columns(self, columns):
        super().add_columns(columns)

class PrepConstructor:

    @staticmethod
    def createPrepTool(choice, fimpPath, applPath=None):
        """
        Create PrepTool (object that manages data preprocessing)

        Parameters
        ----------
            choice (str): specifies which concrete object should be created.
                Valid choices are 'TPC', 'hybrid', 'MC', 'MCtpc'
            fimpPath (str): file to import train and test data from
            applPath (str, optional): file to import application data from
        """
        if choice == 'TPC':         return PrepConstructor.createTPC(fimpPath, applPath)
        elif choice == 'hybrid':    return PrepConstructor.createHybrid(fimpPath, applPath)
        elif choice == 'MC':        return PrepConstructor.createMC(fimpPath, applPath)
        elif choice == 'MCtpc':     return PrepConstructor.createMCtpc(fimpPath, applPath)
        else:                       raise ValueError('Invalid PrepTool choice')

    @staticmethod
    def createTPC(fimpPath, applPath=None):
        """
        Create TPC_prep object

        Parameters
        ----------
            fimpPath (str): .parquet file from which train data should be imported
            applPath (str): .parquet file from which application data should be imported
        """

        print(f'Input data: {fimpPath}')
        data = readFile(fimpPath, 'ITStreeML')
        if applPath is not None:    application = readFile(fimpPath, 'ITStreeML')
        else:                       application = None

        print('Preprocessing TPC data...')
        return TPC_prep(data, application)

    @staticmethod
    def createHybrid(fimpPath, applPath=None):
        """
        Create Hybrid_prep object

        Parameters
        ----------
            fimpPath (str): .parquet file from which train data should be imported
            applPath (str): .parquet file from which application data should be imported
        """

        print(f'Input data: {fimpPath}')
        data = readFile(fimpPath, 'ITStreeML')
        if applPath is not None:    application = readFile(fimpPath, 'ITStreeML')
        else:                       application = None

        print('Preprocessing hybrid data...')
        return Hybrid_prep(data, application)

    @staticmethod
    def createMC(fimpPath, applPath=None):
        """
        Create MC_prep object

        Parameters
        ----------
            fimpPath (str): .root file from which train data should be imported
            applPath (str): .parquet file from which application data should be imported
        """

        print(f'Input data: {fimpPath}')
        data = readFile(fimpPath, 'ITStreeML')
        if applPath is not None:    application = readFile(fimpPath, 'ITStreeML')
        else:                       application = None

        print('Preprocessing MC sample...')
        return MC_prep(data, application)

    @staticmethod    
    def createMCtpc(fimpPath, applPath=None):
        """
        Create MCtpc_prep object

        Parameters
        ----------
            fimpPath (str): .root file from which train data should be imported
            applPath (str): .parquet file from which application data should be imported
        """

        print(f'Input data: {fimpPath}')
        data = readFile(fimpPath, 'ITStreeML')
        if applPath is not None:    application = readFile(fimpPath, 'ITStreeML')
        else:                       application = None

        print('Preprocessing MC sample and TPC data...')
        return MCtpc_prep(data, application)

########################################
#
# Filtering
#_______________________________________

def filtering(full_df, particle, tag=None, mass=None, label=True):
    """
    Filter the full dataframe to include only data for a specified particle.

    Parameters:
        full_df (pd.DataFrame): The full dataframe.
        particle (str): The name of the particle to filter.
        tag (str, optional): A tag used by pd.DataFrame.query for the selection.
        mass (float, optional): The mass of the particle (in GeV/c^2).
        label (bool, optional): If True, add a 'label' column to the filtered dataframe with the particle name.

    Returns:
        The filtered dataframe.
    """
    
    if tag is not None:     df = full_df.query(tag, inplace=False).reset_index(drop=True)
    else:                   df = full_df.query(f"label == '{particle}'").reset_index(drop=True)
 
    if label:               df['label'] = particle
    if mass is not None:    df.eval(f'beta = p / sqrt({mass}**2 + p**2)', inplace=True)

    return df

def process_application(applData, selection_tag, particle_dict, mass_dict, tag_dict=None):
    """
    Process application data (particle tagging, filtering, ...)
    Apply on this dataset all filters applied on the training dataset

    Parameters
    ----------
        applData (pd.DataFrame): dataset which filters should be applied on
        selection_tag (str): String used to filter the dataset
        particle_dict (dict): dictionary (index, particle) 
        mass_dict (dict): dictionary (particle_name, mass)
        tag_dict (dict): dictionary (particle_name, selection_tag)

    Returns
    -------
        applData (pd.DataFrame): filtered application dataset
    """

    applData.query(selection_tag, inplace=True)

    # filter data and application
    applList = []
    for name in particle_dict.values():     
        
        applFilt = pd.DataFrame()
        
        if tag_dict is not None:    applFilt = filtering(applData, name, mass=mass_dict[name], tag=tag_dict[name], label=True)
        else:                       applFilt = filtering(applData, name, mass=mass_dict[name], label=True)
        
        applList.append(applFilt)
        applFiltToDrop = applFilt.drop(columns=['label', 'beta'])
        applData = pd.concat([applData, applFiltToDrop, applFiltToDrop]).drop_duplicates(keep=False)
    
    applData = pd.concat(applList)
    
    return applData


########################################
#
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

def augmentation_fine(df, mother, daughter, mass_mother, mass_daughter, pmin, pmax):
    """
    This function performs data augmentation, generating new data for the daughter species from the pre-existing data of the mother species.

    Parameters
    ----------------------------------
    - df: full dataframe of already identified particles (with a column 'label' with theie names)
    - mother: label of the mother species
    - daughter: label of the daughter species
    - mass_mother, mass_daughter: mass of the mother and the daughter
    - pmin, pmax: momentum range to perform the data augmentation in
    """

    betamin = pmin / sqrt(mass_mother**2 + pmin**2) 
    betamax = pmax / sqrt(mass_mother**2 + pmax**2) 
    mother_to_augm = df.query(f'label == "{mother}" and {betamin} <= beta < {betamax}')

    # This check should be included when working without weights
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

    '''