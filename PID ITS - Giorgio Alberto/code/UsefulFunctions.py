import pandas as pd
import numpy as np
from math import ceil, sqrt
from itertools import combinations


from ROOT import TH2F, TCanvas, TH1F, gStyle, kBird, TFile, kCyan, kRed, gPad, TLegend
from ROOT_graph import set_obj_style





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
    - filename: the output name will be "{filename}.root"
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

    plot_spec_hist = [f'#Delta'] + plot_specifics[3:]
    hist(delta, f'{filename}_score_hist', plot_spec_hist)

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
    multiple_hist([delta_train, delta_test], '', plot_specifics[3:], f'{filename}_score_hist', hist_names=['Train', 'Test'])




# Kolmogorov - Smirnov test
#_______________________________________

def KolmogorovHist(h1, h2, canvas, pad, legend, name, **kwargs):
    """
    Does a Kolmogorov test for two given histograms, plots the histograms and returns the result

    **kwargs
    - leg_entry1
    - leg_entry2
    """

    factor = 1.
    legend.Clear()
    
    h1.Scale(factor/h1.Integral(), 'width') 
    h1.SetError( np.array([ sqrt(h1.GetEntries())/(h1.GetEntries()) for i in range( int(h1.GetEntries()) )], dtype='float') )
    h1.SetLineColor(kCyan)
    h1.SetMaximum(10)
    h1.SetMinimum(0.00001)
    if 'leg_entry1' in kwargs:  legend.AddEntry(h1, kwargs['leg_entry1'], 'l')
    else:                       legend.AddEntry(h1, 'V0', 'l')

    h2.Scale(factor/h2.Integral(), 'width')
    h2.SetError( np.array([ sqrt(h2.GetEntries())/(h2.GetEntries()) for i in range( int(h2.GetEntries()) ) ], dtype='float') )
    h2.SetLineColor(kRed)
    h2.SetMaximum(10)
    h2.SetMinimum(0.00001)
    if 'leg_entry2' in kwargs:  legend.AddEntry(h2, kwargs['leg_entry2'], 'l')
    else:                       legend.AddEntry(h2, 'TPC', 'l')

    KS_result = h1.KolmogorovTest(h2)
    h1.SetTitle(f'{name}: {round(KS_result, 3)}')       

    canvas.cd(pad)      
    h1.Draw('hist e1')
    h2.Draw('hist same e1')
    gPad.SetLogy()
    legend.Draw()
    canvas.Draw()

    return KS_result

def MultipleKS(dfs, column, hist_specifics, filename, leg_options=[]):
    """
    Saves multiple histigrams (different particle species, same variable) on the same file. You can reiterate for different variables as well.
    
    Parameters
    --------------------------------------------
    - dfs: list of dataframes whose columns will fill the histograms (it also works with pd.Series)
    - column: variable you want to draw histograms of
    - hist_specific: list with the following content -> [nbinsx, xlow, xup] 
    - leg_options: [xlow, xup, ylow, yup]

    You could also pass a list of lists as dfs. If that is the case, please input hist_names and x_label. YOu should anyway pass a nonempty list for columns. Any single elements inside of it will be fine
    - hist_names: list of names for each of the histograms
    - x_label: label of the x axis 
    """
    
    nbinsx, xlow, xup = hist_specifics
    results = []

    if leg_options == []:   leg = TLegend(0.65, 0.8, 0.75, 0.95)
    else:                   leg = TLegend(leg_options[0], leg_options[1], leg_options[2], leg_options[3])
    leg.SetFillStyle(0)
    leg.SetTextSize(0.045)
    leg.SetNColumns(1)



    hists = [TH1F(f"h{i+1}", "", nbinsx, xlow, xup) for i in range(len(dfs))]
    for histo, df in zip(hists, dfs):    
        for x in df[column]:    histo.Fill(x)



    for i, h in enumerate(combinations(hists, 2)):    

        canvas = TCanvas('canvas', 'canvas', 900, 1200)

        h0_empty, h1_empty = False, False
        if h[0].GetEntries() == 0:    
            h0_empty = True
            print('First histogram is empty')
        if h[1].GetEntries() == 0:    
            h1_empty = True
            print('Second histogram is empty')
        if h0_empty or h1_empty:    continue
        else:                       results.append(KolmogorovHist(h[0], h[1], canvas=canvas, pad=1, legend=leg, name=column, leg_entry1=f'{h[0].GetName()}', leg_entry2=f'{h[1].GetName()}'))

        canvas.SaveAs(f'{filename}_{column}_{i}.root')

        del canvas
    
    del hists, leg



