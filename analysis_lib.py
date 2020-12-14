import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 80 # high quality images
import seaborn as sns
pd.set_option('display.max_columns', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import glob
import logging as log
import os
import re

import bat_class
import dataset
import spike_gen
import spike_plots
import SVM_utils
import behavior_parse
import shuffling
import importance
import feature_engineering
import parse_real_neural_data
import uuid

from models import SVM_model, XGBoost_model
from constants import *

FORMAT = '%(asctime)s %(message)s'
log.basicConfig(format=FORMAT)
EXPERIMENT_READY = False

CONF = {
    "GEN_DATA" : False,
    "REAL_BEHAVIORAL_DATA" : True,
    "CACHED_BEHAVIORAL_DATA" : True,
    "SHOW_PLOTS" : False,
    "PER_BAT_ANALYSIS" : False,
    "FE" : True,
    "BINNING" : False,
    "NOISE_CANCELLATION" : True,
    "SQRT_WEIGHT" : False,
    "WITH_PAIRS" : True,

    "CV" : 10,
    "N_SHUFFLES" : 50,
    "FI_NUM" : 5,
    "POPUP_RESULTS" : True,
}

VERBOSE = log.INFO #log.INFO
WARNING = log.WARNING
log.getLogger().setLevel(VERBOSE) #log.WARNING)
EXPERIMENT_READY = True

BAT_NAME_TO_ID = {
    "A" : 0,
    "O" : 1,
    "2" : 2,
    "P" : 3,
    "X" : 4,
}

# print(BAT_NAME_TO_ID)


# TODO: remove this function from here, it is too manual and has nothing to do with analysis library.
def spike_loading(day):
    """ Manual function, loads specific neural data per day.
        According to Saikat's list of suspected neurons.
    """
    neurons = []
    days_cells = {
        "191222" : ["20", "22", "29"],
        "191221" : ["31", "50", "52"],
        "191220" : ["62", "73", "76", "80"],
        "191223" : ["83", "93"],
        "191224" : ["106", "111", "116", "123"],
    }
    # run_on = days_cells[day] -> SAIKAT NOTE
    files = "\n".join(os.listdir(r"data\neural_data\raw"))
    run_on = re.findall(f"(\d*)_b2305_d{day}.mat", files)
    titles = []
    for neuron_id in run_on:
        # if neuron_id not in ["72", "73", "74"]: continue
        titles.append(neuron_id)
        res = parse_real_neural_data.parse_neural_data(neuron_id=neuron_id, day='d' + day, recorded_bat = 'b2305')
        display(res.mean())
        res[res < 2] = 0 # make spikes a binary vector
        res[res > 1] = 1
        neurons.append(res)
    
    return neurons, titles

def per_bat_analysis(df, n):
    """ Takes behavioral data, a neural data.  
        Splits the beheivoral data to each of the bats.  
        Each split always includes the recoreded bats features.  
        Then it runs analysis separately per bat.  
        All the bats whose analysis passes certain confidence threshold, are kept aside.  
        Finally, it runs an analysis over the bats that were kept aside.  

        Comment: this approach is an old approach, we currently don't use it since it didn't help with analysis.  
    """
    passed_threshold = []
    total_features = []
    THRESHOLD = 0

    for b in dataset.get_bats_names():
        bat_features = SVM_utils.get_bat_features(df, b).to_list()
        if str(b) != "0":
            bat_features += SVM_utils.get_bat_features(df, "0").to_list()

        if cell_analysis(df[bat_features], n, "") > THRESHOLD:
            passed_threshold.append(b)

    for b in passed_threshold:
        total_features += SVM_utils.get_bat_features(df, b).to_list()
        
    if len(passed_threshold) < 1:
        log.warning("no bats passed threshold!")
    return df[total_features]

def print_analysis_header(nid, spikes, cell_title):
    """ Prints a nice logo before analysis.  """
    print("*" * 80)
    print(f"Neuron: {nid+1}/{len(spikes)} {cell_title.title()}")
    print(f"Shuffles: {N_SHUFFLES}, CV: {CV}")
    print("*" * 80)

def build_shuffling_result_dataset(shuffled_vals_df, imp_table, features_names):
    """ Takes the shuffles importance table.
        Takes the cross-validation importance table.
        Takes features names sorted by importance.
        Puts their values in a dataframe to be plotted in swarmplot.
        Returns a dataframe to be plotted.
    """
    dfs = []
    for feature_name in features_names:
        dfs.append(pd.DataFrame.from_dict({"importance" : shuffled_vals_df[feature_name], "name" : feature_name, "type" : "shuffle"}))
        dfs.append(pd.DataFrame.from_dict({"importance" : imp_table[feature_name], "name" : feature_name, "type" : "real"}))
    df = pd.concat(dfs)
    
    return df

def extract_features_names_from_importance_table(imp_table):
    """ Extracts features names in a sorted order (starting from the one with the highest importance).  """
    return imp_table.mean().sort_values(ascending=False).index.to_list()

def handle_shuffle_vals(shuffled_vals, imp_table, agg_imp_table, train_cm, test_cm):
    """ Takes shuffled_vals which are stored in a transposed manner -> bunch of lists, each stores information per shuffle.  
        When transposed, each list stores categorized-information for all shuffles.  
        The categories are: [svm_model, train_cm, test_cm, imp_table, agg_imp_table].  

        This function also gets: imp_table, agg_imp_table, train_cm, test_cm.
        This functions builds dataframes in the format required by swarmplot.
    """
    t_shuffled_vals = list(zip(*shuffled_vals)) # [svm_model, train_cm, test_cm, imp_table, agg_imp_table]
    shuffles_svm_models, shuffles_train_cms, shuffles_test_cms, shuffles_imp_tables, shuffles_agg_imp_tables = t_shuffled_vals

    f_imp_names = extract_features_names_from_importance_table(imp_table)
    agg_imp_names = extract_features_names_from_importance_table(agg_imp_table)
    n_features = len(f_imp_names)

    imp_table_for_all_shuffles = pd.concat(shuffles_imp_tables)
    agg_imp_table_for_all_shuffles = pd.concat(shuffles_agg_imp_tables)

    sh_df = build_shuffling_result_dataset(imp_table_for_all_shuffles, imp_table, f_imp_names)
    sh_df2 = build_shuffling_result_dataset(agg_imp_table_for_all_shuffles, agg_imp_table, agg_imp_names)

    # we run sparisty measure per shuffle, and per cross-validation.
    shuffled_sparisty = list(map(lambda x: importance.sparsity_measure(x.mean(0), total_features=n_features), shuffles_imp_tables))
    cv_sparsity = list(map(lambda x: importance.sparsity_measure(x[1], total_features=n_features), imp_table.iterrows()))

    shuffled_sparisty_agg = list(map(lambda x: importance.sparsity_measure(x.mean(0), total_features=n_features), shuffles_agg_imp_tables))
    cv_sparsity_agg = list(map(lambda x: importance.sparsity_measure(x[1], total_features=n_features), agg_imp_table.iterrows()))

    shuffled_b_accuracy = list(map(SVM_utils.cm_to_ba, shuffles_train_cms))
    cv_train_b_accuracy = list(map(SVM_utils.cm_to_ba, train_cm))
    cv_test_b_accuracy = list(map(SVM_utils.cm_to_ba, test_cm))

    dfs = []
    dfs.append(pd.DataFrame.from_dict({"value" : shuffled_sparisty, "name" : "sparsity", "type" : "shuffle"}))
    dfs.append(pd.DataFrame.from_dict({"value" : cv_sparsity, "name" : "sparsity", "type" : "real"}))
    dfs.append(pd.DataFrame.from_dict({"value" : shuffled_sparisty_agg, "name" : "agg_sparsity", "type" : "shuffle"}))
    dfs.append(pd.DataFrame.from_dict({"value" : cv_sparsity_agg, "name" : "agg_sparsity", "type" : "real"}))

    dfs.append(pd.DataFrame.from_dict({"value" : shuffled_b_accuracy, "name" : "b_accuracy", "type" : "shuffle"}))
    dfs.append(pd.DataFrame.from_dict({"value" : cv_train_b_accuracy, "name" : "b_accuracy", "type" : "real"}))
    dfs.append(pd.DataFrame.from_dict({"value" : cv_test_b_accuracy, "name" : "b_accuracy", "type" : "real_test"}))

    shuffles_df = pd.concat(dfs)
    return f_imp_names, agg_imp_names, t_shuffled_vals, shuffles_df, sh_df, sh_df2, shuffled_b_accuracy

def plot_basic_plots_per_bat(dataframe, neuron, model_spikes, axes, bat_name):
    """ Plots the basic plots of a given bat with spikes:
        - trajectory plot
        - rate_map plot

        - ego_trajectory plot
        --  or head-direction plot for the recorded bat
        - ego rate map plot
    """
    spike_plots.trajectory_spike_plot(dataframe, neuron, model_spikes, ax=axes[0], bat_name = bat_name)
    spike_plots.rate_map_plot(dataframe, neuron, ax=axes[1], bat_name=bat_name)

    if bat_name != "0":
        spike_plots.ego_trajectory_spike_plot(dataframe, neuron, model_spikes, ax=axes[2], bat_name = bat_name)
        spike_plots.ego_rate_map_plot(dataframe, neuron, ax=axes[3], bat_name=bat_name)
    else:
        spike_plots.hd_plot_1d(dataframe, neuron, model_spikes, ax=axes[2])

def plot_basic_plots(dataframe, neuron, model_spikes, axes, bats, design_shape):
    # Plot basic plots
    for i, bat_id in enumerate(bats):
        # handle polar plot for head-direction plot for the recoreded bat
        if str(bat_id) == "0": axes[i*4+2] = plt.subplot2grid(design_shape, (i+1, 2), projection='polar')

        axes_per_bat = axes[i*4:i*4+4]
        plot_basic_plots_per_bat(dataframe, neuron, model_spikes, axes_per_bat, str(bat_id))

def plot_cm_plots(train_cm, test_cm, std_train_cm, std_test_cm, axes):
    avg_train_cm = train_cm
    avg_test_cm = test_cm
    if isinstance(train_cm, list):
        avg_train_cm = np.mean(train_cm, axis=(0))
        avg_test_cm = np.mean(test_cm, axis=(0))

    SVM_utils.plot_confusion_matrix(avg_train_cm, std_train_cm, ax=axes[0], normalize=True, title="Train")
    SVM_utils.plot_confusion_matrix(avg_test_cm, std_test_cm, ax=axes[1], normalize=True, title="Test")

def plot_feature_importance_no_shuffles(imp_table, agg_imp_table, axes, total_features):
    importance.plot_f_importance_from_table(imp_table[::-1], axes[0], CONF["FI_NUM"], total_features, None)
    importance.plot_f_importance_from_table(agg_imp_table[::-1], axes[1], CONF["FI_NUM"], total_features, None)

def plot_feature_importance_shuffles(imp_table, agg_imp_table, axes, total_features, shuffled_vals, train_cm, test_cm):
    f_imp_names, agg_imp_names, t_shuffled_vals, shuffles_df, sh_df, sh_df2, shuffled_b_accuracy = handle_shuffle_vals(shuffled_vals, imp_table, agg_imp_table, train_cm, test_cm)
    importance.plot_f_importance_from_table(imp_table[f_imp_names], axes[0], CONF["FI_NUM"], total_features, pd.concat(t_shuffled_vals[3]))
    sns.swarmplot(x="importance", y="name", hue="type", data=sh_df, ax=axes[0], order=f_imp_names[:CONF["FI_NUM"]])
    importance.plot_f_importance_from_table(imp_table[f_imp_names], axes[0], CONF["FI_NUM"], total_features, pd.concat(t_shuffled_vals[3]))
    axes[0].get_legend().remove() # shitty sns lib, doesn't let you remove the legend itself.

    importance.plot_f_importance_from_table(agg_imp_table[agg_imp_names], axes[1], CONF["FI_NUM"], total_features, pd.concat(t_shuffled_vals[4]))
    sns.swarmplot(x="importance", y="name", hue="type", data=sh_df2, ax=axes[1], order=agg_imp_names[:CONF["FI_NUM"]])
    importance.plot_f_importance_from_table(agg_imp_table[agg_imp_names], axes[1], CONF["FI_NUM"], total_features, pd.concat(t_shuffled_vals[4]))
    axes[1].get_legend().remove() # shitty sns lib, doesn't let you remove the legend itself.
    return t_shuffled_vals, shuffles_df, shuffled_b_accuracy

def plot_other_plots(shuffles_df, t_shuffled_vals, train_cm, test_cm, shuffled_b_accuracy, axes):
    sns.swarmplot(x="value", y="name", hue="type", data=shuffles_df, ax=axes[0])
    axes[0].get_legend().remove() # shitty sns lib, doesn't let you remove the legend itself.

    dfs = []
    cm_to_tp = lambda x: x[1][1] / (x[1][0] + x[1][1])
    cm_to_fp = lambda x: x[0][1] / (x[0][0] + x[0][1])
    # cm_to_fp = 
    shuffled_tp = list(map(cm_to_tp, t_shuffled_vals[1]))
    shuffled_fp = list(map(cm_to_fp, t_shuffled_vals[1]))

    cv_train_tp = list(map(cm_to_tp, train_cm))
    cv_test_tp = list(map(cm_to_tp, test_cm))

    cv_train_fp = list(map(cm_to_fp, train_cm))
    cv_test_fp = list(map(cm_to_fp, test_cm))

    axes[1].scatter(shuffled_fp, shuffled_tp, 5)
    axes[1].scatter(cv_train_fp, cv_train_tp)
    axes[1].scatter(cv_test_fp, cv_test_tp)

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    sns.distplot(shuffled_b_accuracy, bins=10, kde=True, ax=axes[2])

# TODO: this needs to be split into different functions, each plotting different things.
def plot_result_per_model(axes, fig, design_shape,\
 df, normalized_df, neuron, model,\
 train_cm, test_cm, imp_table, agg_imp_table, std_train_cm, std_test_cm, shuffled_vals):
    """ Plots basic plots for the best bats of the analysis.
        Plots confusion matrix plot.    
        Plots feature importance plot, with or without shuffles.  
        Stores image and show plot.  
    """

    N_BATS_TO_PLOT = 3
    # the axes for confusion matrices are after the bats axes. each bat get 4 plots.
    axes_for_cm = axes[N_BATS_TO_PLOT*4:N_BATS_TO_PLOT*4+2] 
    axes_for_fi = axes[N_BATS_TO_PLOT*4+2:N_BATS_TO_PLOT*4+4]
    total_features = len(imp_table.columns)

    best_bats_id = SVM_utils.get_best_bats_it(agg_imp_table.mean(0).sort_values().index, N_BATS_TO_PLOT)
    model_spikes = model.predict(normalized_df)
    
    plot_basic_plots(df, neuron, model_spikes, axes, best_bats_id, design_shape)        
    plot_cm_plots(train_cm, test_cm, std_train_cm, std_test_cm, axes_for_cm)
    
    # Plot feature importance plot
    if shuffled_vals != []:
        t_shuffled_vals, shuffles_df, shuffled_b_accuracy = plot_feature_importance_shuffles(imp_table, agg_imp_table, axes_for_fi, total_features, shuffled_vals, train_cm, test_cm)
        axes_for_others = axes[N_BATS_TO_PLOT*4+4:N_BATS_TO_PLOT*4+7]
        plot_other_plots(shuffles_df, t_shuffled_vals, train_cm, test_cm, shuffled_b_accuracy, axes_for_others)
    else:
        plot_feature_importance_no_shuffles(imp_table, agg_imp_table, axes_for_fi, total_features)
        shuffles_df = None
        
    # Store image and show plot
    plt.tight_layout()
    filename = str(uuid.uuid4())
    img_path = "pics/" + filename + ".png"
    plt.savefig(img_path, dpi=100)
    if CONF["SHOW_PLOTS"]: plt.show()
    plt.close(fig)
    
    return img_path, shuffles_df
        

# TODO: split this function into smaller pieces.
# TODO: add support for MIN_SPIKES configuration.
def cell_analysis(df, neuron, neuron_description="", new_confs={}):
    CONF.update(new_confs)
    
    if neuron.sum() < 500:
        print("Too few spikes (< 500)")
        return True
    if CONF["BINNING"]: df, neuron = SVM_utils.bin_df_and_neuron(df, neuron, bin_size=5)
    
    WEIGHT = dict(zip([0,1], len(neuron) / (2 * np.bincount(neuron) ** 0.5)))
    
    df = df.dropna()
    normalized_df = SVM_utils.manual_normalization(df)
    normalized_df.reset_index(drop=True, inplace=True)
    df_ = df.reset_index(drop=True)

    shuffled_vals = []
    threads = []
    
    neuron_dropped = neuron[df.index].reset_index(drop=True)
    if neuron_dropped.sum() < 500:
        print("Too few spikes (< 500)")
        return True
    if len(df_) < 50000: # ~ 33 minutes
        print("Too few data points ( < 50000)")
        return True
    
    for xid, shuffled_neuron in enumerate(shuffling.shuffling(neuron_dropped, CONF["N_SHUFFLES"])):
        shuffled_neuron = shuffled_neuron.astype('int')
        
        WEIGHT = dict(zip([0,1], len(shuffled_neuron) / (2*np.bincount(shuffled_neuron))))
        if CONF["SQRT_WEIGHT"]: WEIGHT = dict(zip([0,1], len(shuffled_neuron) / (2*np.bincount(shuffled_neuron)) ** 0.5))
            
        model = SVM_model(multi_threaded=True)
        model.set_weight(WEIGHT)
        thread = ThreadWithReturnValue(target=model.single_run, args=(normalized_df, shuffled_neuron, 0)) # no test on purpose!
        threads.append(thread)
        thread.start()

        if len(threads) == CONF["N_SHUFFLES"]:
            for thread in threads:
                svm_model, train_cm, test_cm, imp_table, agg_imp_table, std_train_cm, std_test_cm = thread.join()
                shuffled_vals.append( [svm_model, train_cm[0], test_cm[0], imp_table, agg_imp_table] )
                
    assert(len(shuffled_vals) == CONF["N_SHUFFLES"])    
    neuron = neuron[df.index].reset_index(drop=True)
    df = df.reset_index(drop=True)
            
    if neuron.sum() < 500:
        print("Too few spikes (< 500)")
        return True
    
    #model = SVM_model(cv=CV)
    model = SVM_model(cv=CONF["CV"], multi_threaded=True)
    WEIGHT = dict(zip([0,1], len(neuron) / ( 2 * np.bincount(neuron) )))
    if CONF["SQRT_WEIGHT"]: WEIGHT = dict(zip([0,1], len(neuron) / ( 2 * np.bincount(neuron) ) ** 0.5))
    
    model.set_weight(WEIGHT)
    design_shape, axes, fig = SVM_utils.design_axes("") # neuron description
    svm_model, train_cm, test_cm, imp_table, agg_imp_table, std_train_cm, std_test_cm = model(normalized_df, neuron)
    img_path, t_shuffles_df = plot_result_per_model(axes, fig, design_shape, df, normalized_df, neuron, svm_model, train_cm, test_cm, imp_table, agg_imp_table, std_train_cm, std_test_cm, shuffled_vals)
    return df, normalized_df, neuron, svm_model, train_cm, test_cm, imp_table, agg_imp_table, std_train_cm, std_test_cm, shuffled_vals, img_path, t_shuffles_df

def exclude_bats(dataset, exclude=[]):
    remove_features = []
    for b in exclude:
        remove_features += SVM_utils.get_bat_features(dataset, b).to_list()

    dataset = dataset.drop(columns=remove_features)
    return dataset

def behavioral_data_to_dataframe(behavioral_data_path, net="NET1", conf={}):
    CONF.update(conf)
    df = SVM_utils.get_df_from_file_path(behavioral_data_path, CONF["REAL_BEHAVIORAL_DATA"], CONF["CACHED_BEHAVIORAL_DATA"], net)
    df = exclude_bats(df, exclude)
    if CONF["FE"]: df = feature_engineering.engineer(df, CONF)
    return df

def run_experiment():
    """ An example on how to run experiments with the library.
        Takes behavioral data matlab files from a specific directory.  
        Parses them into a dataframe.  
        Loads spikes that were recorded for that day.  
        Runs cell analysis.  
    """
    behavioral_files = glob.glob("data/behavioural_data/raw/b2*.mat")[1:]
    
    for real_data_file in behavioral_files:
        print("*" * 80 + f"\nfile path: {real_data_file}\n" + "*" * 80)
        import re
        day = re.findall("_d(\d+)_", real_data_file)[0]
        print(day)
        
        df = behavioral_data_to_dataframe(real_data_file)
        spikes, titles = spike_loading(day)
        if not len(spikes): continue
        
        if CONF["NOISE_CANCELLATION"]: spikes = list(map(SVM_utils.noise_cancellation, spikes))
        [print(s.mean()) for s in spikes]
        for nid, n in enumerate(spikes): 
            print_analysis_header(nid, spikes, f"\nDay: {day}, Neuron: {titles[nid]}")
            df_ = df.copy()
            t = time.time()
            cell_analysis(df_, n, "Real Neuron")
            print("Time for cell analysis:", time.time() - t)

if __name__ == "__main__":
    run_experiment()
    print("Done!")