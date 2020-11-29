#!/usr/bin/env python3

import sys
import analysis_lib
import parse_real_neural_data
from pathlib import Path
import os
import pickle
from shutil import copyfile
import json 
import argparse
import pandas as pd
import logging

def handle_args(args):
    """ Handles arguments with argparse.  
        Verifies that behavioral and neuronal data days are matching.
    """
    parser = argparse.ArgumentParser(description='Analyze cell.')
    parser.add_argument('bpath', metavar='behavioral_path', type=str, nargs=1, help='Path for behavioral data (mat/csv)')
    parser.add_argument('npath', metavar='neural_path', type=str, nargs=1, help='Path for neural data (mat/csv)')
    parser.add_argument('cpath', metavar='config_path', type=str, nargs='?', help='Path for configuration file (json)', default='config.json')
    parser.add_argument('opath', metavar='output_dir', type=str, nargs='?', help='Output directory', default='data/results')

    args = parser.parse_args()
    
    behavioral_data_path = args.bpath[0]
    neural_data_path = args.npath[0]
    conf = json.load(open(args.cpath))
    output_path = args.opath

    bat_name, day, _, _ = Path(behavioral_data_path).stem.split('_')
    nid, bat_name2, day2 = Path(neural_data_path).stem.split('_')
    
    bat_names_err_msg = "Warning behavioral bat name:", bat_name, "and neuronal bat name:", bat_name2, "are not equal!"
    days_err_msg = "Warning behavioral day:", day, "and neuronal day:", day2, "are not equal!"

    if bat_name != bat_name2:
        logging.warning(bat_names_err_msg)

    if day != day2:
        logging.warning(days_err_msg)

    return behavioral_data_path, neural_data_path, conf, output_path, nid, day2

def create_new_results_dir(nid, day, output_path="/data/results"):
    """ Scans the results path, assume all results dirs starts with an index.  
        Finds the next index and create a directory for it, where the results will be stored.  
    """
    dirs = ["-1 nid day"] + os.listdir(output_path)
    dirs_idx = list(map(lambda x: x.split(' ')[0], dirs))
    new_dir_idx = max(map(lambda x: int(x) if x.isnumeric() else 0, dirs_idx)) + 1
    new_dir_path = f"{output_path}/{new_dir_idx} - {nid} {day}"
    os.mkdir(new_dir_path)
    return new_dir_path
                    

def store_results(results, output_path, nid, day, conf):
    """ `results` is a tuple returned by cell_analysis.  
        Stores the results in csvz in some output folder.  
        Returns the folder path of the results.  
    """
    logging.info("storing results...")
    df, normalized_df, neuron, svm_model, train_cm, test_cm, imp_table, agg_imp_table, std_train_cm, std_test_cm, shuffled_vals, img_path, shuffles_df = results
    new_dir_path = create_new_results_dir(nid, day, output_path)

    if conf.get('STORE_DATAFRAME', 1):
        df.to_csv(f"{new_dir_path}/dataframe.csv.zip")
    if conf.get('STORE_NORMALIZED_DATAFRAME', 1):
        normalized_df.to_csv(f"{new_dir_path}/normalized_df.csv.zip")
    neuron.to_csv(f"{new_dir_path}/neuron.csv.zip")

    # pickle.dump(svm_model, open(f"{new_dir_path}/model.pkl", 'wb')) -> instead of pickling the model, we store only the coefficients.
    d = dict(zip(df.columns.to_list(), svm_model.get_importances()))
    for k in d: d[k] = [d[k]]
    df = pd.DataFrame.from_dict(d)
    df.to_csv(f"{new_dir_path}/model_coeffs.csv")

    imp_table.to_csv(f"{new_dir_path}/imp_table.csv")
    agg_imp_table.to_csv(f"{new_dir_path}/agg_imp_table.csv")

    shuffled_imp_table = pd.concat(list(list(zip(*shuffled_vals))[3]), ignore_index=True)
    shuffled_agg_imp_table = pd.concat(list(list(zip(*shuffled_vals))[4]), ignore_index=True)

    shuffled_imp_table.to_csv(f"{new_dir_path}/shuffled_imp_table.csv")
    shuffled_agg_imp_table.to_csv(f"{new_dir_path}/shuffled_agg_imp_table.csv")

    copyfile(img_path, f"{new_dir_path}/plot.png")
    
    if shuffles_df is not None: shuffles_df.to_csv(f"{new_dir_path}/shuffles_dataframe.csv")
    return new_dir_path

def main(args):
    """ Expects paths for running and storing the analaysis.  
    Checks and converts the neural data to binary.  
    Stores results and pop-up the results directory.  
    """
    behavioral_data_path, neural_data_path, conf, output_path, nid, day = handle_args(args)
    
    dataset = analysis_lib.behavioral_data_to_dataframe(behavioral_data_path, conf)
    neuron = parse_real_neural_data.parse_neural_data_from_path(neural_data_path, behavioral_data_path)
    
    if len(neuron.value_counts()) != 2:
        logging.warning("Error, neural data is not binary!")
        logging.warning("replacing all > 1 labels with 1")
        neuron[neuron > 0] = 1 # make spikes a binary vector
    
    results = analysis_lib.cell_analysis(dataset, neuron, "Real Neuron", conf)
    results_dir = store_results(results, output_path, nid, day, conf)
    logging.info("Done!")
    if conf['POPUP_RESULTS']: os.startfile(os.getcwd() + "/" + results_dir)
    
if __name__ == "__main__":
    main(sys.argv)