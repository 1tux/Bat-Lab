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
import shutil
import os.path
import config

logger = logging.getLogger()
fh = logging.FileHandler('log.log', mode='w')
logger.addHandler(fh)

def handle_args():
    """ Handles arguments with argparse.  
        Verifies that behavioral and neuronal data days are matching.
    """
    parser = argparse.ArgumentParser(description='Analyze cell.')
    parser.add_argument('bpath', metavar='behavioral_path', type=str, help='Path for behavioral data (mat/csv)')
    parser.add_argument('npath', metavar='neural_path', type=str, help='Path for neural data (mat/csv)')
    parser.add_argument('cpath', metavar='config_path', type=str, nargs='?', help='Path for configuration file (json)',
                        default='config.json')
    parser.add_argument('opath', metavar='output_dir', type=str, nargs='?', help='Output directory',
                        default='data/results')
    parser.add_argument('-n', metavar='net', type=int, help='which net, could be 1 or 3', default=1)
    parser.add_argument('-r', type=str, default='A')
    parser.add_argument('-X', metavar='eXclude', type=str, nargs='*', default=[])
    parser.add_argument('-I', metavar='Include', type=str, nargs='*', default=[])

    args = parser.parse_args()

    behavioral_data_path = args.bpath
    neural_data_path = args.npath
    config.Config.from_file(args.cpath)
    output_path = args.opath
    try:
        net = {1: "NET1", 3: "NET3"}[args.n]
    except:
        raise Exception("Wrong Net! should have been either 1 or 3 %s" % str(args.n))

    exclude = args.X
    include = args.I
    assert not exclude or not include, "exclude and include arguments are mutually exclusive"

    config.Config.set("RECORED_BAT", args.r)

    bat_name, day, _, _ = Path(behavioral_data_path).stem.split('_')
    nid, bat_name2, day2 = Path(neural_data_path).stem.split('_')

    bat_names_err_msg = "Warning behavioral bat name:", bat_name, "and neuronal bat name:", bat_name2, "are not equal!"
    days_err_msg = "Warning behavioral day:", day, "and neuronal day:", day2, "are not equal!"

    if bat_name != bat_name2:
        logging.warning(bat_names_err_msg)

    if day != day2:
        logging.warning(days_err_msg)

    return behavioral_data_path, neural_data_path, output_path, nid, day2, net, exclude


def create_new_results_dir(nid, day, output_path="/data/results"):
    """ Scans the results path, assume all results dirs starts with an index.  
        Finds the next index and create a directory for it, where the results will be stored.  
    """
    dirs = ["-1 nid day"] + os.listdir(output_path)
    dirs_idx = list(map(lambda x: x.split(' ')[0], dirs))
    new_dir_idx = max(map(lambda x: int(x) if x.isnumeric() else 0, dirs_idx)) + 1
    new_dir_path = f"{output_path}/{new_dir_idx} - {nid} {day}"
    new_dir_path_override = f"{output_path}/LAST - {nid} {day}"
    os.mkdir(new_dir_path)
    return new_dir_path, new_dir_path_override


def store_img_plot(img_path, nid, day, new_dir_path):
    copyfile(img_path, f"{new_dir_path}/plot {nid} {day}.png")
    img_path_p = Path(img_path)
    uid = -1
    while uid == -1 or new_img_path_p.exists():
        uid += 1
        new_img_path_p = img_path_p.parent / f"{nid} {day} {uid}.png"
    img_path_p.rename(new_img_path_p)
    # print(img_path_p)
    # print(new_img_path_p)

def model_coeffs_to_store(columns_names, coeffs):
    dfs = []
    for coeff in coeffs:
        d = dict(zip(columns_names, coeff))
        for k in d: d[k] = [d[k]]
        df = pd.DataFrame.from_dict(d)
        dfs.append(df)
    df = pd.concat(dfs)
    return df

def cms_to_store(confusion_matrices):
    column_names = ["Actual=0, Predicted=0", "Actual=0, Predicted=1","Actual=1, Predicted=0","Actual=1, Predicted=1"]
    confusion_matrices = map(lambda x: x.reshape(4), confusion_matrices)

    return pd.DataFrame(confusion_matrices, columns=column_names)


def store_results(results, output_path, nid, day):
    """ `results` is a tuple returned by cell_analysis.  
        Stores the results in csvz in some output folder.  
        Returns the folder path of the results.  
    """
    logging.info("storing results...")
    logging.shutdown()
    df, normalized_df, neuron, svm_model, train_cm, test_cm, imp_table, agg_imp_table, std_train_cm, std_test_cm, shuffled_vals, img_path, shuffles_df = results
    new_dir_path, new_dir_path_override = create_new_results_dir(nid, day, output_path)

    t_shuffled_vals = list(zip(*shuffled_vals))
    shuffles_cm_df = cms_to_store(t_shuffled_vals[1])
    train_cm_df = cms_to_store(train_cm)
    test_cm_df = cms_to_store(test_cm)
    shuffles_cm_df.to_csv(f"{new_dir_path}/shuffles_cm_table.csv")
    train_cm_df.to_csv(f"{new_dir_path}/train_cm_table.csv")
    test_cm_df.to_csv(f"{new_dir_path}/test_cm_table.csv")

    if config.Config.get("STORE_DATAFRAME"):
        df.to_csv(f"{new_dir_path}/dataframe.csv.zip")
    if config.Config.get("STORE_NORMALIZED_DATAFRAME"):
        normalized_df.to_csv(f"{new_dir_path}/normalized_df.csv.zip")
    neuron.to_csv(f"{new_dir_path}/neuron.csv.zip")

    # pickle.dump(svm_model, open(f"{new_dir_path}/model.pkl", 'wb')) -> instead of pickling the model, we store only the coefficients.
    columns_names = df.columns.to_list()
    model_coeffs_df = model_coeffs_to_store(columns_names, svm_model.get_all_importances())
    model_coeffs_df.to_csv(f"{new_dir_path}/model_coeffs.csv")

    shuffle_coeffs_df = model_coeffs_to_store(columns_names, [x.get_importances() for x in t_shuffled_vals[0]])
    shuffle_coeffs_df.to_csv(f"{new_dir_path}/shuffles_coeffs.csv")

    imp_table.to_csv(f"{new_dir_path}/imp_table.csv")
    agg_imp_table.to_csv(f"{new_dir_path}/agg_imp_table.csv")

    shuffled_imp_table = pd.concat(list(list(zip(*shuffled_vals))[3]), ignore_index=True)
    shuffled_agg_imp_table = pd.concat(list(list(zip(*shuffled_vals))[4]), ignore_index=True)

    shuffled_imp_table.to_csv(f"{new_dir_path}/shuffled_imp_table.csv")
    shuffled_agg_imp_table.to_csv(f"{new_dir_path}/shuffled_agg_imp_table.csv")

    store_img_plot(img_path, nid, day, new_dir_path)
    copyfile("log.log", f"{new_dir_path}/log.log")
    copyfile(config.Config.get("CONF_PATH"), f"{new_dir_path}/config.json")


    if shuffles_df is not None: shuffles_df.to_csv(f"{new_dir_path}/shuffles_dataframe.csv")
    open(f"{new_dir_path}/execution_line.txt", "w").write(" ".join(sys.argv))

    if config.Config.get('OVERWRITE'):
        # overwrite last neuron directory
        if os.path.exists(new_dir_path_override):
            shutil.rmtree(new_dir_path_override)
        shutil.copytree(new_dir_path, new_dir_path_override)

    return new_dir_path


def main():
    """ Expects paths for running and storing the analaysis.  
    Checks and converts the neural data to binary.  
    Stores results and pop-up the results directory.  
    """
    behavioral_data_path, neural_data_path, output_path, nid, day, net, exclude = handle_args()

    dataset = analysis_lib.behavioral_data_to_dataframe(behavioral_data_path, net, exclude)
    neuron = parse_real_neural_data.parse_neural_data_from_path(neural_data_path, behavioral_data_path)

    # print(dataset.columns)

    if len(neuron.value_counts()) != 2:
        logging.warning("Error, neural data is not binary!")
        logging.warning("replacing all > 1 labels with 1")
        neuron[neuron > 0] = 1  # convert spikes vector to a binary vector

    results = analysis_lib.cell_analysis(dataset, neuron, "Real Neuron")
    results_dir = store_results(results, output_path, nid, day)
    logging.info("Done!")
    if config.Config.get('POPUP_RESULTS'): os.startfile(os.getcwd() + "/" + results_dir)


if __name__ == "__main__":
    main()
