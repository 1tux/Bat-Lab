import pandas as pd
import h5py
import functools
import os.path
from pathlib import Path

def parse_behavioral_time_table_from_path(path):
    behavioral_file = h5py.File(path, "r")
    time_table = behavioral_file[behavioral_file["simplified_behaviour"]["time"][0][0]]
    time_table_length = time_table.shape[0]
    POWER_OF_TWO = [2**i for i in range(30)]
    l = []

    for i in range(time_table_length-1):
        tf = time_table[i][0]
        next_tf = time_table[i+1][0]

        l.append(pd.Interval(tf, next_tf))

    return l

def parse_behavioral_time_table(day, recorded_bat):
    behavioral_path = f"data/behavioural_data/raw/{recorded_bat}_{day}_simplified_behaviour.mat"
    return parse_behavioral_time_table_from_path(behavioral_data_path)

def parse_neural_data_internal_from_path(path, intervals):
    f = h5py.File(path, "r")

    neural_table = f["cell_struct"]["spikes_ts_msec"]
    neural_table_length = neural_table.shape[0]
    inter_idx = 0
    POWER_OF_TWO = [2**i for i in range(30)]
    d = {0 : 0}

    for i in range(neural_table_length):

        current_time = neural_table[i][0]
        if current_time < intervals[0].left: continue
        if current_time > intervals[-1].right: break

        while inter_idx < len(intervals):
            if current_time not in intervals[inter_idx]:
                inter_idx += 1
                d[inter_idx] = 0
            else:
                break
        if current_time in intervals[inter_idx]:
            d[inter_idx] += 1

    # >=, adding 0 at the end, is on purpse
    while inter_idx <= len(intervals):
        d[inter_idx] = 0
        inter_idx += 1

    result = pd.Series(d.values(), d.keys())
    return result

def parse_neural_data_internal(neuron_id, day, recorded_bat, intervals):
    path = f"data/neural_data/raw/{neuron_id}_{recorded_bat}_{day}.mat"
    return parse_neural_data_internal_from_path(path, intervals)


def parse_neural_data_from_path(path, behavioral_data_path=""):
    if os.path.exists(path):
        if path.endswith("csv"): return pd.read_csv(path)['0']
        if path.endswith("mat"):
            output_path = path.replace(".mat", ".csv")
            # split into neuron id, day, recoreded bat
            # find the correct behavioral data file
            # save the results as csv

            if os.path.exists(output_path):
                print("Warning: corresponding csv file already exists. loading csv file")
                return pd.read_csv(output_path)['0']

            if behavioral_data_path == "":
                print("Error, can't parse matlab file without the corresponding behavioral data path")
                return -1
            if not os.path.exists(behavioral_data_path):
                print("Error, behavioral data path was not found", behavioral_data_path)
                return -1

            print("Parsing behavioral data....")
            intervals = parse_behavioral_time_table_from_path(behavioral_data_path)
            print("Parsing neural data.....")
            result = parse_neural_data_internal_from_path(path, intervals)
            print("storing neural data")
            new_path = path.replace(".mat", ".csv")
            result.to_csv(output_path)
            return result
    else:
        print("Error! path:", path, "doesn't exist")
        print("current working dir is:", os.getcwd())
        print("trying changing path / using a script to convert matlab neural data to csv")
        return -1

def parse_neural_data(neuron_id='55', day='d191220', recorded_bat = 'b2305', data_dir="data/neural_data"):
    path = f"/parsed/{data_dir}/{neuron_id}_{recorded_bat}_{day}.csv"
    if os.path.exists(path): return pd.read_csv(path)['0']

    intervals = parse_behavioral_time_table(day, recorded_bat)
    result = parse_neural_data_internal(neuron_id, day, recorded_bat, intervals)

    return result
