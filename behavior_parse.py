import h5py
import numpy as np
import pandas as pd
import dataset
from constants import *
import config

BAT_NAME_TO_ID = dict()

NET1_MIN_X = 150
NET1_MAX_X = 250
NET1_MIN_Y = 10
NET1_MAX_Y = 60

NET3_MIN_X = 25
NET3_MAX_X = 75
NET3_MIN_Y = 170
NET3_MAX_Y = 220


def construct_bat_name_to_id_map(f):
    global BAT_NAME_TO_ID
    nbats = config.Config.get("N_BATS")

    bat_names = []
    for bat_id in range(nbats):
        bat_name = chr(f[f['simplified_behaviour']['name'][bat_id][0]][0][0])
        if bat_name != config.Config.get("RECORED_BAT"):
            bat_names.append(bat_name)

    BAT_NAME_TO_ID = dict(zip(bat_names, range(1, len(bat_names)+1)))
    BAT_NAME_TO_ID[config.Config.get("RECORED_BAT")] = 0

def points_to_hd(x1, y1, x2, y2):
    dx = (x2 - x1)
    dy = (y2 - y1)
    with np.errstate(invalid='ignore'):
        hd = np.round((np.arctan2(dy, dx) % (2 * np.pi)) * 180 / np.pi)
    return hd


def extract_data_from_file(f, i):
    global BAT_NAME_TO_ID
    name = chr(f[f["simplified_behaviour"]["name"][i][0]][0][0])
    bat_id = BAT_NAME_TO_ID[name]
    x1 = f[f["simplified_behaviour"]["pos_on_net"][i][0]][0] * 100
    y1 = f[f["simplified_behaviour"]["pos_on_net"][i][0]][1] * 100
    x2 = f[f["simplified_behaviour"]["pos_on_net"][i][0]][2] * 100
    y2 = f[f["simplified_behaviour"]["pos_on_net"][i][0]][3] * 100

    return name, bat_id, x1, y1, x2, y2


def split_to_nets(df):
    x1 = df.X
    y1 = df.Y

    net1 = (NET1_MIN_X <= x1) & (x1 <= NET1_MAX_X) & (NET1_MIN_Y <= y1) & (y1 <= NET1_MAX_Y)
    net3 = (NET3_MIN_X <= x1) & (x1 <= NET3_MAX_X) & (NET3_MIN_Y <= y1) & (y1 <= NET3_MAX_Y)

    df['net1'] = net1
    df['net3'] = net3

    df_net1 = df.copy()
    df_net1[~df_net1.net1] = np.nan

    df_net3 = df.copy()
    df_net3[~df_net3.net3] = np.nan

    df_net1.drop(columns=["net1", "net3"], inplace=True)
    df_net3.drop(columns=["net1", "net3"], inplace=True)
    df.drop(columns=["net1", "net3"], inplace=True)

    df_net1.X -= NET1_MIN_X
    df_net1.Y -= NET1_MIN_Y
    df_net3.X -= NET3_MIN_X
    df_net3.Y -= NET3_MIN_Y

    return df, df_net1, df_net3


def parse_matlab_file(path):
    # path = "data/behavioural_data/b2305_d191220_simplified_behaviour.mat"
    f = h5py.File(path, "r")
    IDs, DF1s, DF3s, DFs = [], [], [], []

    nbats = f['simplified_behaviour']['name'].shape[0]
    config.Config.set("N_BATS", nbats)
    construct_bat_name_to_id_map(f)

    for i in range(nbats):
        name, bat_id, x1, y1, x2, y2 = extract_data_from_file(f, i)
        IDs.append(bat_id)

        # Head Direction only for recorded bat (Bat 0)
        hd = np.nan
        if bat_id == 0:
            hd = points_to_hd(x1, y1, x2, y2)

        df = pd.DataFrame({"X": x1, "Y": y1, "HD": hd, "bat_id": bat_id})
        df, df_net1, df_net3 = split_to_nets(df)

        # DFs.append(df)
        DF1s.append(df_net1)
        DF3s.append(df_net3)

    # sort dataframes by id
    IDs, DF1s, DF3s = list(zip(*sorted(zip(IDs, DF1s, DF3s))))
    DF1s = dict(zip(range(nbats), DF1s))
    DF3s = dict(zip(range(nbats), DF3s))

    # DF1s = dataset.add_pairs_bats(DF1s)
    # DF3s = dataset.add_pairs_bats(DF3s)

    df1 = dataset.build_dataset_inline(DF1s)
    df3 = dataset.build_dataset_inline(DF3s)

    return df1, df3
