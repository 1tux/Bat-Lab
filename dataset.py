# converting each perosnal bats data
# to information bat0 is holding for all others

import constants
import imp
imp.reload(constants)
from constants import *
import numpy as np
import pandas as pd

import recordings
imp.reload(recordings)
from recordings import load_data, generate_data
import os
import re

def replace_nans_with_rand(df):
    import numpy as np; import pandas as pd
    M = len(df.index)
    N = len(df.columns)
    ran = pd.DataFrame(np.random.uniform(-2, -1, (M,N)), columns=df.columns, index=df.index)
    df.update(ran, overwrite=False)
    return df

def load_bats_data():
    bats = {}
    for i in range(N_BATS):
        X, Y, HD = load_data(i)
        X = pd.Series(X)
        Y = pd.Series(Y)
        HD = pd.Series(HD)

        df = pd.DataFrame({"X" : X, "Y" : Y, "HD": HD, "bat_id": i})
        bats[i] = df
    return bats

def get_bats_names(include_nearest=False):
    res = list(map(str, range(N_BATS)))
    if include_nearest:
        res += ["n"]
    
    # res += ["E12", "E13", "E14", "E23", "E24", "E34"]
    return res

def get_other_bats_names():
    return get_bats_names()[1:] # no Bat0

def get_feature_names():
    return ["X", "Y", "A", "D", "HD", "ID"]

def get_col_name(bat_name, feature_name, PREFIX="BAT"):
    if PREFIX not in ["BAT", "PAIR"]:
        raise Exception("Unknown prefix for feature!")

    if feature_name in get_feature_names():
        prefix = "F"
    else:
        prefix = "EF"

    if len(str(bat_name)) == 2:
        PREFIX = "PAIR"
        prefix = "EF"
    col_name = f"{PREFIX}_{bat_name}_{prefix}_{feature_name}" 
        
    return col_name

def extract_bat_name(col_name):
    return re.findall('^(.*?)_(.*?)_', col_name)[0][1]

def extract_feature_name(col_name):
    return re.findall('F_(.*?)$', col_name)[0]

def build_dataset_inline(bats):
    # expects bat list, which is a list of dataframes, having X,Y,HD(not important), and bat_id
    
    bat0 = bats[0].copy() # x,y,hd of implanted bat
    bat0.drop("bat_id", axis=1,inplace=True)

    for i in bats.keys():
        if i == 0: continue
        # relative position (4. 1 for each bat) - V
        # relative position from nearest bat - V
        # absolute position for other bat (8. X,Y - for each bat) - V
        # bat id for nearest bat (1-4) - V
        # absolute position for nearest bat - V
        # angle toward each of the bats, relative to HD
        
        bat0[get_col_name(i, "D")] = np.sqrt((bats[0].X - bats[i].X) ** 2 + (bats[0].Y - bats[i].Y) ** 2)
        bat0[get_col_name(i, "X")] = bats[i].X
        bat0[get_col_name(i, "Y")] = bats[i].Y
        
        dx = bats[i].X - bat0.X
        dy = bats[i].Y - bat0.Y
        
        bat0[get_col_name(i, "A")] = (bat0.HD - ((np.arctan2(dy, dx) % (2*np.pi)) * 180 / np.pi)) % 360
    
    bat0 = bat0.rename(columns={\
                                "X" : get_col_name(0, "X"),\
                                "Y" : get_col_name(0, "Y"),\
                                "HD" : get_col_name(0, "HD")\
                               })
    
    # remove NAN values
    #print("BEFORE AND AFTER REMOVING NANS")
    #print(bat0.shape)
    #bat0.dropna(inplace=True)
    #print(bat0.shape)
    
    # removing nearest bat
    """
    distance_columns = [get_col_name(i, "D") for i in range(1, N_BATS)]
    bat0[get_col_name("n", "D")] = bat0[distance_columns].min(axis=1)
    bat0[get_col_name("n", "ID")] = bat0[distance_columns].idxmin(axis=1).str.extract(r'BAT_(.)_F_D').astype('float')

    nearest_xs = []
    nearest_ys = []
    nearest_as = []
    for i in range(1, N_BATS):
        nb = bat0[bat0[get_col_name("n", "ID")] == i]
        nearest_xs.append( nb[get_col_name(i, "X")] )
        nearest_ys.append( nb[get_col_name(i, "Y")] )
        nearest_as.append( nb[get_col_name(i, "A")] )
    

    bat0[get_col_name("n", "X")] = pd.concat(nearest_xs).sort_index()
    bat0[get_col_name("n", "Y")] = pd.concat(nearest_ys).sort_index()
    bat0[get_col_name("n", "A")] = pd.concat(nearest_as).sort_index()
    """
    return bat0

def add_pairs_bats(bats):
    for i in range(1, N_BATS):
        for j in range(i+1, N_BATS):
            pair = [i, j]
            dx = bats[i].X - bats[j].X
            dy = bats[i].Y - bats[j].Y
            
            avg_x = (bats[i].X + bats[j].X) / 2
            avg_y = (bats[i].Y + bats[j].Y) / 2
            avg_hd = (bats[i].HD + bats[j].HD) / 2
            
            pair_distance = np.sqrt(dx ** 2 + dy ** 2)
            bat_ij_X = avg_x
            bat_ij_Y = avg_y
            bat_ij_HD = avg_hd
            
            bat_ij_X[pair_distance > 10] = np.nan
            bat_ij_Y[pair_distance > 10] = np.nan
            bat_ij_HD[pair_distance > 10] = np.nan
            
            bat_ij = pd.DataFrame({"X" : bat_ij_X, "Y" : bat_ij_Y, "HD" : bat_ij_HD})
            bats[f"E{i}{j}"] = bat_ij
            
    return bats

def build_dataset(generate = False):
    if generate:
        generate_data()
        
    bats = load_bats_data()
    # bats = add_pairs_bats(bats)
    df = build_dataset_inline(bats)
    return df
    

def get_dataset_path():
    return os.path.join(DATA_PATH, "dataset.csv")

def store_dataset(generate = False):
    df = build_dataset(generate)
    df.to_csv(get_dataset_path(), index=False)
    
    
def load_dataset():
    return pd.read_csv(get_dataset_path())