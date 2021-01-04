from bat_class import Bat
import constants
import imp

imp.reload(constants)
from constants import *
import pickle
import os
import random
import numpy as np


def get_path(i, param):
    return os.path.join(DATA_PATH, f"bat_{i}_{param}.pkl")


def generate_data_per_bat():
    b = Bat()
    Xs = []
    Ys = []
    HDs = []
    for i in range(TIME_POINTS):
        b.behave()
        # only decimal precision

        if random.random() < MISSING_DATA_PROB:
            Xs.append(np.nan)
            Ys.append(np.nan)
            HDs.append(np.nan)
        else:
            Xs.append(int(b.X))
            Ys.append(int(b.Y))
            HDs.append(int(b.hd))

    print("generated bat data!")
    return Xs, Ys, HDs


def generate_data(store=True):
    Xss = []
    Yss = []
    HDss = []

    for i in range(N_BATS):
        Xs, Ys, HDs = generate_data_per_bat()
        Xss.append(Xs)
        Yss.append(Ys)
        HDss.append(HDss)

        if store:
            pickle.dump(Xs, open(get_path(i, "Xs"), "wb"))
            pickle.dump(Ys, open(get_path(i, "Ys"), "wb"))
            pickle.dump(HDs, open(get_path(i, "HDs"), "wb"))

    return Xss, Yss, HDss


def load_data(i):
    Xs = pickle.load(open(get_path(i, "Xs"), "rb"))
    Ys = pickle.load(open(get_path(i, "Ys"), "rb"))
    HDs = pickle.load(open(get_path(i, "HDs"), "rb"))
    return Xs, Ys, HDs
