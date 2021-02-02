import itertools
import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.preprocessing import minmax_scale
from sklearn.utils import resample

import behavior_parse
import dataset
import config


def upsample(df, neuron):
    # print("upsampling...")

    # we need this to avoid annoying warning regarding 'SettingWithCopyWarning'
    df_copy = df.copy()

    if "neuron" in df_copy.columns.to_list():
        df_copy.drop(columns=['neuron'], inplace=True)

    df_copy['neuron'] = pd.Series(neuron)  # .copy()

    df_majority = df_copy[df_copy.neuron == 0]
    df_minority = df_copy[df_copy.neuron == 1]

    # this is a fix, sometimes 1s appear more than 0s!
    if df_majority.shape[0] < df_minority.shape[0]:
        df_majority, df_minority = df_minority, df_majority

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=1337)  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    df_upsampled = df_upsampled.reset_index(drop=True)
    neuron_upsampled = df_upsampled['neuron']
    df_upsampled.drop(columns=['neuron'], inplace=True)
    df_copy.drop(columns=['neuron'], inplace=True)
    # print("upsampled!")
    return df_upsampled, neuron_upsampled


def get_bat_features(df, bat_name):
    return df.columns[df.columns.str.match(f'BAT_{bat_name}')]


def angle_avg(angles):
    # convert to x, y. avg them, convert back to angle
    radians = angles * np.pi / 180
    avg_x = np.cos(radians).mean()
    avg_y = np.sin(radians).mean()
    avg_radian = np.arctan2(avg_y, avg_x) % (2 * np.pi)
    avg_angle = (avg_radian * 180 / np.pi) % 360
    return avg_angle


def id_avg(ids):
    # most frequent
    return ids.mode().sample(1).iloc[0]


def bin_func(grouped_df):
    mean_cols = []
    mean_features = ['X', 'Y']
    for b in dataset.get_bats_names():
        for f in mean_features:
            mean_cols.append(dataset.get_col_name(b, f))

    result_df = grouped_df[mean_cols].mean()
    hd_col_name = dataset.get_col_name("0", "HD")
    result_df = pd.concat([result_df, grouped_df[[hd_col_name]].apply(angle_avg)])

    return result_df


def remove_best_bat_features(df, best_bat_id):
    cols = ["X", "Y", "X^2", "Y^2"]
    columns_to_delete = [best_bat_id + str(x) for x in cols]
    df_r = df.copy()
    df_r.drop(columns=columns_to_delete, inplace=True)
    return df_r


def bin_dataset(df, bin_size):
    df2 = df.copy()
    df2['group'] = df2.index.to_series() // bin_size

    t = time.time()
    df3 = df2.groupby('group').apply(bin_func)
    print("took:", time.time() - t, "sec")

    bats = []
    for b in dataset.get_bats_names():
        cols = df3.columns.to_series().str.startswith(f"BAT_{b}")
        new_cols_names = (cols[cols].index.str.extract('_F_(.*)'))
        bat_i_df = df3[cols[cols].index.to_series()]
        new_cols_names = new_cols_names[0].values
        map_dict = dict(zip(cols[cols].index, new_cols_names))
        bat_i_df = bat_i_df.rename(columns=map_dict)
        bat_i_df['bat_id'] = b
        bats.append(bat_i_df)

    # imp.reload(dataset)
    grouped_df = dataset.build_dataset_inline(bats)

    # feature engineering
    e_df = grouped_df.copy()

    e_df[dataset.get_col_name(0, "X^2")] = e_df[dataset.get_col_name(0, "X")] ** 2
    e_df[dataset.get_col_name(0, "Y^2")] = e_df[dataset.get_col_name(0, "Y")] ** 2
    e_df[dataset.get_col_name(0, "HD^2")] = e_df[dataset.get_col_name(0, "HD")] ** 2

    for bat_name in dataset.get_other_bats_names():
        for f_name in ["X", "Y", "A", "D"]:
            e_df[dataset.get_col_name(bat_name, f_name + "^2")] = e_df[dataset.get_col_name(bat_name, f_name)] ** 2

    return e_df


def bin_df_and_neuron(df, neuron, bin_size=5):
    e_df = bin_dataset(df, bin_size)
    binned_spikes = neuron.groupby(neuron.index.to_series() // bin_size).apply(lambda x: x.sum())

    # multiply features according to neuron spike
    res_df = pd.DataFrame(np.repeat(e_df.values,
                                    binned_spikes.replace(0, 1).tolist(),
                                    axis=0),
                          columns=e_df.columns)
    binned_spikes = pd.Series(np.repeat(binned_spikes.values, binned_spikes.replace(0, 1).tolist())).reindex(
        res_df.index)
    binned_spikes[binned_spikes != 0] = 1
    return res_df, binned_spikes


def cm_to_ba(cm):
    # confusion matrix to balanaced accuracy
    total_zeros = cm[0][0] + cm[0][1]
    total_ones = cm[1][0] + cm[1][1]

    total = total_ones + total_zeros

    weight_ones = 1 - total_ones / total
    weight_zeros = 1 - weight_ones

    correct_ones = cm[1][1] / total_ones
    correct_zeroes = cm[0][0] / total_zeros

    balanced_accuracy = weight_ones * correct_ones + weight_zeros * correct_zeroes

    return balanced_accuracy


def plot_confusion_matrix(cm,
                          std_cm,
                          ax=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    if ax is None:
        fig, ax = plt.subplots()

    accuracy = np.trace(cm) / float(np.sum(cm))
    balanced_accuracy = cm_to_ba(cm)

    if cmap is None:
        cmap = plt.get_cmap('coolwarm')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    # plt.colorbar()

    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    if normalize:
        std_cm = np.round(std_cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2) * 100
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2) * 100

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, "E: {:0.2f}\nstd: {:0.2f}".format(cm[i, j], std_cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            ax.text(j, i, "E: {:0.2f}\nstd: {:0.2f}".format(cm[i, j], std_cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted\nb-accuracy={:0.4f}\naccuracy={:0.4f}'.format(balanced_accuracy, accuracy))


def design_axes(description):
    fig = plt.figure(figsize=(24, 20))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    axes = []
    design_shape = (8, 8)

    # title
    title_ax = plt.subplot2grid(design_shape, (0, 0), rowspan=1, colspan=4)
    if description:
        title_ax.text(0.2, 0.3, description, fontsize=30)
        title_ax.set(xticks=[], yticks=[])
        title_ax.axis('off')
        title_ax.figure.set_size_inches(18, 18)
    else:
        title_ax.set_visible(False)

    axes += [plt.subplot2grid(design_shape, (1, 0))]
    axes += [plt.subplot2grid(design_shape, (1, 1))]
    axes += [plt.subplot2grid(design_shape, (1, 2))]
    axes += [plt.subplot2grid(design_shape, (1, 3))]

    axes += [plt.subplot2grid(design_shape, (2, 0))]
    axes += [plt.subplot2grid(design_shape, (2, 1))]
    axes += [plt.subplot2grid(design_shape, (2, 2))]
    axes += [plt.subplot2grid(design_shape, (2, 3))]

    axes += [plt.subplot2grid(design_shape, (3, 0))]
    axes += [plt.subplot2grid(design_shape, (3, 1))]
    axes += [plt.subplot2grid(design_shape, (3, 2))]
    axes += [plt.subplot2grid(design_shape, (3, 3))]

    axes += [plt.subplot2grid(design_shape, (3, 4))]
    axes += [plt.subplot2grid(design_shape, (3, 5))]

    # axes += [plt.subplot2grid(design_shape, (3, 2))]
    axes += [plt.subplot2grid(design_shape, (1, 4), rowspan=2, colspan=1)]
    axes += [plt.subplot2grid(design_shape, (1, 5), rowspan=2, colspan=1)]

    axes += [plt.subplot2grid(design_shape, (4, 0))]
    axes += [plt.subplot2grid(design_shape, (4, 1))]
    axes += [plt.subplot2grid(design_shape, (4, 2))]
    axes += [plt.subplot2grid(design_shape, (4, 3))]

    return design_shape, axes, fig


def get_best_bats_it(sorted_col_names, amount=3):
    best_bats_ordered_with_rep = list(map(dataset.extract_bat_name, sorted_col_names))[::-1]
    seen = set()
    best_bats_orderded_no_rep = []
    for b in best_bats_ordered_with_rep:
        if b not in seen:
            seen.add(b)
            best_bats_orderded_no_rep.append(b)
        if len(seen) == amount:
            break
    return best_bats_orderded_no_rep


def noise_cancellation(n):
    import scipy
    new_n = pd.Series(scipy.signal.medfilt(n, 3))
    return new_n


def get_df_from_file_path(file_path, net="NET1"):
    REAL_BEHAVIORAL_DATA = config.Config.get("REAL_BEHAVIORAL_DATA")
    CACHED_BEHAVIORAL_DATA = config.Config.get("CACHED_BEHAVIORAL_DATA")
    if REAL_BEHAVIORAL_DATA:
        cache_file_path1 = file_path.replace(".mat", "net1.csv")
        cache_file_path2 = file_path.replace(".mat", "net3.csv")
        cache_file_path = {"NET1": cache_file_path1, "NET3": cache_file_path2}[net]
        if CACHED_BEHAVIORAL_DATA and os.path.isfile(cache_file_path):
            print("loading cached file....")
            df = pd.read_csv(cache_file_path)
        else:
            print("parsing real data file")
            df, df2 = behavior_parse.parse_matlab_file(file_path)
            print("storing to cache...")
            df.to_csv(cache_file_path1)
            df2.to_csv(cache_file_path2)
            df = df if cache_file_path == cache_file_path1 else df2
    else:
        df = dataset.load_dataset()

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df


# TODO: remove this old code.
def manual_normalization(df):
    df_scaled = pd.DataFrame(data=minmax_scale(df), columns=df.columns, index=df.index)
    return df_scaled

    df = df_.copy()
    normalization_factors = {
        "X": 100,
        "Y": 50,
        "D": 50,
        "A": 360,
        "HD": 360,
        "XY": 70,
        "YX": 70,
        "AD": 134,
        "DA": 134
    }
    cols = df_.columns.to_list()
    bats = []
    import re

    for c in cols:
        bats.append(re.findall("BAT_(.*?)_", c)[0])

    bats = list(set(bats))
    for i in bats:
        if dataset.get_col_name(i, "X") not in df.columns:
            continue  # BAT doesn't exist

        for k in normalization_factors:
            if i == "0" and k in ["D", "A"]:
                continue
            if i != "0" and k in ["HD"]:
                continue
            if dataset.get_col_name(i, k) in df.columns:
                df[dataset.get_col_name(i, k)] /= normalization_factors[k]
                df[dataset.get_col_name(i, k + "^2")] /= normalization_factors[k] ** 2

    return df


def interleaved_indices(X, cv):
    percent = len(X) // 100
    batch_idx = (X.index % (cv * percent) // percent)
    for i in range(cv):
        train_index, test_index = X[batch_idx != i].index, X[batch_idx == i].index
        yield train_index, test_index
