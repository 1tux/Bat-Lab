import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage

import dataset
from constants import *

MODEL_SPIKE_ALPHA = 0.05


def get_net_dims(net):
    if net == "net1":
        width = NET1_WIDTH
        height = NET1_HEIGHT
    elif net == "net3":
        width = NET3_WIDTH
        height = NET3_HEIGHT
    else:
        raise Exception(f"Unknown net! {net}")

    return width, height


def get_max_ego_distance(width, height):
    max_ego_distance = int(np.sqrt(width ** 2 + height ** 2) * 0.6)
    max_ego_distance -= max_ego_distance % 3
    return max_ego_distance


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


# for place_cell
def trajectory_spike_plot(df, neuron, model_spikes, ax=None, bat_name=0, net="net1"):
    if ax is None:
        fig, ax = plt.subplots()

    width, height = get_net_dims(net)

    ax.set_xticks(np.arange(0, width + 1, 50))
    ax.set_yticks(np.arange(0, height + 1, 50))

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    prefix = "BAT"
    if len(str(bat_name)) == 2:
        prefix = "PAIR"
    ax.set_xlabel(f"{prefix} {bat_name}# X")
    ax.set_ylabel(f"{prefix} {bat_name}# Y")

    ax.set_aspect('equal', 'box')

    df_neuron = df[neuron.astype('bool')]

    bat_X = df[dataset.get_col_name(bat_name, "X")].copy()
    bat_Y = df[dataset.get_col_name(bat_name, "Y")].copy()

    spikes_X = df_neuron[dataset.get_col_name(bat_name, "X")].copy()
    spikes_Y = df_neuron[dataset.get_col_name(bat_name, "Y")].copy()

    df_model_neuron = df[model_spikes.astype('bool')]

    model_spikes_X = df_model_neuron[dataset.get_col_name(bat_name, "X")].copy()
    model_spikes_Y = df_model_neuron[dataset.get_col_name(bat_name, "Y")].copy()

    if prefix == "PAIR":
        bat_X **= 0.5
        bat_Y **= 0.5
        spikes_X **= 0.5
        spikes_Y **= 0.5
        model_spikes_X **= 0.5
        model_spikes_Y **= 0.5

    # return ax
    ax.scatter(bat_X.values, bat_Y.values, marker='.', color='#cccccc')
    ax.scatter(spikes_X.values, spikes_Y.values, marker='.', color='r')
    ax.scatter(model_spikes_X.values, model_spikes_Y.values, marker='.', color='b', alpha=MODEL_SPIKE_ALPHA)
    ax.set_title(f"#spikes: {neuron.sum()}")

    return ax


def ego_trajectory_spike_plot(df, neuron, model_spikes, bat_name, ax=None, net="net1"):
    # assert str(bat_name) in dataset.get_other_bats_names(), "Err: Ego-centric plot has to be related to other bat"
    if ax is None:
        fig, ax = plt.subplots()

    width, height = get_net_dims(net)
    ego_max_distance = get_max_ego_distance(width, height)

    ax.set_xticks(np.linspace(-ego_max_distance, ego_max_distance, 3))
    ax.set_yticks(np.linspace(-ego_max_distance, ego_max_distance, 3))

    ax.set_ylim(-ego_max_distance, ego_max_distance)
    ax.set_xlim(-ego_max_distance, ego_max_distance)

    ax.set_xlabel(f"Bat {bat_name}# -ego X")
    ax.set_ylabel(f"Bat {bat_name}# -ego Y")

    ax.set_aspect('equal', 'box')
    # df_neuron = df[neuron.astype('bool')]

    bat_D = df[dataset.get_col_name(bat_name, "D")].copy()
    bat_A = df[dataset.get_col_name(bat_name, "A")].copy()

    if len(str(bat_name)) == 2:
        # prefix = "PAIR"
        bat_D **= 0.5
        bat_A **= 0.5

    relX = bat_D * np.cos(bat_A * np.pi / 180)
    relY = bat_D * np.sin(bat_A * np.pi / 180)

    relX_spikes = relX[neuron.astype('bool')]
    relY_spikes = relY[neuron.astype('bool')]

    model_spikes_relX = relX[model_spikes.astype('bool')]
    model_spikes_relY = relY[model_spikes.astype('bool')]

    ax.scatter(relX.values, relY.values, marker='.', color='#cccccc')
    ax.scatter(relX_spikes.values, relY_spikes.values, marker='.', color='r')
    ax.scatter(model_spikes_relX.values, model_spikes_relY.values, marker='.', color='b', alpha=MODEL_SPIKE_ALPHA)
    ax.set_title(f"#spikes: {neuron.sum()}")

    return ax


def rate_map_plot(df, neuron, bat_name=0, ax=None, net="net1"):
    BIN_SIZE = 3
    width, height = get_net_dims(net)

    np.seterr(divide='ignore', invalid='ignore')
    if ax is None:
        fig, ax = plt.subplots()

    x_plot_range = np.linspace(0, width // BIN_SIZE - BIN_SIZE / 2 + 1, 3).round(1)
    y_plot_range = np.linspace(0, height // BIN_SIZE - BIN_SIZE / 2 + 1, 3).round(1)
    ax.set_xticks(x_plot_range)
    ax.set_yticks(y_plot_range)

    ax.set_xticklabels((x_plot_range * BIN_SIZE + BIN_SIZE / 2).round(1))
    ax.set_yticklabels((y_plot_range * BIN_SIZE + BIN_SIZE / 2).round(1))

    prefix = "BAT"
    if len(str(bat_name)) == 2:
        prefix = "PAIR"

    ax.set_xlabel(f"{prefix} {bat_name}# X")
    ax.set_ylabel(f"{prefix} {bat_name}# Y")

    ax.set_aspect('equal', 'box')

    bat_name = str(bat_name)

    bat_X = df[dataset.get_col_name(bat_name, "X")].copy()
    bat_Y = df[dataset.get_col_name(bat_name, "Y")].copy()

    if prefix == "PAIR":
        bat_X **= 0.5
        bat_Y **= 0.5

    time_spent = np.histogram2d(bat_X, bat_Y, [width // BIN_SIZE, height // BIN_SIZE], range=[(0, width), (0, height)])[
        0]
    time_spent = time_spent * (time_spent >= TIME_SPENT_THRESHOLD)

    spikes = np.histogram2d(bat_X, bat_Y, [width // BIN_SIZE, height // BIN_SIZE], weights=neuron,
                            range=[(0, width), (0, height)])[0]
    spikes2 = spikes * (time_spent >= TIME_SPENT_THRESHOLD)

    gauss_filter = fspecial_gauss(GAUSSIAN_FILTER_SIZE, GAUSSIAN_FILTER_SIGMA)  # divides by 3, multiply by 4

    smooth_spikes = scipy.ndimage.correlate(spikes2, gauss_filter, mode='constant')
    smooth_time_spent = scipy.ndimage.correlate(time_spent, gauss_filter, mode='constant')

    # result = spikes2 / time_spent
    smoothed_result = smooth_spikes / smooth_time_spent
    smoothed_result[time_spent < TIME_SPENT_THRESHOLD] = np.nan

    ax.set_title(f"Max firing rate: {FRAME_RATE * np.nanmax(smoothed_result):.2}")  # 25Hz
    img = ax.imshow(smoothed_result.T, cmap='jet')
    img.set_clim(0, np.nanmax(smoothed_result))

    return img


def ego_rate_map_plot(df, neuron, bat_name=0, ax=None, net="net1"):
    # assert str(bat_name) in dataset.get_other_bats_names(), "Err: Ego-centric plot has to be related to other bat"
    np.seterr(divide='ignore', invalid='ignore')
    if ax is None:
        fig, ax = plt.subplots()

    BIN_SIZE = 5
    width, height = get_net_dims(net)
    max_ego_distance = get_max_ego_distance(width, height)
    max_ego_distance2 = int(max_ego_distance / 0.6)  # unscaling

    max_linspace = np.linspace(-max_ego_distance2, max_ego_distance2, 2 * max_ego_distance2 // BIN_SIZE)

    negative_idx = np.argmin(np.abs(max_linspace + max_ego_distance))
    positive_idx = np.argmin(np.abs(max_linspace - max_ego_distance))

    ax.set_xticks(np.linspace(0, positive_idx - negative_idx, 3))
    ax.set_yticks(np.linspace(0, positive_idx - negative_idx, 3))

    ax.set_xticklabels([-max_ego_distance, 0, max_ego_distance])
    ax.set_yticklabels([-max_ego_distance, 0, max_ego_distance])

    bat_name = str(bat_name)
    bat_D = df[dataset.get_col_name(bat_name, "D")].copy()
    bat_A = df[dataset.get_col_name(bat_name, "A")].copy()

    if len(str(bat_name)) == 2:
        bat_D **= 0.5
        bat_A **= 0.5

    relX = bat_D * np.cos(bat_A * np.pi / 180)
    relY = bat_D * np.sin(bat_A * np.pi / 180)

    time_spent = np.histogram2d(relX, relY, 2 * max_ego_distance // BIN_SIZE,
                                range=[[-max_ego_distance, max_ego_distance], [-max_ego_distance, max_ego_distance]])[0]
    time_spent = time_spent * (time_spent >= TIME_SPENT_THRESHOLD)

    spikes = np.histogram2d(relX, relY, 2 * max_ego_distance // BIN_SIZE, weights=neuron,
                            range=[[-max_ego_distance, max_ego_distance], [-max_ego_distance, max_ego_distance]])[0]
    spikes2 = spikes * (time_spent >= TIME_SPENT_THRESHOLD)

    # result = spikes2 / time_spent

    gauss_filter = fspecial_gauss(GAUSSIAN_FILTER_SIZE, GAUSSIAN_FILTER_SIGMA)
    # smooth_spikes = scipy.signal.convolve2d(gauss_filter, spikes2)
    # smooth_time_spent = scipy.signal.convolve2d(gauss_filter, time_spent)

    smooth_spikes = scipy.ndimage.correlate(spikes2, gauss_filter, mode='constant')
    smooth_time_spent = scipy.ndimage.correlate(time_spent, gauss_filter, mode='constant')

    smoothed_result = smooth_spikes / smooth_time_spent
    smoothed_result[time_spent < TIME_SPENT_THRESHOLD] = np.nan

    ax.set_title(f"Max firing rate: {FRAME_RATE * np.nanmax(smoothed_result):.2}")
    img = ax.imshow(smoothed_result.T, origin="lower", cmap="jet")
    img.set_clim(0, np.nanmax(smoothed_result))

    return img


def distance_plot_1d(df, neuron):
    df_neuron = df.where(neuron.astype('bool'))
    plt.hist(df_neuron)
    plt.show()


def hd_plot_1d(df, neuron, model_spikes, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    df_neuron = df.where(neuron.astype('bool'))
    df_model_spikes = df[model_spikes.astype('bool')]

    hd_spikes_radians = df_neuron[dataset.get_col_name(0, "HD")] / 180 * np.pi
    ax.set_yticks([])
    behavior = df[dataset.get_col_name(0, "HD")] / 180 * np.pi
    behavior_map = np.histogram(behavior, bins=np.linspace(0, 1, 36) * 2*np.pi)[0]
    hd_spikes_radians_map = np.histogram(hd_spikes_radians, bins=np.linspace(0, 1, 36) * 2*np.pi)[0]
    model_spikes = df_model_spikes[dataset.get_col_name(0, "HD")] / 180 * np.pi
    model_spikes_map = np.histogram(model_spikes, bins=np.linspace(0, 1, 36) * 2*np.pi)[0]

    print(hd_spikes_radians_map)
    print(behavior_map)
    print(hd_spikes_radians_map / behavior_map)
    print(model_spikes_map / behavior_map)
    #ax.hist(behavior_map, bins=36, color="#cccccc", density=True)
    #ax.hist(hd_spikes_radians, bins=36, color="red", density=True)
    ax.plot(hd_spikes_radians_map / behavior_map)
    ax.plot(model_spikes_map / behavior_map)
    #ax.hist(df_model_spikes[dataset.get_col_name(0, "HD")] / 180 * np.pi, bins=36, color="blue", density=True, alpha=0.2)
