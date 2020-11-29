import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
import pandas as pd
from PIL import ImageFilter

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

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()    
    
# for place_cell
def trajectory_spike_plot(df, neuron, model_spikes, ax = None, bat_name = 0, net="net1"):
    if ax is None:
        fig, ax = plt.subplots()
        
    width, height = get_net_dims(net)
        
    ax.set_xticks(np.arange(0, width+1, 50)) 
    ax.set_yticks(np.arange(0, height+1, 50)) 
    
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    
    ax.set_xlabel(f"Bat {bat_name}# X")
    ax.set_ylabel(f"Bat {bat_name}# Y")
    
    ax.set_aspect('equal','box')

    df_neuron = df[neuron.astype('bool')]


    bat_X = df[dataset.get_col_name(bat_name, "X")]
    bat_Y = df[dataset.get_col_name(bat_name, "Y")]
    
    spikes_X = df_neuron[dataset.get_col_name(bat_name, "X")]
    spikes_Y = df_neuron[dataset.get_col_name(bat_name, "Y")]

    df_model_neuron = df[model_spikes.astype('bool')]

    model_spikes_X = df_model_neuron[dataset.get_col_name(bat_name, "X")]
    model_spikes_Y = df_model_neuron[dataset.get_col_name(bat_name, "Y")]
    
    # return ax
    ax.plot(bat_X.values, bat_Y.values, color='#cccccc')
    ax.plot(spikes_X.values, spikes_Y.values, '.', color='r')
    ax.plot(model_spikes_X.values, model_spikes_Y.values, '.', color='b', alpha=MODEL_SPIKE_ALPHA)
    ax.set_title(f"#spikes: {neuron.sum()}")
    
    return ax

def ego_trajectory_spike_plot(df, neuron, model_spikes, bat_name, ax = None, net="net1"):
    assert str(bat_name) in dataset.get_other_bats_names(), "Err: Ego-centric plot has to be related to other bat"
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
    
    ax.set_aspect('equal','box')
    # df_neuron = df[neuron.astype('bool')]
    
    relX = df[dataset.get_col_name(bat_name, "D")] * np.cos(df[dataset.get_col_name(bat_name, "A")] * np.pi / 180)
    relY = df[dataset.get_col_name(bat_name, "D")] * np.sin(df[dataset.get_col_name(bat_name, "A")] * np.pi / 180)
    
    relX_spikes = relX[neuron.astype('bool')]
    relY_spikes = relY[neuron.astype('bool')]
    
    model_spikes_relX = relX[model_spikes.astype('bool')]
    model_spikes_relY = relY[model_spikes.astype('bool')]
    
    ax.plot(relX.values, relY.values, color='#cccccc')
    ax.plot(relX_spikes.values, relY_spikes.values, '.', color='r')
    ax.plot(model_spikes_relX.values, model_spikes_relY.values, '.', color='b', alpha=MODEL_SPIKE_ALPHA)
    ax.set_title(f"#spikes: {neuron.sum()}")
    
    return ax

def rate_map_plot(df, neuron, bat_name=0, ax=None, net="net1"):
    BIN_SIZE = 3
    width, height = get_net_dims(net)
    
    np.seterr(divide='ignore', invalid='ignore')
    if ax is None:
        fig, ax = plt.subplots()
     
    x_plot_range = np.linspace(0, width // BIN_SIZE-BIN_SIZE/2+1, 3).round(1)
    y_plot_range = np.linspace(0, height // BIN_SIZE-BIN_SIZE/2+1, 3).round(1)
    ax.set_xticks(x_plot_range) 
    ax.set_yticks(y_plot_range) 
    
    ax.set_xticklabels((x_plot_range * BIN_SIZE + BIN_SIZE / 2).round(1))
    ax.set_yticklabels((y_plot_range * BIN_SIZE + BIN_SIZE / 2).round(1))
    
    ax.set_xlabel(f"Bat {bat_name}# X")
    ax.set_ylabel(f"Bat {bat_name}# Y")
    
    ax.set_aspect('equal','box')
        
    bat_name = str(bat_name)
    
    bat_X = df[dataset.get_col_name(bat_name, "X")]
    bat_Y = df[dataset.get_col_name(bat_name, "Y")]
    
    time_spent = np.histogram2d(bat_X, bat_Y,  [width // BIN_SIZE, height // BIN_SIZE], range=[(0, width), (0, height)])[0]
    time_spent = time_spent * (time_spent >= TIME_SPENT_THRESHOLD)
    
    spikes = np.histogram2d(bat_X, bat_Y, [width // BIN_SIZE, height // BIN_SIZE], weights=neuron, range=[(0, width), (0, height)])[0]
    spikes2 = spikes * (time_spent >= TIME_SPENT_THRESHOLD)

    gauss_filter = fspecial_gauss(GAUSSIAN_FILTER_SIZE, GAUSSIAN_FILTER_SIGMA) # divides by 3, multiply by 4

    smooth_spikes = scipy.ndimage.correlate(spikes2, gauss_filter, mode='constant')
    smooth_time_spent = scipy.ndimage.correlate(time_spent, gauss_filter, mode='constant')
    
    result = spikes2 / time_spent
    smoothed_result = smooth_spikes / smooth_time_spent
    smoothed_result[time_spent < TIME_SPENT_THRESHOLD] = np.nan
    
    ax.set_title(f"Max firing rate: {FRAME_RATE*np.nanmax(smoothed_result):.2}") # 25Hz
    img = ax.imshow(smoothed_result.T, cmap='jet')
    img.set_clim(0, np.nanmax(smoothed_result))
    
    return img

def ego_rate_map_plot(df, neuron, bat_name=0, ax=None, net="net1"):
    assert str(bat_name) in dataset.get_other_bats_names(), "Err: Ego-centric plot has to be related to other bat"
    np.seterr(divide='ignore', invalid='ignore')
    if ax is None:
        fig, ax = plt.subplots()
    
    BIN_SIZE = 5
    width, height = get_net_dims(net)
    max_ego_distance = get_max_ego_distance(width, height)
    max_ego_distance2 = int(max_ego_distance / 0.6) # unscaling
    
    max_linspace = np.linspace(-max_ego_distance2, max_ego_distance2, 2 * max_ego_distance2 // BIN_SIZE)
    
    negative_idx = np.argmin(np.abs(max_linspace + max_ego_distance))
    positive_idx = np.argmin(np.abs(max_linspace - max_ego_distance))
    
    ax.set_xticks(np.linspace(0, positive_idx - negative_idx, 3)) 
    ax.set_yticks(np.linspace(0, positive_idx - negative_idx, 3)) 
    
    ax.set_xticklabels([-max_ego_distance, 0, max_ego_distance])
    ax.set_yticklabels([-max_ego_distance, 0, max_ego_distance])

    bat_name = str(bat_name)
    
    relX = df[dataset.get_col_name(bat_name, "D")] * np.cos(df[dataset.get_col_name(bat_name, "A")] * np.pi / 180)
    relY = df[dataset.get_col_name(bat_name, "D")] * np.sin(df[dataset.get_col_name(bat_name, "A")] * np.pi / 180)
    
    time_spent = np.histogram2d(relX, relY,  2 * max_ego_distance // BIN_SIZE, range=[[-max_ego_distance, max_ego_distance], [-max_ego_distance, max_ego_distance]])[0]
    time_spent = time_spent * (time_spent >= TIME_SPENT_THRESHOLD)
    
    spikes = np.histogram2d(relX, relY,  2 * max_ego_distance // BIN_SIZE, weights=neuron, range=[[-max_ego_distance, max_ego_distance], [-max_ego_distance, max_ego_distance]])[0]
    spikes2 = spikes * (time_spent >= TIME_SPENT_THRESHOLD)
    
    result = spikes2 / time_spent

    gauss_filter = fspecial_gauss(GAUSSIAN_FILTER_SIZE, GAUSSIAN_FILTER_SIGMA)
    smooth_spikes = scipy.signal.convolve2d(gauss_filter, spikes2)
    smooth_time_spent = scipy.signal.convolve2d(gauss_filter, time_spent)

    smooth_spikes = scipy.ndimage.correlate(spikes2, gauss_filter, mode='constant')
    smooth_time_spent = scipy.ndimage.correlate(time_spent, gauss_filter, mode='constant')
    
    smoothed_result = smooth_spikes / smooth_time_spent
    smoothed_result[time_spent < TIME_SPENT_THRESHOLD] = np.nan
    
    ax.set_title(f"Max firing rate: {FRAME_RATE*np.nanmax(smoothed_result):.2}")
    img = ax.imshow(smoothed_result.T, origin="lower", cmap="jet")
    img.set_clim(0, np.nanmax(smoothed_result))
    
    return img

def distance_plot_1d(df, neuron, bat_id=""):
    df_neuron = df.where(neuron.astype('bool'))
    plt.hist(df_neuron)
    plt.show()
    
    
def hd_plot_1d(df, neuron, model_spikes, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    df_neuron = df.where(neuron.astype('bool'))
    df_model_spikes = df[model_spikes.astype('bool')]
    
    hd_spikes_radians = df_neuron[dataset.get_col_name(0, "HD")] / 180 * np.pi
    ax.set_yticks([])
    ax.hist(df[dataset.get_col_name(0, "HD")] / 180 * np.pi, bins=36, color="#cccccc", density=True)
    ax.hist(hd_spikes_radians, bins=36, color="red", density=True)
    ax.hist(df_model_spikes[dataset.get_col_name(0, "HD")] / 180 * np.pi, bins=36, color="blue", density=True)