import numpy as np

import dataset
from constants import *
from scipy.special import i0


def von_mises(mu, k, x):
    return np.exp(k * np.cos(x - mu)) / (2 * np.pi * i0(k))


def probabilistic_spike(neuron, spike_prob=None):
    if spike_prob is None:
        spike_prob = SPIKE_P
    np.random.seed(SEED)
    return (np.random.random(len(neuron)) * neuron > 1 - spike_prob).astype('int')


def add_noise(neuron, noise_prob=None):
    if noise_prob is None:
        noise_prob = SPIKE_NOISE_P

    noise_prob *= neuron.mean()

    return (np.random.random(len(neuron)) >= 1 - noise_prob).astype('int') | neuron


def place_cell(df, Xcenter, Ycenter, radius, bat_name=0):
    if isinstance(bat_name, list):
        n = 0
        for i in bat_name:
            n |= place_cell(df, Xcenter, Ycenter, radius, i)
        return n
    bat_name = str(bat_name)
    bat_X = df[dataset.get_col_name(bat_name, "X")]
    bat_Y = df[dataset.get_col_name(bat_name, "Y")]
    n = (((bat_X - Xcenter) ** 2 + (bat_Y - Ycenter) ** 2) <= radius ** 2).astype('int')

    return n


def gaussian_place_cell(df, Xcenter, Ycenter, sigma, bat_name=0):
    if isinstance(bat_name, list):
        n = 0
        for i in bat_name:
            n |= gaussian_place_cell(df, Xcenter, Ycenter, sigma, i)
        return n
    bat_name = str(bat_name)
    bat_X = df[dataset.get_col_name(bat_name, "X")]
    bat_Y = df[dataset.get_col_name(bat_name, "Y")]
    sigma /= 2
    gaussian_vals = 1 * np.exp(-((bat_X - Xcenter) ** 2 + (bat_Y - Ycenter) ** 2) / (2 * sigma ** 2))
    n = (gaussian_vals > np.random.random(len(gaussian_vals))).astype('int')

    return n


def ellipse_place_cell(df, Xcenter, Ycenter, a, b, bat_name=0):
    if isinstance(bat_name, list):
        n = 0
        for i in bat_name:
            n |= ellipse_place_cell(df, Xcenter, Ycenter, a, b, i)
        return n

    bat_name = str(bat_name)
    bat_X = df[dataset.get_col_name(bat_name, "X")]
    bat_Y = df[dataset.get_col_name(bat_name, "Y")]
    n = ((bat_X - Xcenter) ** 2 / a ** 2) + ((bat_Y - Ycenter) ** 2 / b ** 2) <= 1
    return n.astype('int')


def gaussian_ellipse_place_cell(df, Xcenter, Ycenter, sigma_x, sigma_y, bat_name=0):
    if isinstance(bat_name, list):
        n = 0
        for i in bat_name:
            n |= gaussian_ellipse_place_cell(df, Xcenter, Ycenter, sigma_x, sigma_y, i)
        return n

    bat_name = str(bat_name)
    bat_X = df[dataset.get_col_name(bat_name, "X")]
    bat_Y = df[dataset.get_col_name(bat_name, "Y")]

    dx_sq = (bat_X - Xcenter) ** 2
    dy_sq = (bat_Y - Ycenter) ** 2
    sigma_x /= 2
    sigma_y /= 2
    gaussian_pow = dx_sq / (2 * sigma_x ** 2) + dy_sq / (2 * sigma_y ** 2)
    gaussian_vals = 1 * np.exp(-gaussian_pow)
    n = (gaussian_vals > np.random.random(len(gaussian_vals))).astype('int')

    return n


def rectangle_place_cell(df, left_most, top_most, width, height, bat_name=0):
    if isinstance(bat_name, list):
        n = 0
        for i in bat_name:
            n |= ellipse_place_cell(df, left_most, top_most, width, height, i)
        return n

    bat_name = str(bat_name)
    bat_X = df[dataset.get_col_name(bat_name, "X")]
    bat_Y = df[dataset.get_col_name(bat_name, "Y")]
    n = ((bat_X >= left_most) & (bat_X <= left_most + width) & (bat_Y >= top_most) & (bat_Y <= top_most + height))
    return n.astype('int')


def distance_cell(df, distance, bat_name=0, distance_width=3):
    if isinstance(bat_name, list):
        n = 0
        for i in bat_name:
            n |= distance_cell(df, distance, i, distance_width)
        return n
    bat_name = str(bat_name)
    assert bat_name in dataset.get_other_bats_names(), "Err: Distance is only defined on other bats"

    bat_D = df[dataset.get_col_name(bat_name, "D")]
    n = abs(bat_D - distance) < distance_width

    return n.astype('int')


def gaussian_distance_cell(df, distance, bat_name=0, distance_width=3):
    if isinstance(bat_name, list):
        n = 0
        for i in bat_name:
            n |= gaussian_distance_cell(df, distance, i, distance_width)
        return n
    bat_name = str(bat_name)
    assert bat_name in dataset.get_other_bats_names(), "Err: Distance is only defined on other bats"

    distance_width /= 2
    bat_D = df[dataset.get_col_name(bat_name, "D")]
    gaussian_vals = 1 * np.exp(-((bat_D - distance) ** 2 / (2 * distance_width ** 2)))
    n = (gaussian_vals > np.random.random(len(gaussian_vals))).astype('int')

    return n.astype('int')


def angle_cell(df, angle, bat_name=0, angle_range=3):
    if isinstance(bat_name, list):
        n = 0
        for i in bat_name:
            n |= angle_cell(df, angle, i, angle_range)
        return n
    bat_name = str(bat_name)
    assert bat_name in dataset.get_other_bats_names(), "Err: Angle is only defined on other bats!"

    bat_A = df[dataset.get_col_name(bat_name, "A")]
    # n = (abs((bat_A - angle)%360) < angle_range) | (abs(bat_A - angle) < angle_range)
    n = ((bat_A - angle) % 360 < angle_range) | ((angle - bat_A) % 360 < angle_range)
    return n.astype('int')


def gaussian_angle_cell(df, angle, bat_name=0, angle_range=3):
    if isinstance(bat_name, list):
        n = 0
        for i in bat_name:
            n |= gaussian_angle_cell(df, angle, i, angle_range)
        return n
    bat_name = str(bat_name)
    assert bat_name in dataset.get_other_bats_names(), "Err: Angle is only defined on other bats!"

    bat_A = df[dataset.get_col_name(bat_name, "A")]
    angle_range = 1 / np.radians(3)
    gaussian_vals = von_mises(np.radians(angle), angle_range, np.radians(bat_A))
    n = (gaussian_vals > np.random.random(len(gaussian_vals))).astype('int')
    return n


def linear_ramping_distance_cell(df, bat_name):
    distance = df[dataset.get_col_name(bat_name, "D")]
    min_distance = 2 * 10
    max_distance = 7000
    distance *= distance > min_distance
    distance *= distance < max_distance
    linear_vals = (distance - 20) / 10 * 0.1
    linear_vals[linear_vals > 0.5] = 0.5
    linear_vals.hist()
    n = (linear_vals > np.random.random(len(linear_vals))).astype('int')
    return n


def head_direction_cell(df, angle, angle_range=3):
    hd = df[dataset.get_col_name(0, "HD")]
    n = ((hd - angle) % 360 < angle_range) | ((angle - hd) % 360 < angle_range)

    return n.astype('int')


def gaussian_head_direction_cell(df, angle, angle_range=3):
    hd = df[dataset.get_col_name(0, "HD")]
    gaussian_vals = von_mises(np.radians(angle), 1 / np.radians(angle_range), np.radians(hd))
    n = (gaussian_vals > np.random.random(len(gaussian_vals))).astype('int')
    return n


def neg_neuron(neuron):
    return -1 * neuron + 1
