import dataset
import numpy as np
import re
import config

def add_squared_features_old(df, other_bats, pair_bats_names=[]):
    for f in ["X", "Y", "HD"]:
        df[dataset.get_col_name(0, f"{f}^2")] = df[dataset.get_col_name(0, f)] ** 2
        # df[dataset.get_col_name(0, f"{f}^0.5")] = df[dataset.get_col_name(0, f)] ** 0.5

    for bat_name in other_bats:
        for f_name in ["X", "Y", "A", "D"]:
            df[dataset.get_col_name(bat_name, f_name + "^2")] = df[dataset.get_col_name(bat_name, f_name)] ** 2
            # df[dataset.get_col_name(bat_name, f_name + "^0.5")] = df[dataset.get_col_name(bat_name, f_name)] ** 0.5

    for bat_name in pair_bats_names:
        for f_name in ["X", "Y", "A", "D", "D*"]:
            df[dataset.get_col_name(bat_name, f_name + "^2", "PAIR")] = df[dataset.get_col_name(bat_name, f_name,
                                                                                                "PAIR")] ** 2
            # df[dataset.get_col_name(bat_name, f_name + "^0.5", "PAIR")] =
            # df[dataset.get_col_name(bat_name, f_name, "PAIR")] ** 0.5

    return df


def add_squared_features(df, other_bats):
    for f in ["X", "Y", "HD"]:
        df[dataset.get_col_name(0, f"{f}^2")] = df[dataset.get_col_name(0, f)] ** 2
        # df[dataset.get_col_name(0, f"{f}^0.5")] = df[dataset.get_col_name(0, f)] ** 0.5

    for bat_name in other_bats:
        for f_name in ["X", "Y", "A", "D"]:
            df[dataset.get_col_name(bat_name, f_name + "^2")] = df[dataset.get_col_name(bat_name, f_name)] ** 2
            # df[dataset.get_col_name(bat_name, f_name + "^0.5")] = df[dataset.get_col_name(bat_name, f_name)] ** 0.5

    return df


def add_pairwise_distance(df, other_bats):
    for bat_name1 in other_bats:
        for bat_name2 in other_bats:
            if bat_name1 <= bat_name2 or bat_name1 == "0" or bat_name2 == "0":
                continue
            pair_name = bat_name1 + bat_name2
            x1 = df[dataset.get_col_name(bat_name1, "X")]
            x2 = df[dataset.get_col_name(bat_name2, "X")]
            y1 = df[dataset.get_col_name(bat_name1, "Y")]
            y2 = df[dataset.get_col_name(bat_name2, "Y")]
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            df[dataset.get_col_name(pair_name, "Dp", "PAIR")] = d
            # df[dataset.get_col_name(pair_name, "Dp^2", "PAIR")] = d ** 2

    return df


def add_pairwise_features(df, other_bats):
    """ takes BAT_1 features name.
    # to support OR/AND neurons:
        # (x_1)^2 * (x_2^2)
        # (x_1)^2 * (x_2)
        # (x_1) * (x_2)^2
        # (x_1) * (x_2)
    # to support AND neurons:
        # (x_1-x_2)^2 + (y_2-y_1)^2 (pairwise distance)
    """
    f_names = map(lambda x: re.findall("BAT_[1-9]_F_(.*?)$", x), df.columns)
    f_names_flat = list(set([item for sublist in f_names for item in sublist]))
    pair_bats_names = []
    for bat_name1 in other_bats:
        for bat_name2 in other_bats:
            if bat_name1 <= bat_name2:
                continue  # or bat_name1 == "0" or bat_name2 == "0": continue

            pair_name = bat_name1 + bat_name2
            pair_bats_names.append(pair_name)

            for f_name_ in f_names_flat:
                for f_name in [f_name_, f_name_ + "^2"]:
                    v1 = df[dataset.get_col_name(bat_name1, f_name)]
                    v2 = df[dataset.get_col_name(bat_name2, f_name)]
                    df[dataset.get_col_name(pair_name, f_name, "PAIR")] = v1 * v2

    return df, pair_bats_names


def add_pairwise_rotational_features(df, other_bats):
    pair_bats_name = []
    for bat_name1 in other_bats:
        for bat_name2 in other_bats:
            if bat_name1 <= bat_name2:
                continue
            pair_name = bat_name1 + bat_name2
            pair_bats_name.append(pair_name)

            for f_name1 in ["X", "Y"]:
                for f_name2 in ["X", "Y"]:
                    if f_name1 == f_name2:
                        continue
                    df[dataset.get_col_name(pair_name, f_name1 + f_name2, "PAIR")] = np.sqrt(
                        df[dataset.get_col_name(bat_name1, f_name1)] * df[dataset.get_col_name(bat_name2, f_name2)])

            for f_name1 in ["D", "A"]:
                for f_name2 in ["D", "A"]:
                    if f_name1 == f_name2:
                        continue
                    df[dataset.get_col_name(pair_name, f_name1 + f_name2, "PAIR")] = np.sqrt(
                        df[dataset.get_col_name(bat_name1, f_name1)] * df[dataset.get_col_name(bat_name2, f_name2)])
    return df


def add_nearest_distance(df):
    df[dataset.get_col_name('0', 'nD', 'BAT')] = df[df.columns[df.columns.str.endswith('_D')]].dropna().min(axis=1)
    df[dataset.get_col_name('0', 'nD^2', 'BAT')] = df[dataset.get_col_name('0', 'nD', 'BAT')]
    return df


def get_bets_in_df(df):
    return list(set(df.columns.str.extract(r'BAT_(\d)')[0].to_list()))


def engineer(df):
    bats = get_bets_in_df(df)
    other_bats = [x for x in bats if x != '0']

    # pairwise_distance
    df = add_nearest_distance(df)
    df = add_squared_features(df, other_bats)
    if config.Config.get("WITH_PAIRS"):
        df = add_pairwise_distance(df, other_bats)
        df, pair_bats_names = add_pairwise_features(df, other_bats)
    # df = add_squared_features(df, pair_bats_names)
    return df
