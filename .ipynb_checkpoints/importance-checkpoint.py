import SVM_utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.pyplot as plt
import dataset

def avg_models_importance(models, data, agg=False):
    dicts = list(map(lambda x: get_feature_importances(x, data,agg=agg).to_dict(orient='list'), models))
    acc_d = {}
    for d in dicts:
        for k in d:
            if k not in acc_d:
                acc_d[k] = []
            acc_d[k].append(d[k][0])    
    return pd.DataFrame.from_dict(acc_d)

def agg_importance_by_meaning(imp, names):
    
    feature_importance_dict = dict(zip(names, imp))
    import re
    
    feature_names = list(feature_importance_dict.keys())
    bats = sorted(list(set(map(lambda x: re.findall('BAT_(.*?)_', x)[0], feature_names))))
    
    # print("agg_importance_by_meaning(), bats:", bats)
    # bat_i- position (X, Y)
    # bat_i - ego-centric position (A, D)
    ret_d = {}
    agg_map = {"position" : ["X", "Y", "X^2", "Y^2"],\
               "ego-centric" : ["A", "D", "A^2", "D^2"]}
    
    for i in bats:
        if i != '0':
            for agg_f in agg_map:
                ret_d[f"BAT_{i}_EF_{agg_f}"] = 0
                for f in agg_map[agg_f]:
                    ret_d[f"BAT_{i}_EF_{agg_f}"] += feature_importance_dict.pop(dataset.get_col_name(i, f)) # also removes item
    
    # BAT 0
    if '0' in bats:
        agg_map.pop("ego-centric")
        agg_map['head-direction'] = ["HD", "HD^2"]
        for agg_f in agg_map:
                ret_d[f"BAT_0_EF_{agg_f}"] = 0
                for f in agg_map[agg_f]:
                    ret_d[f"BAT_0_EF_{agg_f}"] += feature_importance_dict.pop(dataset.get_col_name(0, f)) # also removes item
                        
                
    # merge 2 dictions
    merged = {**ret_d, **feature_importance_dict} 
    # print(merged)
    return list(merged.values()), list(merged.keys())

def sort_f_importances(coef, names, agg=False, max_=0):
    if agg:
        coef, names = agg_importance_by_meaning(coef, names)
        
    sorted_imp,sorted_names = zip(*sorted(zip(coef,names))[-max_:])
    scale = StandardScaler()
    sorted_imp = np.array(sorted_imp)
    norm = LA.norm(sorted_imp) # np.sum(np.abs(sorted_imp)) # LA.norm(sorted_imp)
    normalized_imp = (sorted_imp / norm) ** 2
    #normalized_imp = np.exp(sorted_imp) / np.sum(np.exp(sorted_imp))
    return normalized_imp, sorted_names

def plot_f_importances(imp, names, ax=None):
    ax.barh(range(len(names)), imp, align='center')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, horizontalalignment = "left")

def sparsity_measure(table, total_features):
    weights = [1/total_features] * len(table)
    feature_names = table.index.to_list()

    for i in range(len(feature_names)):
        name = feature_names[i]
        if name.endswith("position"):
            weights[i] *= 4
        if name.endswith("ego-centric"):
            weights[i] *= 4
        if name.endswith("head-direction"):
            weights[i] *= 2

    numerator = ((weights * table.values)).sum() ** 2
    denominator = 0
    for i in range(len(table.values)):
        denominator +=  (weights[i] * table.values[i] ** 2)
        
    return (numerator / denominator)
    
    
def plot_f_importance_from_table(table, ax, amount, total_features, shuffled_table):
    orig_table = table.copy()
    orig_shuffled_table = shuffled_table.copy()
    ordered_columns = table.mean(axis=0).sort_values().index.to_list()[-amount:][::-1]
    table = table[ordered_columns]
    shuffled_table = shuffled_table[ordered_columns]
    # ax.tick_params(axis="y", pad=-20)
    ax.set_yticks(range(len(ordered_columns)))
    coeff_of_variations = (table.std().values / table.mean().values).round(2)
    shuffled_cov = (shuffled_table.std().values / shuffled_table.mean().values).round(2)
    ordered_columns_with_cov = [f"{i[0]}\n    {i[2]}\n{i[1]}" for i in zip(coeff_of_variations, shuffled_cov, ordered_columns)]
    ax.set_yticklabels(ordered_columns_with_cov, horizontalalignment = "left")
    ax.set_title(f"Sparsity: {sparsity_measure(orig_table.mean(0), total_features):.2} / {sparsity_measure(orig_shuffled_table.mean(0), total_features):.2}")
    ax.barh(table.columns.to_list(), table.mean().values, align='center', color="#cccccc") # xerr=table.std().values
    
    # return table.mean().values[-1]
    
def get_feature_importances(s, df,agg=False, max_=0):
    #imp, names = sort_f_importances(abs(s.coef_[0]), df.columns.values, agg=agg, max_=max_)
    imp, names = sort_f_importances(s.get_importances(), df.columns.values, agg=agg, max_=max_)
    res = pd.DataFrame([imp.tolist()], columns=names)
    return res