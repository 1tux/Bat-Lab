from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold, StratifiedKFold
import importance
from sklearn.metrics import confusion_matrix
import logging as log
import numpy as np
from constants import *
import SVM_utils
# from xgboost import XGBClassifier
import copy
from math import sqrt
from sklearn.svm import SVC
import pandas as pd

def per_cross_validation(model, X_train, X_test, y_train, y_test):
    if model.upsample:
        X_train, y_train = SVM_utils.upsample(X_train, y_train)
    model.train(X_train, y_train)

    model.train_cm = confusion_matrix(y_train, model.predict(X_train))
    print("predicting:", model.predict(X_test).sum(), "out of", y_test.sum(), "spikes")
    model.test_cm = confusion_matrix(y_test, model.predict(X_test))

    return model, model.train_cm, model.test_cm


def evaluate(confusion_matrix):
    # calculating MCC score: https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    tp = confusion_matrix[1][1]
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]

    mcc = (tp * tn - fp*fn) / sqrt((tp + fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return mcc

def evaluate_against_shuffles(model_cms, shuffles_cms):
    # assuming we got several confusion_matrices from running the model with CV.
    # assuming we got several confusion_matrices from shuffles.
    # we return the difference between the medians of the MCCs of the confusion_matrices.

    models_mcc = []
    for cm in models_cms:
        models_mcc.append(evaluate(cm))

    shuffles_mcc = []
    for cm in shuffles_cms:
        shuffles_mcc.append(evaluate(cm))

    mcc1 = np.median(models_mcc)
    mcc2 = np.median(shuffles_mcc)

    return mcc1 - mcc2

class Model:
    def __init__(self, model=None, cv=False, multi_threaded=True, upsample=True):
        self.model = model
        self.cv = (cv > 1) * cv
        self.multi_threaded = multi_threaded
        self.upsample = upsample

        self.imp_table = None
        self.agg_imp_table = None
        self.train_cm = None
        self.test_cm = None
        self.std_train_cm = None
        self.std_test_cm = None
        self.svm_models = None
        pass

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def get_importances(self):
        pass

    def single_run(self, X, y, test_size=0.2):
        #print(X.shape)
        #print(len(y))
        #X = X.sample(30000, random_state=1337)
        #y = pd.Series(y)[X.index].to_numpy()

        #print(X.shape)
        #print(len(y))
        #print("CAT") 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1337)
        if self.upsample:
            X_train, y_train = SVM_utils.upsample(X_train, y_train)

        #X_train = X_train.sample(10000, random_state=1337)
        #y_train = pd.Series(y_train)[X_train.index].to_numpy()
        #print("CAT")

        self.train(X_train, y_train)

        self.imp_table = importance.get_feature_importances(self, X, agg=False, max_=0)
        self.agg_imp_table = self.imp_table # self.agg_imp_table = importance.get_feature_importances(self, X, agg=True, max_=0)

        self.train_cm = confusion_matrix(y_train, self.model.predict(X_train))
        if test_size:
            self.test_cm = confusion_matrix(y_test, self.model.predict(X_test))
        else:
            self.test_cm = self.train_cm * np.nan

        self.std_train_cm = self.train_cm * np.nan
        self.std_test_cm = self.test_cm * np.nan

        return self, [self.train_cm], [self.test_cm], self.imp_table, self.agg_imp_table, self.std_train_cm, self.std_test_cm

    def cross_validate(self, X, y):
        svm_models = []
        train_cms = []
        test_cms = []

        # interleaved_indices(df, cv): #KFold(n_splits=cv).split(X):  ## StratifiedKFold
        threads = []
        if not self.upsample:
            f = lambda: StratifiedKFold(n_splits=self.cv).split(X, y)
        else:
            f = lambda: KFold(n_splits=self.cv, shuffle=bool(self.upsample), random_state=1337).split(X, y)

        for train_index, test_index in f():
            model = copy.deepcopy(self)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if self.multi_threaded:
                t = ThreadWithReturnValue(target=per_cross_validation,
                                          args=(model, X_train, X_test, y_train, y_test))
                threads.append(t)
                t.start()
            else:
                svm_model, train_cm, test_cm = per_cross_validation(model, X_train, X_test, y_train, y_test)
                svm_models.append(svm_model)
                train_cms.append(train_cm)
                test_cms.append(test_cm)

        if self.multi_threaded:
            for t in threads:
                svm_model, train_cm, test_cm = t.join()

                svm_models.append(svm_model)
                train_cms.append(train_cm)
                test_cms.append(test_cm)

        # self.svm_models = svm_models
        # TODO: change the code so we don't need this line
        svm_models[0].svm_models = svm_models

        log.info("finished CV, agg results..")
        imp_table = importance.avg_models_importance(svm_models, X, agg=False)
        log.info("finished CV, agg results2..")
        agg_imp_table = importance.avg_models_importance(svm_models, X, agg=True)

        log.info("finished CV, AVG Confusion Matrices..")

        # avg_train_cm = np.mean(train_cms, axis=(0))
        # avg_test_cm = np.mean(test_cms, axis=(0))

        std_train_cm = np.std(train_cms, axis=0)
        std_test_cm = np.std(test_cms, axis=0)

        log.info("finished CV, Done AVG Confusion Matrices!")

        # TODO: return self instead, or actually not return anything at all!
        return svm_models[0], train_cms, test_cms, imp_table, agg_imp_table, std_train_cm, std_test_cm

class SVMModel(Model):

    def __init__(self, model=None, cv=False, multi_threaded=True, upsample=True):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.X_preds = None
        self.svm_models = None

        self.upsample = upsample
        self.model = model
        self.cv = (cv > 1) * cv
        self.multi_threaded = multi_threaded
        if model is None:
            self.model = SGDClassifier(power_t=0.4, n_jobs=1, max_iter=10 ** 5, learning_rate="invscaling", eta0=1,
                                       penalty='L1', n_iter_no_change=10, random_state=1337)

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)

    def predict(self, X_test):
        self.X_test = X_test
        self.X_preds = self.model.predict(X_test)
        return self.X_preds

    def get_importances(self):
        return np.abs(self.model.coef_[0])

    def __call__(self, X, y):
        if self.cv:
            return self.cross_validate(X, y)
        else:
            return self.single_run(X, y)

    def set_weight(self, weight):
        self.model.class_weight = weight

    def get_all_importances(self):
        if self.svm_models is None: # not running CV but running the model 1 time
            return [self.model.coef_[0]]

        coeffs = []
        for m in self.svm_models:
            coeffs.append(m.model.coef_[0])
        return coeffs

class SoftMAXModel(Model):

    def __init__(self, model=None, cv=False, multi_threaded=True, upsample=True):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.X_preds = None
        self.svm_models = None

        self.upsample = upsample
        self.model = model
        self.cv = (cv > 1) * cv
        self.multi_threaded = multi_threaded
        if model is None:
            #self.model = LogisticRegression(n_jobs=1, penalty='l1', random_state=1337, max_iter=10 ** 3, solver='saga')
            self.model = SGDClassifier(loss='log', power_t=0.4, n_jobs=1, max_iter=10 ** 5, learning_rate="invscaling", eta0=1,
                                       penalty='L1', n_iter_no_change=10, random_state=1337)

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)

    def predict(self, X_test):
        self.X_test = X_test
        self.X_preds = self.model.predict(X_test)
        return self.X_preds

    def get_importances(self):
        return np.abs(self.model.coef_[0])

    def __call__(self, X, y):
        if self.cv:
            return self.cross_validate(X, y)
        else:
            return self.single_run(X, y)

    def set_weight(self, weight):
        self.model.class_weight = weight

    def get_all_importances(self):
        if self.svm_models is None: # not running CV but running the model 1 time
            return [self.model.coef_[0]]

        coeffs = []
        for m in self.svm_models:
            coeffs.append(m.model.coef_[0])
        return coeffs

class RBFkernel(Model):

    def __init__(self, model=None, cv=False, multi_threaded=True, upsample=True):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.X_preds = None
        self.svm_models = None
        
        self.upsample = upsample
        self.model = model
        self.cv = (cv > 1) * cv
        self.multi_threaded = multi_threaded
        if model is None:
            #self.model = LogisticRegression(n_jobs=1, penalty='l1', random_state=1337, max_iter=10 ** 3, solver='saga')
            self.model = SVC(random_state=1337)

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)

    def predict(self, X_test):
        self.X_test = X_test
        self.X_preds = self.model.predict(X_test)
        return self.X_preds

    def get_importances(self):
        return [0] * self.X_train.shape[1]

    def __call__(self, X, y):
        if self.cv:
            return self.cross_validate(X, y)
        else:
            return self.single_run(X, y)

    def set_weight(self, weight):
        self.model.class_weight = weight

    def get_all_importances(self):
        if self.svm_models is None: # not running CV but running the model 1 time
            return [[0] * self.X_train.shape[1]]

        coeffs = []
        for m in self.svm_models:
            coeffs.append(m.model.coef_[0])
        return coeffs

class NN_model(Model):
    pass


# class XGBoost_model(Model):
#     def __init__(self, model=None, cv=False, multi_threaded=True):
#         self.model = model
#         self.cv = cv
#         self.multi_threaded = multi_threaded
#         if model is None:
#             self.model = XGBClassifier(n_estimators=100, max_depth=5, n_jobs=8)
#         # self.model = XGBClassifier(booster="gblinear", n_estimators=100, max_depth=5, n_jobs=8)
#
#     def train_(self, X, y):
#         self.X_train = X
#         self.y_train = y
#         self.model.fit(X, y)
#
#     def train(self, X, y):
#         from sklearn.utils.class_weight import compute_sample_weight
#         import pandas as pd
#         self.X_train = X
#         self.y_train = y
#       self.model.fit(X, y, pd.Series(y).replace(0, self.model.class_weight[0]).replace(1, self.model.class_weight[1]))
#
#     def predict(self, X_test):
#         self.X_test = X_test
#         self.X_preds = self.model.predict(X_test)
#         return self.X_preds
#
#     def get_importances(self):
#         # print(self.model.coef_)
#         # return np.abs(self.model.coef_)
#
#         importances = self.model.get_booster().get_score(importance_type='gain')
#         for column_name in self.X_train.columns:
#             if column_name not in importances:
#                 importances[column_name] = 0
#
#         importances_ordered_by_columns = []
#
#         for column_name in self.X_train.columns:
#             importances_ordered_by_columns.append(importances[column_name])
#
#         importances_ordered_by_columns = np.array(importances_ordered_by_columns)
#         importances_ordered_by_columns /= importances_ordered_by_columns.sum()
#
#         # importances_ordered_by_columns -= importances_ordered_by_columns.mean()
#         # importances_ordered_by_columns /= importances_ordered_by_columns.std()
#         importances_ordered_by_columns = np.abs(importances_ordered_by_columns)
#
#         """ import matplotlib.pyplot as plt
#         fig, ax = plt.subplots(figsize=(16, 16))
#
#         ax.barh(list(self.X_train.columns), self.model.feature_importances_, alpha=0.5)
#         ax.barh(list(self.X_train.columns), importances_ordered_by_columns, alpha=0.5)
#         print(importances_ordered_by_columns)
#         print(self.model.feature_importances_)
#         plt.show()
#         """
#         return np.abs(self.model.feature_importances_)
#
#     def __call__(self, X, y):
#         if self.cv > 1:
#             return self.cross_validate(X, y)
#         else:
#             return self.single_run(X, y)
#
#     def set_weight(self, weight):
#         self.model.class_weight = weight

# class XGBoost_model_(Model):
#     def __init__(self, model=None, cv=False, multi_threaded=True):
#         self.model = model
#         self.cv = cv
#         self.multi_threaded = multi_threaded
#         if model is None: self.model = XGBClassifier(booster="gblinear", n_estimators=100, max_depth=5, n_jobs=8)
#         # self.model = XGBClassifier(booster="gblinear", n_estimators=100, max_depth=5, n_jobs=8)
#
#     def train(self, X, y):
#         import pandas as pd
#         self.X_train = X
#         self.y_train = y
#       self.model.fit(X, y, pd.Series(y).replace(0, self.model.class_weight[0]).replace(1, self.model.class_weight[1]))
#
#     def predict(self, X_test):
#         self.X_test = X_test
#         self.X_preds = self.model.predict(X_test)
#         return self.X_preds
#
#     def get_importances(self):
#         # print(self.model.coef_)
#         return np.abs(self.model.coef_)
#
#         importances = self.model.get_booster().get_score(importance_type='gain')
#         for column_name in self.X_train.columns:
#             if column_name not in importances:
#                 importances[column_name] = 0
#
#         importances_ordered_by_columns = []
#
#         for column_name in self.X_train.columns:
#             importances_ordered_by_columns.append(importances[column_name])
#
#         importances_ordered_by_columns = np.array(importances_ordered_by_columns)
#         importances_ordered_by_columns /= importances_ordered_by_columns.sum()
#
#         # importances_ordered_by_columns -= importances_ordered_by_columns.mean()
#         # importances_ordered_by_columns /= importances_ordered_by_columns.std()
#         importances_ordered_by_columns = np.abs(importances_ordered_by_columns)
#
#         # import matplotlib.pyplot as plt
#         # fig, ax = plt.subplots(figsize=(16, 16))
#
#         # ax.barh(list(self.X_train.columns), self.model.feature_importances_, alpha=0.5)
#         # ax.barh(list(self.X_train.columns), importances_ordered_by_columns, alpha=0.5)
#         # print(importances_ordered_by_columns)
#         # print(self.model.feature_importances_)
#         # plt.show()
#
#         return np.abs(self.model.feature_importances_)
#
#     def __call__(self, X, y):
#         if self.cv > 1:
#             return self.cross_validate(X, y)
#         else:
#             return self.single_run(X, y)
#
#     def set_weight(self, weight):
#         self.model.class_weight = weight
