import abc
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit, KFold, StratifiedKFold
import importance
from sklearn.metrics import confusion_matrix
import logging as log
import numpy as np
from constants import *
import SVM_utils
#from xgboost import XGBClassifier
import copy

class Model():
    
    def __init__(self, model = None):
        pass
    
    def train(self, X_train, y_train):
        pass
    
    def predict(X_test):
        pass
    
    def get_importances():
        pass

    
    def single_run(self, X, y, test_size = 0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1337)
        X_train, y_train = SVM_utils.upsample(X_train, y_train)
        self.train(X_train, y_train)
    
        self.imp_table = importance.get_feature_importances(self, X, agg=False, max_=0)
        self.agg_imp_table = importance.get_feature_importances(self, X, agg=True, max_=0)
    
        self.train_cm = confusion_matrix(y_train, self.model.predict(X_train))
        if test_size: self.test_cm = confusion_matrix(y_test, self.model.predict(X_test))
        else: self.test_cm = self.train_cm * np.nan
    
        self.std_train_cm = self.train_cm * np.nan
        self.std_test_cm = self.test_cm * np.nan
    
        return self, [self.train_cm], [self.test_cm], self.imp_table, self.agg_imp_table, self.std_train_cm, self.std_test_cm

    def per_cross_validation(self, model, X_train, X_test, y_train, y_test):
        X_train, y_train = SVM_utils.upsample(X_train, y_train)
        model.train(X_train, y_train)
        
        model.train_cm = confusion_matrix(y_train, model.predict(X_train))
        model.test_cm = confusion_matrix(y_test, model.predict(X_test))
    
        return model, model.train_cm, model.test_cm
    
    def cross_validate(self, X, y):
        svm_models = []
        train_cms = []
        test_cms = []

        #interleaved_indices(df, cv): #KFold(n_splits=cv).split(X):  ## StratifiedKFold
        threads = []
        for train_index, test_index in KFold(n_splits=self.cv, shuffle=True, random_state=1337).split(X, y): # StratifiedKFold
            model = copy.deepcopy(self)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if self.multi_threaded:
                t = ThreadWithReturnValue(target = self.per_cross_validation, args=(model, X_train, X_test, y_train, y_test))
                threads.append(t)
                t.start()
            else:
                svm_model, train_cm, test_cm = self.per_cross_validation(model, X_train, X_test, y_train, y_test)
                svm_models.append(svm_model)
                train_cms.append(train_cm)
                test_cms.append(test_cm)

        if self.multi_threaded:
            for t in threads:
                svm_model, train_cm, test_cm = t.join()

                svm_models.append(svm_model)
                train_cms.append(train_cm)
                test_cms.append(test_cm)

        log.info("finished CV, agg results..")
        imp_table = importance.avg_models_importance(svm_models, X, agg=False)
        log.info("finished CV, agg results2..")
        agg_imp_table = importance.avg_models_importance(svm_models, X, agg=True)

        log.info("finished CV, AVG Confusion Matrices..")

        avg_train_cm = np.mean(train_cms, axis=(0))
        avg_test_cm = np.mean(test_cms, axis=(0))

        std_train_cm = np.std(train_cms, axis=(0))
        std_test_cm = np.std(test_cms, axis=(0))

        log.info("finished CV, Done AVG Confusion Matrices!")

        return svm_models[0], train_cms, test_cms, imp_table, agg_imp_table, std_train_cm, std_test_cm
    
class SVM_model(Model):
    
    def __init__(self, model = None, cv = False, multi_threaded = True):
        self.model = model
        self.cv = (cv > 1) * cv
        self.multi_threaded = multi_threaded
        if model is None: self.model = SGDClassifier(power_t = 0.4, n_jobs=1, max_iter=10**5,  learning_rate="invscaling", eta0=1, penalty = 'L1', n_iter_no_change=10, random_state=1337)
        
            
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
        if self.cv: return self.cross_validate(X, y)
        else: return self.single_run(X, y)

    def set_weight(self, weight):
        self.model.class_weight = weight
        
class NN_model(Model):
    pass

class XGBoost_model(Model):
    def __init__(self, model = None, cv = False, multi_threaded = True):
        self.model = model
        self.cv = cv
        self.multi_threaded = multi_threaded
        if model is None: self.model = XGBClassifier(n_estimators=100, max_depth=5, n_jobs=8) 
        # self.model = XGBClassifier(booster="gblinear", n_estimators=100, max_depth=5, n_jobs=8)
            
    def train_(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)
        
    def train(self, X, y):
        from sklearn.utils.class_weight import compute_sample_weight
        import pandas as pd
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y, pd.Series(y).replace(0, self.model.class_weight[0]).replace(1, self.model.class_weight[1]))
        
    def predict(self, X_test):
        self.X_test = X_test
        self.X_preds = self.model.predict(X_test)
        return self.X_preds
        
    def get_importances(self):
        # print(self.model.coef_)
        #return np.abs(self.model.coef_)
    
        importances = self.model.get_booster().get_score(importance_type = 'gain')
        for column_name in self.X_train.columns:
            if column_name not in importances:
                importances[column_name] = 0
                
        importances_ordered_by_columns = []
        
        for column_name in self.X_train.columns:
            importances_ordered_by_columns.append(importances[column_name])
        
        importances_ordered_by_columns = np.array(importances_ordered_by_columns)
        importances_ordered_by_columns /= importances_ordered_by_columns.sum()
        
        #importances_ordered_by_columns -= importances_ordered_by_columns.mean()
        #importances_ordered_by_columns /= importances_ordered_by_columns.std()
        importances_ordered_by_columns = np.abs(importances_ordered_by_columns)

        """ import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 16))

        ax.barh(list(self.X_train.columns), self.model.feature_importances_, alpha=0.5)
        ax.barh(list(self.X_train.columns), importances_ordered_by_columns, alpha=0.5)
        print(importances_ordered_by_columns)
        print(self.model.feature_importances_)
        plt.show()
        """
        return np.abs(self.model.feature_importances_)
    
    def __call__(self, X, y):
        if self.cv > 1: return self.cross_validate(X, y)
        else: return self.single_run(X, y)

    def set_weight(self, weight):
        self.model.class_weight = weight


class XGBoost_model_(Model):
    def __init__(self, model = None, cv = False, multi_threaded = True):
        self.model = model
        self.cv = cv
        self.multi_threaded = multi_threaded
        if model is None: self.model = XGBClassifier(booster="gblinear", n_estimators=100, max_depth=5, n_jobs=8) 
        # self.model = XGBClassifier(booster="gblinear", n_estimators=100, max_depth=5, n_jobs=8)
            
    def train(self, X, y):
        from sklearn.utils.class_weight import compute_sample_weight
        import pandas as pd
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y, pd.Series(y).replace(0, self.model.class_weight[0]).replace(1, self.model.class_weight[1]))
        
    def predict(self, X_test):
        self.X_test = X_test
        self.X_preds = self.model.predict(X_test)
        return self.X_preds
        
    def get_importances(self):
        # print(self.model.coef_)
        return np.abs(self.model.coef_)
    
        importances = self.model.get_booster().get_score(importance_type = 'gain')
        for column_name in self.X_train.columns:
            if column_name not in importances:
                importances[column_name] = 0
                
        importances_ordered_by_columns = []
        
        for column_name in self.X_train.columns:
            importances_ordered_by_columns.append(importances[column_name])
        
        importances_ordered_by_columns = np.array(importances_ordered_by_columns)
        importances_ordered_by_columns /= importances_ordered_by_columns.sum()
        
        #importances_ordered_by_columns -= importances_ordered_by_columns.mean()
        #importances_ordered_by_columns /= importances_ordered_by_columns.std()
        importances_ordered_by_columns = np.abs(importances_ordered_by_columns)

        """ import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 16))

        ax.barh(list(self.X_train.columns), self.model.feature_importances_, alpha=0.5)
        ax.barh(list(self.X_train.columns), importances_ordered_by_columns, alpha=0.5)
        print(importances_ordered_by_columns)
        print(self.model.feature_importances_)
        plt.show()
        """
        return np.abs(self.model.feature_importances_)
    
    def __call__(self, X, y):
        if self.cv > 1: return self.cross_validate(X, y)
        else: return self.single_run(X, y)

    def set_weight(self, weight):
        self.model.class_weight = weight