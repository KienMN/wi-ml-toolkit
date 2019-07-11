# Importing the libraries
import os
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier

from .helper import *
confusion_matrix = cm_with_percentage

class Classifier:

    # def __init__(self, val_size, feature_cols, label_col, feature_degree, 
    #              feature_scaling, including_classes, preprocess, shuffle):
    def __init__(self, **params):

        self.val_size = params.pop('val_size')
        self.feature_degree = params.pop('feature_degree')
        self.feature_scaling = params.pop('feature_scaling')
        self.including_classes = params.pop('including_classes')
        self.add_cluster_features = params.pop('add_cluster_features')
        self.shuffle = params.pop('shuffle') and self.add_cluster_features
        
        self.model_params = params

        self.his = {'acc': None, 'loss': None, 'val_acc': None, 'val_loss': None}
        self.model = None
        self.cm = None
        self.score = 0

    def preprocess_data(self, X, y):
        y = y.astype('int')

        self.le = LabelEncoder()
        y = self.le.fit_transform(y)

        self.num_labels = max(y) + 1
        self.labels = [i for i in range(self.num_labels)]
        self.labels_origin = self.le.inverse_transform(self.labels).tolist()

        X_train, X_val, y_train, y_val = split_data(X, y, self.val_size, self.shuffle)

        if self.add_cluster_features:
            X_train, y_train = add_cluster_features(X_train, y_train)
            X_val, y_val = add_cluster_features(X_val, y_val)
        
        self.poly = PolynomialFeatures(self.feature_degree, include_bias=False)
        X_train = self.poly.fit_transform(X_train)
        if len(X_val) > 0:
            X_val = self.poly.transform(X_val)

        self.num_samples, self.num_features = X_train.shape
        
        self.sc = StandardScaler()
        if self.feature_scaling:
            X_train = self.sc.fit_transform(X_train)
            if len(X_val) > 0:
                X_val = self.sc.transform(X_val)

        return X_train, X_val, y_train, y_val

    def preprocess_X(self, X):
        
        if self.add_cluster_features:
            X = add_features(X)

        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)

        return X

    def evaluate_helper(self, X, y, radius, verbose):
        prob = self.model.predict_proba(X)

        try:
            loss = log_loss(y, prob)
        except:
            loss = -1

        pred = self.model.predict(X=X)
        accuracy = np.count_nonzero(y == pred) / len(y)

        if verbose:
            print('Accuracy: ', accuracy * 100, ' Loss: ', loss)

        pred = smoothen(pred, radius)
        accuracy = np.count_nonzero(y == pred) / len(y)
        if verbose and radius > 0:
            print('Accuracy after smoothening with radius =', radius, ': ', accuracy * 100)

        self.cm = confusion_matrix(y, pred, labels=self.labels)

        return accuracy, loss

    def evaluate_test(self, radius=0, verbose=False):

        if len(self.X_val) > 0:
            if verbose:
                print('\nEvaluating on test set...')
            X = self.X_val
            y = self.y_val

            accuracy, loss = self.evaluate_helper(X, y, radius, verbose)

            self.score = accuracy * 100

    def evaluate(self, X, y, radius=0, verbose=False):

        X, y = filt_data(X, y, self.including_classes)

        X = self.preprocess_X(X)
        y = self.le.transform(y)

        accuracy, loss = self.evaluate_helper(X, y, radius, verbose)

        return {'acc': accuracy, 'loss': loss}

    def judge(self, X, y, radius=0, verbose=False, threshold=0.8):

        X, y = filt_data(X, y, self.including_classes)

        X = self.preprocess_X(X)
        y = self.le.transform(y)

        prob = self.model.predict_proba(X)
        pred = prob.argmax(axis=1)
        confidence = np.max(prob, axis=1)

        pred = judge(pred, prob, threshold=threshold)

        pred = smoothen(pred, radius)

        cm = confusion_matrix(y, pred, labels=self.labels+[-9999])

        return cm

    def probability(self, X):

        X = self.preprocess_X(X)

        return self.model.predict_proba(X)

    def predict(self, X=None, radius=0, threshold=0.0):

        X = self.preprocess_X(X)

        prob = self.model.predict_proba(X)
        pred = prob.argmax(axis=1)
        confidence = np.max(prob, axis=1)

        pred = self.le.inverse_transform(pred)
        pred = judge(pred, confidence, threshold=threshold, null_type=None)
        pred = smoothen(pred, radius)

        return pred

    def get_result(self, X, radius=0, threshold=0.0):

        X = self.preprocess_X(X)

        prob = self.model.predict_proba(X)
        pred = prob.argmax(axis=1)
        confidence = np.max(prob, axis=1)

        pred = self.le.inverse_transform(pred)
        pred = judge(pred, confidence, threshold=threshold, null_type=None)
        pred = smoothen(pred, radius)

        cum_prob = cumulate_prob(prob)

        return dict(target=pred.tolist(), prob=cum_prob.tolist())

    def get_cm_data_url(self, id):
        if self.cm is None:
            return None

        draw_confusion_matrix(self.cm, self.labels_origin + [''])
        img = id + '.png'
        plt.savefig(img)
        data_url = image_to_data_url(img)
        os.remove(img)

        return data_url

    @classmethod
    def load(Classifier, file_name):
        return joblib.load(file_name)