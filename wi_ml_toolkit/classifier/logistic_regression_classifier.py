# Kernel SVM

# Importing the libraries
from sklearn.linear_model import LogisticRegression

from .base import *

class LogisticRegressionClassifier(Classifier):

    default_preprocess_params = dict(
        val_size = 0.0,
        feature_degree = 1,
        feature_scaling = True,
        including_classes = None,
        add_cluster_features = True,
        shuffle = False,
        random_state = None,
    )

    default_model_params = dict(
        c=20, 
        max_iter=10000, 
        solver='liblinear',
    )

    def __init__(self, **params):
        default_params = {**self.default_preprocess_params, 
                          **self.default_model_params}
        default_params.update(params)

        super().__init__(**default_params)

        self.model_params['C'] = self.model_params.pop('c')

        estimator = LogisticRegression(**self.model_params, verbose=0,
                                       random_state=self.random_state)
        self.model = OneVsRestClassifier(estimator=estimator)

    def fit(self, X, y, verbose=False):
        self.X_train, self.X_val, self.y_train, self.y_val = self.preprocess_data(X, y)

        if verbose:
            print('Using Logistic Regression Classfier...')

        self.model.fit(self.X_train, self.y_train)

        if verbose:
            print('\n--- Training result ---')
            accuracy, loss = self.evaluate_helper(self.X_train, self.y_train, 0, verbose)

        # self.his['acc'] = [accuracy]
        # self.his['loss'] = [loss]

        # Predicting the Test set results
        self.evaluate_test(verbose=verbose)

    def save(self, file_name=None):
        del self.X_train
        del self.y_train
        del self.X_val
        del self.y_val

        if file_name is None:
            file_name = 'lr_model_' + str(int(round(self.score*100,1)))
        joblib.dump(self, file_name)