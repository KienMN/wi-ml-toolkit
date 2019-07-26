# K-Nearest Neighbors (K-NN)

# Importing the libraries
from sklearn.neighbors import KNeighborsClassifier as Model

from .base import *

class KNearestNeighborsClassifier(Classifier):

    default_preprocess_params = dict(
        val_size = 0.0,
        feature_degree = 1,
        feature_scaling = True,
        including_classes = None,
        add_cluster_features = False,
        shuffle = False,
        random_state = None,
    )
    
    default_model_params = dict(
        num_neighbors=100, 
        p=1,
        metric='minkowski',
    )

    def __init__(self, **params):
        default_params = {**self.default_preprocess_params, 
                          **self.default_model_params}
        default_params.update(params)

        super().__init__(**default_params)

        self.model_params['n_neighbors'] = self.model_params.pop('num_neighbors')

        self.model = Model(**self.model_params)

    def fit(self, X, y, verbose=False):
        self.X_train, self.X_val, self.y_train, self.y_val = self.preprocess_data(X, y)

        if verbose:
            print('Using K-Nearest Neighbors Classifier...')

        self.model.fit(self.X_train, self.y_train)

        if verbose:
            print('\n--- Training result ---')
            accuracy, loss = self.evaluate_helper(self.X_train, self.y_train, 0, verbose)
        #
        # self.his['acc'] = [accuracy]
        # self.his['loss'] = [loss]

        self.evaluate_test(verbose=verbose)

    def save(self, file_name=None):
        del self.X_train
        del self.y_train
        del self.X_val
        del self.y_val

        if file_name is None:
            file_name = 'knn_model_' + str(int(round(self.score*100,1)))
        joblib.dump(self, file_name)