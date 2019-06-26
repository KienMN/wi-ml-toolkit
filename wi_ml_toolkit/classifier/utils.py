import pandas as pd

from .base import *
from .neural_network_classifier import *
#from ml_toolkit.HIMFAClassifier import *

def load(file_name):
    if os.path.isdir(file_name):
        if os.path.exists(os.path.join(file_name, 'weights')):
            return NeuralNetworkClassifier.load(file_name)
        #else:
            #return HIMFAClassifier.load(file_name)

    return Classifier.load(file_name)