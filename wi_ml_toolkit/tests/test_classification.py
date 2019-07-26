import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from wi_ml_toolkit.classifier.decision_tree_classifier import DecisionTreeClassifier
from wi_ml_toolkit.classifier.random_forest_classifier import RandomForestClassifier
from wi_ml_toolkit.classifier.k_nearest_neighbors_classifier import KNearestNeighborsClassifier
from wi_ml_toolkit.classifier.logistic_regression_classifier import LogisticRegressionClassifier
from wi_ml_toolkit.classifier.neural_network_classifier import NeuralNetworkClassifier

Classifiers = [DecisionTreeClassifier, RandomForestClassifier,
               KNearestNeighborsClassifier, LogisticRegressionClassifier]

random_state = 17

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for Model in Classifiers:
    model = Model(random_state=random_state)
    print(model.model_params)
    model.fit(X, y)
    print('#' * 50, model.__class__.__name__)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))

    if random_state is not None:
        model_clone = Model(random_state=random_state)
        print(model_clone.model_params)
        model_clone.fit(X, y)
        pred_clone = model_clone.predict(X_test)
        print('Reproducable:', (pred == pred_clone).all())

model = NeuralNetworkClassifier(algorithm='backprop')
print(model.model_params)
model.fit(X, y)
print('#' * 50, model.__class__.__name__, '(backprop)')
pred = model.predict(X_test)
print(classification_report(y_test, pred))
if random_state is not None:
    model_clone = model = NeuralNetworkClassifier(algorithm='backprop', 
                                                  random_state=random_state)
    print(model_clone.model_params)
    model_clone.fit(X, y)
    pred_clone = model_clone.predict(X_test)
    print('Reproducable:', (pred == pred_clone).all())

model = NeuralNetworkClassifier(algorithm='evolution')
print(model.model_params)
model.fit(X, y)
print('#' * 50, model.__class__.__name__, '(evolution)')
pred = model.predict(X_test)
print(classification_report(y_test, pred))
if random_state is not None:
    model_clone = model = NeuralNetworkClassifier(algorithm='evolution', 
                                                  random_state=random_state)
    print(model_clone.model_params)
    model_clone.fit(X, y)
    pred_clone = model_clone.predict(X_test)
    print('Reproducable:', (pred == pred_clone).all())