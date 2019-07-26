import numpy as np

from scipy.optimize import curve_fit
from sympy.parsing.latex import parse_latex;
from sympy import symbols, sin, cos, latex
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr

from functools import partial
from collections import defaultdict

def getModules():
    return {
        'log' : np.log,
        'e' : np.e,
        'sin' : np.sin,
        'cos' : np.cos,
        'tan' : np.tan,
        'sinh' : np.sinh,
        'cosh' : np.cosh,
        'tanh' : np.tanh
    }

class NonLinearRegressor():
    def __init__(self, string_function, variables, parameters):
        variable_list = [symbols(variable) for variable in variables]
        parameter_list = [symbols(parameter) for parameter in parameters]
        print(parameter_list)
        if 'e' in string_function:
            parameter_list.append(symbols('e'))
        self.lambda_function = sympy2lambda(variable_list, parameter_list, string_function)
        self.popt = None
        self.pcov = None
    
    def fit(self, X, y):
        xdata = np.array(X, dtype="float64")
        xdata = xdata.transpose()
        ydata = np.array(y, dtype="float64")
        self.popt, self.pcov = curve_fit(self.lambda_function, xdata, ydata)
        return self

    def predict(self, X):
        X = np.array(X)
        X = X.transpose()
        y_pred = self.lambda_function(X, *self.popt)
        return y_pred

def sympy2latex(sympyStringFunction=""):
    sympyFunction = parse_expr(sympyStringFunction)
    latexFunction =  latex(sympyFunction)
    return latexFunction

def sympy2lambda(variables, parameters, sympyStringFunction, modules=getModules()):
    sympyFunction = parse_expr(sympyStringFunction)
    symbolList = [variables] + parameters
    lambdaFunction = lambdify(symbolList, sympyFunction, modules)
    if symbols('e') in symbolList: 
        lambdaFunction = partial(lambdaFunction, e = np.e)

    return lambdaFunction


def numpy_log(x, base):
    return np.log(x)/np.log(base)

