from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as  plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from wi_ml_toolkit.regressor import NonLinearRegressor

boston = load_boston()
x_train, x_test, y_train, y_test = \
  train_test_split(boston.data[:, :2], boston.target, test_size=0.2)

model = NonLinearRegressor(string_function='a*log(x)+b*y+c', variables=['x', 'y'], parameters=['a', 'b', 'c'])

model.fit(x_train, y_train)
print(model.predict(x_test))