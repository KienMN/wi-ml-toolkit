from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as  plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from wi_ml_toolkit.regressor import MLPRegressor

boston = load_boston()

random_state = 17

x_train, x_test, y_train, y_test = \
  train_test_split(boston.data, boston.target, test_size=0.2, random_state=random_state)

steps = [
  ('normal', MinMaxScaler()),
  ('estimator', MLPRegressor(hidden_layer_sizes=(100),
                            solver='ncg',
                            activation='relu',
                            verbose=True,
                            max_iter=200,
                            random_state=random_state))
]

pipe = Pipeline(steps)
pipe.fit(x_train, y_train)
train = pipe.named_steps['estimator'].lpath['train']
val = pipe.named_steps['estimator'].lpath['val']

if random_state is not None:
  pipe_clone = Pipeline(steps)
  pipe_clone.fit(x_train, y_train)
  train_clone = pipe_clone.named_steps['estimator'].lpath['train']
  val_clone = pipe_clone.named_steps['estimator'].lpath['val']
  print('Reproducable:', train == train_clone, val == val_clone)

plt.plot(list(range(len(val))), val, label='Val')
plt.plot(list(range(len(train))), train, label='Train')
plt.legend()
plt.show()