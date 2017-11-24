import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import src.csv_parser as CsvParser
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn.metrics import mean_squared_error
from keras.activations import relu, elu
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.losses import mean_absolute_error
from keras.models import Model
import keras.backend as K


# See these 2 links about passing custom variables to a Keras layer
# https://www.tensorflow.org/api_docs/python/tf/keras/backend/variable
# https://keras.io/initializers/
class RegularModel:
    def __init__(self, csv_file: str = '../data/corrected_dates.csv',
                 index_of_plotted_feature: int = 0):
        df = pd.read_csv(csv_file)  # By default header will be read from file
        num_of_rows = df.shape[0]
        self.num_features = 4
        self.num_hidden_layer_neurons = 1024
        self.index_of_plotted_feature = index_of_plotted_feature
        print('Head of data frame: \n' + str(df.head()))
        print('Dimensions of data frame (row x col)' + str(df.shape))
        self.training_set_size = int(0.8 * num_of_rows)
        self.test_set_size = int(num_of_rows - self.training_set_size)
        print('Training set size: ' + str(self.training_set_size))
        print('Test set size: ' + str(self.test_set_size))
        df_train, df_test = df[self.test_set_size:], df[:self.test_set_size]
        print('Dimensions of training data frame (row x col)' + str(df_train.shape))
        print('Dimensions of test data frame (row x col)' + str(df_test.shape))
        self.x_tr = df_train[['open', 'high', 'low', 'close']]
        self.x_te = df_test[['open', 'high', 'low', 'close']]
        self.y_tr = df_train[['open', 'high', 'low', 'close']]
        self.y_te = df_test[['open', 'high', 'low', 'close']]

        # Shift arrays so that we predict prices for time t given (t-1)
        self.x_tr = self.x_tr.values[1:, :]
        self.x_te = self.x_te.values[1:, :]
        self.y_tr = self.y_tr.values[:-1, :]
        self.y_te = self.y_te.values[:-1, :]
        print('y_te: ' + str(self.y_te))
        # Create Keras NN model to get H
        self.input_shape = (self.num_features,)
        self.hidden_layer_name = 'hidden_layer'
        self.input_layer_name = 'input_layer'
        self.model = Sequential()
        # self.model.add(Dense(self.num_features, input_shape=self.input_shape, name=self.input_layer_name))
        self.model.add(
            Dense(self.num_hidden_layer_neurons, activation='tanh', name=self.hidden_layer_name, input_shape=self.input_shape))
        self.model.add(Dense(self.num_features))
        self.model.compile('adam', 'mse', metrics=['mse'])
        print('Regular model summary:')
        self.model.summary()

    def train(self):
        # Num of epochs in num of iterations where it takes batch_size number elements from X and y
        self.model.fit(self.x_tr, self.y_tr, epochs=10, batch_size=32)

    def predict(self):
        last_day_in_data = np.array([self.y_te[0]])
        print('Prediction for last day in data with values:' + str(last_day_in_data) + ' with shape: '
              + str(last_day_in_data.shape))
        print(str(self.model.predict(last_day_in_data)))
        training_pred = self.model.predict(self.x_tr)
        test_pred = self.model.predict(self.x_te)
        return training_pred, test_pred
