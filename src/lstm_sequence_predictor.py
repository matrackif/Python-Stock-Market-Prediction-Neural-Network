# Inspired by https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from matplotlib import dates
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import to_datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
import utils as utils


class LSTMSequencePredictor:
    def __init__(self, csv_file: str = '../data/daily_MSFT.csv', index_of_plotted_feature: int = 0,
                 num_of_previous_days: int = 7, num_of_future_days: int = 3, num_of_hidden_neurons: int = 256):
        self.dataset = read_csv(csv_file, header=0)
        print('CSV columns: ' + str(self.dataset.columns.tolist()))
        values = self.dataset[['open', 'high', 'low', 'close', 'volume']].values
        values = values.astype('float32')
        self.num_of_prev_timesteps = num_of_previous_days
        self.num_of_future_timesteps = num_of_future_days
        self.num_features = 5
        self.num_prev_objs = self.num_features * self.num_of_prev_timesteps
        self.num_future_objs = self.num_features * self.num_of_future_timesteps
        # open = 0, high = 1, low = 2, close = 3, volume = 4
        self.index_of_plotted_feature = index_of_plotted_feature  # The feature that shall be plotted (all will be predicted)
        self.plotted_feature_str = \
            {0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}[self.index_of_plotted_feature]
        print('Model will plot: ' + self.plotted_feature_str)
        self.num_rolling_days_ahead = 30

        self.unscaled_values = values.copy()
        # normalize features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        print('Shape of values before transforming: ' + str(values.shape))
        self.scaled = self.scaler.fit_transform(values)

        # frame as supervised learning
        self.reframed = utils.series_to_supervised(self.scaled, self.num_of_prev_timesteps,
                                                   self.num_of_future_timesteps)
        reframed_unscaled = utils.series_to_supervised(self.unscaled_values, self.num_of_prev_timesteps,
                                                       self.num_of_future_timesteps)
        self.unscaled_values = reframed_unscaled.values
        self.plotable_dates = utils.convert_to_matplot_dates(self.dataset)
        self.plotable_dates = np.reshape(self.plotable_dates, newshape=(-1, 1)).copy()
        self.reframed_dates = utils.series_to_supervised(self.plotable_dates, self.num_of_prev_timesteps,
                                                         self.num_of_future_timesteps)
        print('self.reframed_dates: \n' + str(self.reframed_dates.head()))
        print('reframed: \n' + str(self.reframed.head()))
        self.scaled_values = self.reframed.values  # Extract numpy array from a pandas DataFrame

        self.plotable_y_train_real = None
        self.plotable_y_train_pred = None
        self.plotable_y_test_real = None
        self.plotable_y_test_pred = None
        self.plotable_y_future_pred = None

        print('Dim of scaled_values: ' + str(self.scaled_values.shape))
        self.last_observations = self.scaled_values[0, -self.num_prev_objs:]
        print('last_observations: \n' + str(self.last_observations))

        self.data_size = self.reframed.shape[0]
        print('Data size: ' + str(self.data_size))
        # Training set is 80% of the examples
        self.training_set_size = int(0.8 * self.data_size)
        self.test_set_size = self.data_size - self.training_set_size

        self.train_dates = self.reframed_dates.values[self.test_set_size:, -1:].flatten()
        self.test_dates = self.reframed_dates.values[:self.test_set_size, -1:].flatten()

        # split into input and outputs
        # Training set contains the older (time-wise) part of data
        self.training_set_x_y = self.scaled_values[self.test_set_size:, :]
        # Test set has the newer (time-wise) part of data
        self.test_set_x_y = self.scaled_values[:self.test_set_size, :]
        print(self.dataset.head())

        self.unscaled_train_x_y = self.unscaled_values[self.test_set_size:, :]
        self.unscaled_test_x_y = self.unscaled_values[:self.test_set_size, :]
        unscaled_train_y, unscaled_test_y = self.unscaled_train_x_y[:, -self.num_future_objs:], self.unscaled_test_x_y[
                                                                                                :,
                                                                                                -self.num_future_objs:]

        self.train_x, self.train_y = self.training_set_x_y[:, :self.num_prev_objs], self.training_set_x_y[:,
                                                                                    -self.num_future_objs:]
        self.test_x, self.test_y = self.test_set_x_y[:, :self.num_prev_objs], self.test_set_x_y[:,
                                                                              -self.num_future_objs:]

        self.plotable_y_train_real = unscaled_train_y[:, -(self.num_features + self.index_of_plotted_feature)].flatten()
        self.plotable_y_test_real = unscaled_test_y[:, -(self.num_features + self.index_of_plotted_feature)].flatten()

        # reshape input to be 3D [samples, timesteps, features] as expected by LSTM
        self.train_x = self.train_x.reshape(self.train_x.shape[0], self.num_of_prev_timesteps, self.num_features)
        self.test_x = self.test_x.reshape(self.test_x.shape[0], self.num_of_prev_timesteps, self.num_features)
        self.last_observations_reshaped = self.last_observations.reshape(1, self.num_of_prev_timesteps,
                                                                         self.num_features)

        print('Training input size: ' + str(self.train_x.shape) + '\n'
              + 'Training output size: ' + str(self.train_y.shape) + '\n'
              + 'Test input size: ' + str(self.test_x.shape) + '\n'
              + 'Test output size: ' + str(self.test_y.shape))
        print('test_set_x_y: \n' + str(self.test_set_x_y))

        # Create LSTM model
        self.model = Sequential()
        self.model.add(LSTM(num_of_hidden_neurons, input_shape=(self.train_x.shape[1], self.train_x.shape[2])))
        self.model.add(Dense(self.num_future_objs))
        self.model.compile(loss='mae', optimizer='adam')
        self.model.summary()

    def train(self, plot_history: bool = True):
        # fit network
        history = self.model.fit(self.train_x, self.train_y, epochs=15,
                                 batch_size=64, validation_data=(self.test_x, self.test_y),
                                 verbose=2, shuffle=False)
        # plot history
        if plot_history:
            pyplot.figure(0)
            pyplot.plot(history.history['loss'], label='Training loss (error)')
            pyplot.plot(history.history['val_loss'], label='Test loss (error)')
            pyplot.legend()
            pyplot.show()

    def invert(self, input_y):
        input_y = input_y[:, -self.num_features:]
        input_y = input_y.reshape((len(input_y), self.num_features))
        # invert scaling and add missing columns to get back actual value
        inv_y = self.scaler.inverse_transform(input_y)
        # Because our data is in decreasing (time-wise) order we have to reverse the result
        # inv_y = inv_y[::-1]
        return inv_y

    def make_pred_and_invert(self, input_x, is_predicting_real_future: bool = False):
        y_pred = self.model.predict(input_x)

        # We do not wish to extract last columns if we are predicting real feature
        if is_predicting_real_future:
            y_pred = y_pred.reshape(self.num_of_future_timesteps, self.num_features)
        else:
            y_pred = y_pred[:, -self.num_features:]

        inv_pred_y = self.scaler.inverse_transform(y_pred)
        # Because our data is in decreasing (time-wise) order we have to reverse the result
        # inv_pred_y = inv_pred_y[::-1]
        return inv_pred_y

    def predict_all_and_plot(self, do_plot: bool = True):
        # Make prediction for future given last N observations
        inv_y_train = self.invert(input_y=self.train_y)
        inv_y_test = self.invert(input_y=self.test_y)
        inv_y_pred_train = self.make_pred_and_invert(input_x=self.train_x)
        inv_y_pred_test = self.make_pred_and_invert(input_x=self.test_x)
        inv_y_future_pred = self.make_pred_and_invert(input_x=self.last_observations_reshaped,
                                                      is_predicting_real_future=True)
        print('inv_y_future_pred: \n' + str(inv_y_future_pred) + ' shape: ' + str(inv_y_future_pred.shape))
        self.plotable_y_train_pred = inv_y_pred_train[:,
                                     -(self.num_features + self.index_of_plotted_feature)].flatten()
        self.plotable_y_test_pred = inv_y_pred_test[:, -(self.num_features + self.index_of_plotted_feature)].flatten()
        # When predicting the future we have num_future_timesteps amount of future days
        self.plotable_y_future_pred = inv_y_future_pred[:,
                                      self.index_of_plotted_feature::self.num_features].flatten()
        if do_plot:
            # Plot real training data vs. the prediction of training data
            plt.figure(0)
            real_train_graph, = plt.plot_date(self.train_dates, self.plotable_y_train_real, 'b-',
                                              label='Real training data', color='red')
            pred_train_graph, = plt.plot_date(self.train_dates, self.plotable_y_train_pred, 'b-',
                                              label='Prediction of training data', color='blue')
            plt.xlabel('Date')
            plt.ylabel('Prediction of ' + self.plotted_feature_str)
            plt.legend(handles=[real_train_graph, pred_train_graph])
            plt.show()

            # Plot real training data vs. the prediction of training data
            plt.figure(1)
            real_test_graph, = plt.plot_date(self.test_dates, self.plotable_y_test_real, 'b-',
                                             label='Real test data', color='red')
            pred_test_graph, = plt.plot_date(self.test_dates, self.plotable_y_test_pred, 'b-',
                                             label='Prediction of test data', color='blue')
            plt.xlabel('Date')
            plt.ylabel('Prediction of ' + self.plotted_feature_str)
            plt.legend(handles=[real_test_graph, pred_test_graph])
            plt.show()

            # Plot the future prediction
            plt.figure(2)
            future_graph, = plt.plot(range(1, len(self.plotable_y_future_pred) + 1), self.plotable_y_future_pred, 'b-',
                                     label='Future prediction', color='red')
            plt.xlabel('Number of days since last day in data')
            plt.ylabel('Prediction of ' + self.plotted_feature_str)
            plt.legend(handles=[future_graph])
            plt.show()

        print('Found prediction for training_pred, shape is: ' + str(inv_y_pred_train.shape),
              ' and type: ' + str(type(inv_y_pred_train)))
        print('Found prediction for test_pred, shape is: ' + str(inv_y_pred_test.shape),
              ' and type: ' + str(type(inv_y_pred_test)))
        print('Found prediction for future time frame, shape is: ' + str(inv_y_future_pred.shape),
              ' and type: ' + str(type(inv_y_future_pred)))
        print('Found prediction for future time frame, values are: ' + str(
            self.plotable_y_future_pred),
              ' and len: ' + str(len(self.plotable_y_future_pred)))
        return inv_y_train, inv_y_test, inv_y_pred_train, inv_y_pred_test, inv_y_future_pred


if __name__ == '__main__':
    lstm = LSTMSequencePredictor()
    lstm.train()
    lstm.predict_all_and_plot()
