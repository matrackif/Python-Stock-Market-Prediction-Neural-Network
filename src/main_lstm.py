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


# See this link for more details on this function:
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
        In other words n_in is how many days before we shall predict, n_out is how many days ahead
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(-i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def convert_to_matplot_dates(dates_dataframe):
    datetimes = to_datetime(dates_dataframe.values)
    list_datetimes = datetimes.to_pydatetime().tolist()
    return dates.date2num(list_datetimes)


class LSTMPredictor:
    def __init__(self, csv_file: str='../data/lstm_dates.csv'):
        self.dataset = read_csv(csv_file, header=0, index_col=0)
        values = self.dataset.values
        values = values.astype('float32')
        self.num_of_prev_timesteps = 7
        self.num_of_future_timesteps = 3
        self.num_features = 5
        self.num_prev_objs = self.num_features * self.num_of_prev_timesteps
        self.num_future_objs = self.num_features * self.num_of_future_timesteps
        # open = 0, high = 1, low = 2, close = 3, volume = 4
        self.index_of_plotted_feature = 0  # The feature that shall be plotted (all will be predicted)
        self.num_rolling_days_ahead = 30

        # normalize features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled = self.scaler.fit_transform(values)

        # frame as supervised learning
        self.reframed = series_to_supervised(self.scaled, self.num_of_prev_timesteps, self.num_of_future_timesteps)

        print('reframed: \n' + str(self.reframed.head()))
        self.scaled_values = self.reframed.values  # Extract numpy array from a pandas DataFrame

        print('Dim of scaled_values: ' + str(self.scaled_values.shape))
        self.last_observations = self.scaled_values[:self.num_of_future_timesteps, -self.num_prev_objs:]
        print('last_observations: \n' + str(self.last_observations))

        self.data_size = self.reframed.shape[0]
        print('Data size: ' + str(self.data_size))
        # Training set is 80% of the examples
        self.training_set_size = int(0.8 * self.data_size)
        self.test_set_size = self.data_size - self.training_set_size
        # split into input and outputs
        # Training set contains the older (time-wise) part of data
        self.training_set_x_y = self.scaled_values[self.test_set_size:, :]
        # Test set has the newer (time-wise) part of data
        self.test_set_x_y = self.scaled_values[:self.test_set_size, :]
        print(self.dataset.head())

        self.train_x, self.train_y = self.training_set_x_y[:, :self.num_prev_objs], self.training_set_x_y[:, -self.num_features:]
        self.test_x, self.test_y = self.test_set_x_y[:, :self.num_prev_objs], self.test_set_x_y[:, -self.num_features:]
        # reshape input to be 3D [samples, timesteps, features] as expected by LSTM
        self.train_x = self.train_x.reshape(self.train_x.shape[0], self.num_of_prev_timesteps, self.num_features)
        self.test_x = self.test_x.reshape(self.test_x.shape[0], self.num_of_prev_timesteps, self.num_features)
        self.last_observations_reshaped = self.last_observations.reshape(self.last_observations.shape[0], self.num_of_prev_timesteps,
                                                                         self.num_features)

        print('Training input size: ' + str(self.train_x.shape) + '\n'
              + 'Training output size: ' + str(self.train_y.shape) + '\n'
              + 'Test input size: ' + str(self.test_x.shape) + '\n'
              + 'Test output size: ' + str(self.test_y.shape))
        print('test_set_x_y: \n' + str(self.test_set_x_y))

        # Create LSTM model
        self.model = Sequential()
        self.model.add(LSTM(512, input_shape=(self.train_x.shape[1], self.train_x.shape[2])))
        self.model.add(Dense(self.num_features))
        self.model.compile(loss='mae', optimizer='adam')
        self.model.summary()

    def train(self):
        # fit network
        history = self.model.fit(self.train_x, self.train_y, epochs=5,
                                 batch_size=64, validation_data=(self.test_x, self.test_y),
                                 verbose=2, shuffle=False)
        # plot history
        pyplot.figure(0)
        pyplot.plot(history.history['loss'], label='Training loss (error)')
        pyplot.plot(history.history['val_loss'], label='Test loss (error)')
        pyplot.legend()
        pyplot.show()

    def invert(self, input_y):
        input_y = input_y.reshape((len(input_y), self.num_features))
        # invert scaling and add missing columns to get back actual value
        inv_y = self.scaler.inverse_transform(input_y)
        # Because our data is in decreasing (time-wise) order we have to reverse the result
        inv_y = inv_y[::-1]
        return inv_y

    def make_pred_and_invert(self, input_x):
        y_pred = self.model.predict(input_x)
        # print('y_pred: \n' + str(y_pred))
        inv_pred_y = self.scaler.inverse_transform(y_pred)
        # Because our data is in decreasing (time-wise) order we have to reverse the result
        inv_pred_y = inv_pred_y[::-1]
        return inv_pred_y

    def predict_all(self):
        # Make prediction for future given last N observations
        inv_y_train = self.invert(input_y=self.train_y)
        inv_y_test = self.invert(input_y=self.test_y)
        inv_y_pred_train = self.make_pred_and_invert(input_x=self.train_x)
        inv_y_pred_test = self.make_pred_and_invert(input_x=self.test_x)
        inv_y_future_pred = self.make_pred_and_invert(input_x=self.last_observations_reshaped)
        inv_y_rolling_pred = self.make_rolling_pred_and_invert(input_x=self.last_observations)
        return inv_y_train, inv_y_test, inv_y_pred_train, inv_y_pred_test, inv_y_future_pred, inv_y_rolling_pred

    def make_rolling_pred_and_invert(self, input_x):
        count = 0
        x_copy = input_x.copy()
        x_copy = x_copy[::-1]  # Reverse order of time steps
        output_y = np.zeros(shape=(self.num_rolling_days_ahead, self.num_features))
        row_idx = 0
        while count < self.num_rolling_days_ahead:
            x_copy_reshaped = x_copy.reshape(x_copy.shape[0], self.num_of_prev_timesteps, self.num_features)
            current_pred = self.model.predict(x_copy_reshaped)
            temp = 1
            for k in range(self.num_of_future_timesteps):
                output_y[row_idx] = current_pred[k]
                row_idx += 1
                if row_idx >= self.num_rolling_days_ahead:
                    break
                temp_row = x_copy[k][(temp * self.num_features):]
                tail = current_pred[:temp]
                x_copy[k] = np.append(temp_row, tail)
                temp += 1
            count += self.num_of_future_timesteps
        inv_y_rolling = self.scaler.inverse_transform(output_y)
        return inv_y_rolling


if __name__ == '__main__':
    lstm_predictor = LSTMPredictor()
    lstm_predictor.train()
    index_of_plotted_feature = 0
    plotted_feature_str = \
        {0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}[index_of_plotted_feature]
    print('We will plot: ' + plotted_feature_str)

    inv_y_train, inv_y_test, inv_y_pred_train, inv_y_pred_test, inv_y_future_pred, inv_y_rolling_pred = \
        lstm_predictor.predict_all()

    # calculate RMSE
    rmse = sqrt(
        mean_squared_error(inv_y_test[:, index_of_plotted_feature], inv_y_pred_test[:, index_of_plotted_feature]))
    print('Test RMSE (for ' + plotted_feature_str + ') is: %.3f' % rmse)

    matplot_dates = convert_to_matplot_dates(lstm_predictor.dataset.reset_index()["date"])
    print('lstm_predictor.test_set_size: \n' + str(lstm_predictor.test_set_size) + ' len of matplot_dates=' + str(len(matplot_dates)))
    matplot_test_dates = matplot_dates[:lstm_predictor.test_set_size]
    matplot_train_dates = matplot_dates[lstm_predictor.test_set_size:lstm_predictor.data_size]
    # Reverse dates because they are in decreasing order
    matplot_test_dates = matplot_test_dates[::-1]
    matplot_train_dates = matplot_train_dates[::-1]

    # Train vs Pred
    pyplot.figure(0)
    train_data_graph, = pyplot.plot_date(matplot_train_dates, inv_y_pred_train[:, index_of_plotted_feature], 'b-',
                                         label='Training data', color="red")
    train_prediction_graph, = pyplot.plot_date(matplot_train_dates, inv_y_train[:, index_of_plotted_feature], 'b-',
                                              label='Prediction', color="blue")
    pyplot.xlabel('Date')
    pyplot.ylabel(plotted_feature_str)
    pyplot.title('Prediction of ' + plotted_feature_str + ' vs. Training data')
    pyplot.legend(handles=[train_data_graph, train_prediction_graph])
    pyplot.show()

    # Test vs Pred
    pyplot.figure(1)
    test_data_graph, = pyplot.plot_date(matplot_test_dates, inv_y_pred_test[:, index_of_plotted_feature], 'b-',
                                        label='Test data', color="red")
    test_prediction_graph, = pyplot.plot_date(matplot_test_dates, inv_y_test[:, index_of_plotted_feature], 'b-',
                                              label='Prediction', color="blue")
    pyplot.xlabel('Date')
    pyplot.ylabel(plotted_feature_str)
    pyplot.title('Prediction of ' + plotted_feature_str + ' vs. Test data')
    pyplot.legend(handles=[test_data_graph, test_prediction_graph])
    pyplot.show()

    # Future (not rolling pred)
    pyplot.figure(2)
    print('inv_y_future_pred: \n' + str(inv_y_future_pred))
    future_prediction_graph, = pyplot.plot(range(1, lstm_predictor.num_of_future_timesteps + 1),
                                           inv_y_future_pred[:, index_of_plotted_feature], label='Future Prediction')
    pyplot.xlabel('N days ahead')
    pyplot.ylabel(plotted_feature_str)
    pyplot.title('Prediction of ' + plotted_feature_str + ' for ' + str(lstm_predictor.num_of_future_timesteps)
                 + ' days after last data entry')
    pyplot.legend(handles=[future_prediction_graph])
    pyplot.show()

    # Rolling Prediction
    pyplot.figure(3)
    print(str(inv_y_rolling_pred[:, index_of_plotted_feature]))
    rolling_prediction_graph, = pyplot.plot(range(1, lstm_predictor.num_rolling_days_ahead + 1),
                                            inv_y_rolling_pred[:, index_of_plotted_feature],
                                            label='Future Prediction')
    pyplot.xlabel('N days ahead')
    pyplot.ylabel(plotted_feature_str)
    pyplot.title('Rolling Prediction of ' + plotted_feature_str + ' for ' + str(lstm_predictor.num_rolling_days_ahead)
                 + ' days after last data entry')
    pyplot.legend(handles=[rolling_prediction_graph])
    pyplot.show()
    '''
    def invert(input_y):
    input_y = input_y.reshape((len(input_y), num_features))
    # invert scaling and add missing columns to get back actual value
    inv_y = scaler.inverse_transform(input_y)
    # Because our data is in decreasing (time-wise) order we have to reverse the result
    inv_y = inv_y[::-1]
    return inv_y


def make_pred_and_invert(input_x, nn_model):
    y_pred = nn_model.predict(input_x)
    # print('y_pred: \n' + str(y_pred))
    inv_pred_y = scaler.inverse_transform(y_pred)
    # Because our data is in decreasing (time-wise) order we have to reverse the result
    inv_pred_y = inv_pred_y[::-1]
    return inv_pred_y


def make_rolling_pred_and_invert(input_x, num_days_ahead, nn_model):
    count = 0
    x_copy = input_x.copy()
    x_copy = x_copy[::-1]  # Reverse order of time steps
    output_y = np.zeros(shape=(num_days_ahead, num_features))
    row_idx = 0
    while count < num_days_ahead:
        x_copy_reshaped = x_copy.reshape(x_copy.shape[0], num_of_prev_timesteps, num_features)
        current_pred = nn_model.predict(x_copy_reshaped)
        temp = 1
        for k in range(num_of_future_timesteps):
            output_y[row_idx] = current_pred[k]
            row_idx += 1
            if row_idx >= num_days_ahead:
                break
            temp_row = x_copy[k][(temp * num_features):]
            lol = (current_pred[:temp])
            x_copy[k] = np.append(temp_row, lol)
            temp += 1
        count += num_of_future_timesteps
    inv_y_rolling = scaler.inverse_transform(output_y)
    return inv_y_rolling
    
    
    dataset = read_csv('../data/lstm_dates.csv', header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    num_of_prev_timesteps = 7
    num_of_future_timesteps = 3
    num_features = 5
    num_prev_objs = num_features * num_of_prev_timesteps
    num_future_objs = num_features * num_of_future_timesteps
    # open = 0, high = 1, low = 2, close = 3, volume = 4
    index_of_predicted_feature = 0  # The feature that shall be plotted (all will be predicted)
    num_rolling_days_ahead = 30
    predicted_feature_str = {0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}[index_of_predicted_feature]
    print('LSTM will predict ' + predicted_feature_str)

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, num_of_prev_timesteps, num_of_future_timesteps)

    print('reframed: \n' + str(reframed.head()))
    scaled_values = reframed.values  # Extract numpy array from a pandas DataFrame

    print('Dim of scaled_values: ' + str(scaled_values.shape))
    last_observations = scaled_values[:num_of_future_timesteps, -num_prev_objs:]
    print('last_observations: \n' + str(last_observations))

    data_size = reframed.shape[0]
    print('Data size: ' + str(data_size))
    # Training set is 80% of the examples
    training_set_size = int(0.8 * data_size)
    test_set_size = data_size - training_set_size
    # split into input and outputs
    # Training set contains the older (time-wise) part of data
    training_set_x_y = scaled_values[test_set_size:, :]
    # Test set has the newer (time-wise) part of data
    test_set_x_y = scaled_values[:test_set_size, :]
    print(dataset.head())

    test_dates_dataframe = dataset.reset_index()["date"]
    test_datetimes = to_datetime(test_dates_dataframe.values)
    list_test_datetimes = test_datetimes.to_pydatetime().tolist()[:test_set_size]
    matplot_test_dates = dates.date2num(list_test_datetimes)
    matplot_test_dates = matplot_test_dates[::-1]

    train_x, train_y = training_set_x_y[:, :num_prev_objs], training_set_x_y[:, -num_features:]
    test_x, test_y = test_set_x_y[:, :num_prev_objs], test_set_x_y[:, -num_features:]
    # reshape input to be 3D [samples, timesteps, features] as expected by LSTM
    train_x = train_x.reshape(train_x.shape[0], num_of_prev_timesteps, num_features)
    test_x = test_x.reshape(test_x.shape[0], num_of_prev_timesteps, num_features)
    last_observations_reshaped = last_observations.reshape(last_observations.shape[0], num_of_prev_timesteps,
                                                           num_features)

    print('Training input size: ' + str(train_x.shape) + '\n'
          + 'Training output size: ' + str(train_y.shape) + '\n'
          + 'Test input size: ' + str(test_x.shape) + '\n'
          + 'Test output size: ' + str(test_y.shape))
    print('test_set_x_y: \n' + str(test_set_x_y))

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(512, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(num_features))
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    # fit network
    history = model.fit(train_x, train_y, epochs=20, batch_size=64, validation_data=(test_x, test_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.figure(0)
    pyplot.plot(history.history['loss'], label='Training loss (error)')
    pyplot.plot(history.history['val_loss'], label='Test loss (error)')
    pyplot.legend()
    pyplot.show()

    # Make prediction for future given last N observations
    inv_y = invert(input_y=test_y)
    inv_y_pred_test = make_pred_and_invert(input_x=test_x, nn_model=model)
    inv_y_future_pred = make_pred_and_invert(input_x=last_observations_reshaped, nn_model=model)
    inv_y_rolling_pred = make_rolling_pred_and_invert(input_x=last_observations, num_days_ahead=num_rolling_days_ahead, nn_model=model)

    pyplot.figure(1)
    test_prediction_graph, = pyplot.plot_date(matplot_test_dates, inv_y[:, index_of_predicted_feature], 'b-',
                                              label='Prediction', color="blue")
    test_data_graph, = pyplot.plot_date(matplot_test_dates, inv_y_pred_test[:, index_of_predicted_feature], 'b-',
                                        label='Test data', color="red")
    pyplot.xlabel('Date')
    pyplot.ylabel(predicted_feature_str)
    pyplot.title('Prediction of ' + predicted_feature_str + ' vs. Test data')
    pyplot.legend(handles=[test_data_graph, test_prediction_graph])
    pyplot.show()

    pyplot.figure(2)
    print('inv_y_future_pred: \n' + str(inv_y_future_pred))
    future_prediction_graph, = pyplot.plot(range(1, num_of_future_timesteps + 1),
                                           inv_y_future_pred[:, index_of_predicted_feature], label='Future Prediction')
    pyplot.xlabel('N days ahead')
    pyplot.ylabel(predicted_feature_str)
    pyplot.title('Prediction of ' + predicted_feature_str + ' for ' + str(num_of_future_timesteps)
                 + ' days after last data entry')
    pyplot.legend(handles=[future_prediction_graph])
    pyplot.show()

    pyplot.figure(3)
    print(str(inv_y_rolling_pred[:, index_of_predicted_feature]))
    rolling_prediction_graph, = pyplot.plot(range(1, num_rolling_days_ahead + 1),
                                            inv_y_rolling_pred[:, index_of_predicted_feature], label='Future Prediction')
    pyplot.xlabel('N days ahead')
    pyplot.ylabel(predicted_feature_str)
    pyplot.title('Rolling Prediction of ' + predicted_feature_str + ' for ' + str(num_rolling_days_ahead)
                 + ' days after last data entry')
    pyplot.legend(handles=[rolling_prediction_graph])
    pyplot.show()

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y[:, index_of_predicted_feature], inv_y_pred_test[:, index_of_predicted_feature]))
    print('Test RMSE (for ' + predicted_feature_str + ') is: %.3f' % rmse)
    '''
