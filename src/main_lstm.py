# Inspired by https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
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


def invert(input_x, input_y):
    input_y = input_y.reshape((len(input_y), 1))
    # invert scaling and add missing columns to get back actual value
    # The only purpose of concatenating the columns is to
    input_x = input_x.reshape((input_x.shape[0], num_prev_objs))
    inv_y = concatenate((input_y, input_x[:, -(num_features - 1):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # Because our data is in decreasing (time-wise) order we have to reverse the result
    inv_y = inv_y[::-1]
    return inv_y


def make_pred_and_invert(input_x, nn_model):
    inv_pred_y = None
    y_pred = nn_model.predict(input_x)
    # invert scaling and add missing columns to get back actual value
    # The only purpose of concatenating the columns is to
    input_x = input_x.reshape((input_x.shape[0], num_prev_objs))
    inv_pred_y = concatenate((y_pred, input_x[:, -(num_features - 1):]), axis=1)
    inv_pred_y = scaler.inverse_transform(inv_pred_y)
    inv_pred_y = inv_pred_y[:, 0]
    # Because our data is in decreasing (time-wise) order we have to reverse the result
    inv_pred_y = inv_pred_y[::-1]

    return inv_pred_y

if __name__ == '__main__':
    dataset = read_csv('../data/lstm_dates.csv', header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    num_of_prev_timesteps = 7
    num_of_future_timesteps = 7
    num_features = 5
    num_prev_objs = num_features * num_of_prev_timesteps
    num_future_objs = num_features * num_of_future_timesteps
    # open = 0, high = 1, low = 2, close = 3, volume = 4
    index_of_predicted_feature = 1
    predicted_feature_str = {0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}[index_of_predicted_feature]
    print('LSTM will predict ' + predicted_feature_str)
    # normalize features
    copied_dataset = dataset.copy()
    copied_dataset.drop(copied_dataset.columns[[i for i in range(num_features) if i is not index_of_predicted_feature]],
                        axis=1, inplace=True)
    copied_dataset_values = copied_dataset.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled_y_only = scaler.fit_transform(copied_dataset_values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, num_of_prev_timesteps, num_of_future_timesteps)
    reframed_y_only = series_to_supervised(scaled_y_only, num_of_prev_timesteps, num_of_future_timesteps)
    print('reframed_y_only: \n' + str(reframed_y_only.head()))
    print('reframed: \n' + str(reframed.head()))
    scaled_values = reframed.values  # Extract numpy array from a pandas DataFrame
    scaled_values_y_only = reframed_y_only.values

    print('Dim of scaled_values: ' + str(scaled_values.shape))
    last_observations = scaled_values[0:num_of_future_timesteps, -num_prev_objs:]
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
    # y_only contains only the time intervals of the variable we want to predict
    training_set_y_only = scaled_values_y_only[test_set_size:, :]
    test_set_y_only = scaled_values_y_only[:test_set_size, :]

    train_x, train_y = training_set_x_y[:, :num_prev_objs], training_set_y_only[:, -1]
    test_x, test_y = test_set_x_y[:, :num_prev_objs], test_set_y_only[:, -1]
    # reshape input to be 3D [samples, timesteps, features] as expected by LSTM
    train_x = train_x.reshape(train_x.shape[0], num_of_prev_timesteps, num_features)
    test_x = test_x.reshape(test_x.shape[0], num_of_prev_timesteps, num_features)
    last_observations = last_observations.reshape(last_observations.shape[0], num_of_prev_timesteps, num_features)

    print('Training input size: ' + str(train_x.shape) + '\n'
          + 'Training output size: ' + str(train_y.shape) + '\n'
          + 'Test input size: ' + str(test_x.shape) + '\n'
          + 'Test output size: ' + str(test_y.shape))
    print('test_set_x_y: \n' + str(test_set_x_y))

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(1024, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_x, train_y, epochs=20, batch_size=64, validation_data=(test_x, test_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='Training loss (error)')
    pyplot.plot(history.history['val_loss'], label='Test loss (error)')
    pyplot.legend()
    pyplot.show()

    # Make prediction for future given last N observations
    inv_y = invert(input_x=test_x, input_y=test_y)
    inv_y_pred_test = make_pred_and_invert(input_x=test_x, nn_model=model)
    inv_y_future_pred = make_pred_and_invert(input_x=last_observations, nn_model=model)

    pyplot.figure(1)
    test_prediction_graph, = pyplot.plot(inv_y_pred_test, label='Prediction')
    test_data_graph, = pyplot.plot(inv_y, label='Test data')
    pyplot.xlabel('Days')
    pyplot.ylabel(predicted_feature_str)
    pyplot.title('Prediction of ' + predicted_feature_str + ' vs. Test data')
    pyplot.legend(handles=[test_data_graph, test_prediction_graph])
    pyplot.show()

    pyplot.figure(2)
    future_prediction_graph, = pyplot.plot(inv_y_future_pred, label='Future Prediction')
    pyplot.xlabel('N days ahead')
    pyplot.ylabel(predicted_feature_str)
    pyplot.title('Prediction of ' + predicted_feature_str + ' for ' + str(num_of_future_timesteps)
                 + ' days after last data entry')
    pyplot.legend(handles=[future_prediction_graph])
    pyplot.show()

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_y_pred_test))
    print('Test RMSE: %.3f' % rmse)