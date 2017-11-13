import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import src.csv_parser as CsvParser
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn.metrics import mean_squared_error
from keras.activations import relu, elu
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.losses import mean_absolute_error


if __name__ == '__main__':
    normalizer = StandardScaler()
    df = pd.read_csv('../data/corrected_dates.csv')  # By default header will be read from file

    print('Head of data frame: \n' + str(df.head()))
    print('Dimensions of data frame (row x col)' + str(df.shape))

    num_of_rows = df.shape[0]

    print('Number of rows in data frame: ' + str(num_of_rows))
    print('----------SPLITTING DATA INTO TRAINING AND TEST SETS----------')

    training_set_size = int(0.8 * num_of_rows)
    test_set_size = int(num_of_rows - training_set_size)

    print('Training set size: ' + str(training_set_size))
    print('Test set size: ' + str(test_set_size))

    df_train, df_test = df[test_set_size:], df[:test_set_size]
    print('Dimensions of training data frame (row x col)' + str(df_train.shape))
    print('Dimensions of test data frame (row x col)' + str(df_test.shape))

    X_tr, y_tr = df_train['timestamp'], df_train[['open', 'high', 'low', 'close']]
    X_te, y_te = df_test['timestamp'], df_test[['open', 'high', 'low', 'close']]
    # X_tr_tmp is all training dates except for last one (oldest) because we do not have open price of previous day
    X_tr_tmp = X_tr.values[:-1]
    # X_tr2_tmp represents the open price of the previous day
    # So for example if X_tr_tmp[0] is Nov 3, then X_tr2_tmp[0] is then open price for Nov 2
    # We take all rows besides first one because first open price y_tr.values[0,0] does not have a future date
    X_tr2_tmp = y_tr.values[1:, 0]

    # We do the same thing as we did above for the test set
    X_te_tmp = X_te.values[:-1]
    X_te2_tmp = y_te.values[1:, 0]

    X_tmp = np.concatenate([X_tr_tmp.reshape(-1, 1), X_tr2_tmp.reshape(-1, 1)], axis=1)
    # Remove last (oldest) open price in y, since it does not have a previous open day
    y_tmp = y_tr.values[:-1, :]
    X_tmp_te = np.concatenate([X_te_tmp.reshape(-1, 1), X_te2_tmp.reshape(-1, 1)], axis=1)
    test_set_size = X_te_tmp.shape[0]
    training_set_size = X_tr_tmp.shape[0]
    print('Training set size (after shifting): ' + str(training_set_size))
    print('Test set size: (after shifting)' + str(test_set_size))

    print('X_tmp: \n' + str(X_tmp))
    model = Sequential()
    model.add(BatchNormalization(input_shape=(2,)))  # Equivalent to input_dim=2 since we have 2 columns of input
    model.add(Dense(1024, use_bias=False, activation='tanh'))  # Dense means all nodes are connected with each other
    model.add(Dense(1, use_bias=False))
    model.summary()
    model.compile('adam', 'mse', metrics=['mse'])
    # Num of epochs in num of iterations where it takes batch_size number elements from X and y
    model.fit(X_tmp, y_tmp[:, 0], epochs=40, batch_size=32)

    some_day = '2017-11-09'
    open_of_prev_day = 84.1100
    unix_time = CsvParser.from_timestamp_to_unix_time(some_day)

    print('Prediction for ' + some_day + ' and open of previous day ' + str(open_of_prev_day) + ' is: ' +
          str(model.predict(np.array([[unix_time, open_of_prev_day]]))))

    prediction = model.predict(X_tmp_te)
    prediction_tr = model.predict(X_tmp)

    for i in range(test_set_size):
        d = CsvParser.from_unix_time_to_timestamp(X_tmp_te[i, 0])
        op = str(X_tmp_te[i, 1])
        print('Pred for day: ' + d + ' given open price of prev day: ' + op + ' is: ' + str(prediction[i, 0]))

    last_known_open_price = y_te.values[0, 0]
    last_day_in_data = X_te[0]
    print('Last known open price: ' + str(last_known_open_price) + ' last day in data: '
          + CsvParser.from_unix_time_to_timestamp(last_day_in_data))
    # Prediction of open price for the first future
    next_day = last_day_in_data + (3600 * 24)
    print('Next day after last day in data: ' + CsvParser.from_unix_time_to_timestamp(next_day))
    open_pred = model.predict(np.array([[last_known_open_price, next_day]]))
    print(str('Pred for day: '
              + CsvParser.from_unix_time_to_timestamp(next_day)
              + ' is: ' + str(open_pred))
              + ' given open price of prev day: ' + str(last_known_open_price))

    predictions = []
    future_days = []

    for i in range(30):
        next_day = next_day + (3600 * 24)
        open_prev = open_pred[0][0]
        open_pred = model.predict(np.array([[next_day, open_pred[0][0]]]))
        print(str('Pred for day: ' + str(datetime.datetime.fromtimestamp(next_day).strftime('%Y-%m-%d %H:%M:%S'))
                    + ' is: ' + str(open_pred))
                    + ' given open price of prev day: ' + str(open_prev))
        predictions.append(open_pred[0][0])
        future_days.append(next_day)

    plt.figure(0)
    plt.plot(future_days, predictions)
    plt.xlabel('Date (Unix time)')
    plt.ylabel('Open price')
    plt.title('Future Prediction: 1 Month after ' + CsvParser.from_unix_time_to_timestamp(last_day_in_data))
    plt.show()

    plt.figure(1)
    # I add a comma after the var name because of ...
    # https://stackoverflow.com/questions/36329269/python-legend-attribute-error
    training_data_graph, = plt.plot(X_tr.values, y_tr.values[:, 0], label='Actual training data')
    training_prediction_graph, = plt.plot(X_tmp[:, 0], prediction_tr[:, 0], label='Prediction of training data')
    plt.xlabel('Date (Unix time)')
    plt.ylabel('Open price')
    plt.title('Prediction of Open Price')
    plt.legend(handles=[training_data_graph, training_prediction_graph])
    plt.show()
    plt.figure(2)
    test_data_graph, = plt.plot(X_te.values, y_te.values[:, 0], label='Test set')
    test_prediction_graph, = plt.plot(X_tmp_te[:, 0], prediction[:, 0], label='Prediction of test set')
    plt.xlabel('Date (Unix time)')
    plt.ylabel('Open price')
    plt.title('Prediction of Open Price')
    plt.legend(handles=[test_data_graph, test_prediction_graph])
    plt.show()
