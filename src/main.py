import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn.metrics import mean_squared_error
from keras.activations import relu, elu
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.losses import mean_absolute_error


if __name__ == '__main__':
    # df_dates = pd.read_csv('./dates_daily_CDR.csv', header=None)
    # df_target = pd.read_csv('./results_daily_CDR.csv', header=None)
    # print('Head of dates: \n' + str(df_dates.head()))
    # print('Head of targets: \n' + str(df_target.head()))

    normalizer = StandardScaler()
    # df = df_target
    df = pd.read_csv('../data/corrected_dates.csv') # By default header will be read
    # df['dates'] = df_dates
    # df.rename(columns={0: 'open', 1: 'high', 2: 'low', 3: 'close'}, inplace=True)  # Rename columns of data frame
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

    model = Sequential()
    model.add(BatchNormalization(input_shape=(2,)))
    model.add(Dense(1024, use_bias=False, activation='tanh'))  # Dense means all nodes are connected with each other
    model.add(Dense(1, use_bias=False))
    model.summary()
    model.compile('adam', 'mse', metrics=['mse'])
    # Num of epochs in num of iterations where it takes batch_size number elements from X and y
    model.fit(X_tmp, y_tmp[:, 0], epochs=40, batch_size=32)

    row = '2017-11-07'
    open_of_prev_day = 5.8600
    # Convert date of form year-month-day to unix time
    timestamp_as_list = row.split('-')
    year = int(timestamp_as_list[0])
    month = int(timestamp_as_list[1])
    day = int(timestamp_as_list[2])
    date = datetime.datetime(year=year, month=month, day=day, tzinfo=datetime.timezone.utc)  # Convert to UTC time
    unix_time = int(date.timestamp())

    print('Prediction for ' + row + ' and open of previous day ' + str(open_of_prev_day) + ' is: ' +
          str(model.predict(np.array([[unix_time, open_of_prev_day]]))))

    prediction = model.predict(X_tmp_te)
    prediction_tr = model.predict(X_tmp)

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