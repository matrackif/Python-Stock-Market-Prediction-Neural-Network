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
from src.elm import ELM
from src.regular_model import RegularModel

if __name__ == '__main__':
    '''
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
    '''
    reg_model = RegularModel()
    elm = ELM()
    reg_model.train()
    elm.train()
    real_train = elm.x_tr
    real_test = elm.x_te
    elm_train_pred, elm_test_pred = elm.predict()
    reg_train_pred, reg_test_pred = reg_model.predict()
    index_of_plotted_feature = elm.index_of_plotted_feature

    plt.figure(0)
    training_data_graph, = plt.plot(real_train[:, index_of_plotted_feature], label='Actual training data')
    training_prediction_graph, = plt.plot(reg_train_pred[:, index_of_plotted_feature],
                                          label='Prediction of training data')
    training_prediction_elm_graph, = plt.plot(elm_train_pred[:, index_of_plotted_feature],
                                              label='ELM Prediction of training data')

    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.title('ELM vs Reg model vs Real data')
    plt.legend(handles=[training_data_graph, training_prediction_graph, training_prediction_elm_graph])
    plt.show()

    '''
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
    '''

