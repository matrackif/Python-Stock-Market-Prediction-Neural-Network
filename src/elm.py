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
class ELM:
    def __init__(self, csv_file: str = '../data/corrected_dates.csv',
                 index_of_plotted_feature: int = 0, train_w: bool = False):
        df = pd.read_csv(csv_file)  # By default header will be read from file
        num_of_rows = df.shape[0]
        self.num_features = 4
        self.num_hidden_layer_neurons = 512
        self.index_of_plotted_feature = index_of_plotted_feature
        self.train_w = train_w
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
        # https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
        self.model = Sequential()
        # self.model.add(Dense(200, input_shape=self.input_shape, use_bias=True, name=self.input_layer_name))
        self.model.add(Dense(self.num_hidden_layer_neurons, activation='sigmoid',
                             name=self.hidden_layer_name, input_shape=self.input_shape))

    def create_h(self):
        h = None
        if self.train_w:
            # Train w but do not train "beta"
            temp_hidden_name = 'temp_hidden'
            temp_model = Sequential()
            # temp_model.add(Dense(self.num_features, input_shape=self.input_shape))
            temp_model.add(Dense(self.num_hidden_layer_neurons, activation='sigmoid', name=temp_hidden_name,
                                 weights=self.model.get_layer(self.hidden_layer_name).get_weights(),
                                 input_shape=self.input_shape))
            temp_model.add(Dense(self.num_features, use_bias=False, trainable=False))
            temp_model.compile('adam', 'mse', metrics=['mse'])
            temp_model.fit(self.x_tr, self.x_tr, epochs=10, batch_size=32)
            hidden_layer_only_model = Model(inputs=temp_model.input,
                                            outputs=temp_model.get_layer(temp_hidden_name).output)
            h = np.mat(hidden_layer_only_model.predict(self.x_tr))
        else:
            h = np.mat(self.model.predict(self.x_tr))
        print('First weight matrix with shape: \n' + str(self.model.get_layer(self.hidden_layer_name).get_weights()))
        print('Created h with shape: ' + str(h.shape) + '\n And values: \n' + str(h))
        return h

    def train(self):
        H = self.create_h()
        T = np.mat(self.y_tr)
        print('Shape of T: ' + str(T.shape) + '\n And values: \n' + str(T))
        beta = np.linalg.pinv(H) * np.mat(T)
        self.model.add(Dense(self.num_features, use_bias=False, weights=[beta]))
        print('Finished training ELM, model summary:')
        self.model.summary()
        print('Beta: \n' + str(beta))

    def predict(self):
        last_day_in_data = np.array([self.y_te[0]])
        print('Prediction for last day in data with values:' + str(last_day_in_data) + ' with shape: '
              + str(last_day_in_data.shape))
        print(str(self.model.predict(last_day_in_data)))
        training_pred = self.model.predict(self.x_tr)
        test_pred = self.model.predict(self.x_te)
        return training_pred, test_pred


if __name__ == '__main__':
    elm = ELM()
    elm.train()
    '''
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
    print('shape X_tr2_tmp: ' + str(X_tr2_tmp.shape))

    # We do the same thing as we did above for the test set
    X_te_tmp = X_te.values[:-1]
    X_te2_tmp = y_te.values[1:, 0]
    print('shape X_te2_tmp: ' + str(X_te2_tmp.shape))
    X_tmp = np.concatenate([X_tr_tmp.reshape(-1, 1), X_tr2_tmp.reshape(-1, 1)], axis=1)
    # Remove last (oldest) open price in y, since it does not have a previous open day
    y_tmp = y_tr.values[:-1, :]
    X_tmp_te = np.concatenate([X_te_tmp.reshape(-1, 1), X_te2_tmp.reshape(-1, 1)], axis=1)
    test_set_size = X_te_tmp.shape[0]
    training_set_size = X_tr_tmp.shape[0]
    print('Training set size (after shifting): ' + str(training_set_size))
    print('Test set size: (after shifting)' + str(test_set_size))

    print('X_tmp: \n' + str(X_tmp))
    input_shape = (1,)
    hidden_layer_name = 'hidden_layer'

    model = Sequential()
    model.add(Dense(1, input_shape=input_shape))
    model.add(Dense(1024, use_bias=False, name=hidden_layer_name))
    model.add(Dense(1, use_bias=False))
    model.summary()
    model.compile('adam', 'mse', metrics=['mse'])
    # Num of epochs in num of iterations where it takes batch_size number elements from X and y
    # model.fit(X_tr2_tmp, y_tmp[:, 0], epochs=10, batch_size=32)

    # TODO DO NOT FIT the model, instead initialize weights randomly
    print('model.input: \n' + str(model.input))
    hidden_layer_only_model = Model(inputs=model.input, outputs=model.get_layer(hidden_layer_name).output)
    # hidden_layer_only_model.summary()
    # hidden_layer_only_model = Sequential()
    # hidden_layer_only_model.add(BatchNormalization(input_shape=input_shape))
    # hidden_layer_only_model.add(Dense(1024, use_bias=False, activation='tanh',  weights=model.get_layer(hidden_layer_name).get_weights()))
    # hidden_layer_only_model.compile('adam', 'mse', metrics=['mse'])
    hidden_layer_only_model.summary()
    elm_H = hidden_layer_only_model.predict(X_tr2_tmp)

    # Find beta
    elm_H_mat = np.mat(elm_H)
    elm_H_mat_transposed = np.transpose(elm_H_mat)
    elm_T_mat = np.mat(y_tmp[:, 0])
    print('y_tmp[:, 0]: \n' + str(y_tmp[:, 0]))
    elm_T_mat_transposed = np.transpose(elm_T_mat)
    print('elm_H_mat: \n' + str(elm_H_mat) + '\n' + 'elm_H_mat_transposed: \n' + str(elm_H_mat_transposed)
          + '\n' + 'elm_H_mat_transposed: \n' + str(elm_H_mat_transposed))
    print('Shape elm_H_mat: ' + str(elm_H_mat.shape)
          + ' Shape elm_H_mat_transposed: ' + str(elm_H_mat_transposed.shape)
          + ' Shape elm_t_mat_transposed: ' + str(elm_T_mat_transposed.shape))
    # elm_H_dagger = np.linalg.inv(elm_H_mat_transposed * elm_H_mat) * elm_H_mat_transposed
    pinv_H = np.linalg.pinv(elm_H_mat)
    should_be_identity = elm_H_mat * pinv_H
    print('should_be_identity: \n' + str(should_be_identity))
    print('Type of pinv_H is: ' + str(type(pinv_H)) + ' Shape is: ' + str(pinv_H.shape))
    beta = pinv_H * elm_T_mat_transposed
    beta = np.asarray(beta)
    result = elm_H_mat * beta
    print('result: ' + str(result))
    print('Beta: ' + str(beta) + 'Type of beta: ' + str(type(beta)) + ' Shape of beta: ' + str(beta.shape))
    print('Type of elm_H is: ' + str(type(elm_H)) + ' Shape is: ' + str(elm_H.shape))

    # Create ELM keras model
    elm_model = Sequential()
    elm_model.add(Dense(1, input_shape=input_shape))
    elm_model.add(Dense(1024, use_bias=False, activation='tanh', weights=model.get_layer(hidden_layer_name).get_weights()))
    elm_model.add(Dense(1, use_bias=False, weights=[beta]))
    weights = elm_model.layers[-1].get_weights()
    print('ELM Final layer weights: ' + str(weights) + '\n Shape of weights: ' + str(weights[0].shape))
    weights = beta
    elm_model.summary()
    elm_model.compile('adam', 'mse', metrics=['mse'])

    some_day = '2017-11-09'
    open_of_prev_day = 84.1100
    unix_time = CsvParser.from_timestamp_to_unix_time(some_day)

    print('Prediction for ' + some_day + ' and open of previous day ' + str(open_of_prev_day) + ' is: ' +
          str(model.predict(np.array([[open_of_prev_day]]))))

    prediction = model.predict(X_te2_tmp)
    prediction_tr = model.predict(X_tr2_tmp)
    prediction_elm_tr = elm_model.predict(X_tr2_tmp)
    prediction_elm_test = elm_model.predict(X_te2_tmp)

    for i in range(test_set_size):
        d = CsvParser.from_unix_time_to_timestamp(X_tmp_te[i, 0])
        op = str(X_tmp_te[i, 1])
        # print('Pred for day: ' + d + ' given open price of prev day: ' + op + ' is: ' + str(prediction[i, 0]))

    last_known_open_price = y_te.values[0, 0]
    last_day_in_data = X_te[0]
    print('Last known open price: ' + str(last_known_open_price) + ' last day in data: '
          + CsvParser.from_unix_time_to_timestamp(last_day_in_data))
    # Prediction of open price for the first future
    next_day = last_day_in_data + (3600 * 24)
    print('Next day after last day in data: ' + CsvParser.from_unix_time_to_timestamp(next_day))
    open_pred = model.predict(np.array([[last_known_open_price]]))
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
        # print(str('Pred for day: ' + str(datetime.datetime.fromtimestamp(next_day).strftime('%Y-%m-%d %H:%M:%S'))
        #             + ' is: ' + str(open_pred))
        #             + ' given open price of prev day: ' + str(open_prev))
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
    print('X_tr2_tmp: \n' + str(X_tr2_tmp) + '\n shape X_tr2_tmp: ' + str(X_tr2_tmp.shape))
    print('y_tmp: \n' + str(y_tmp) + '\n shape y_tmp: ' + str(y_tmp.shape))
    training_data_graph, = plt.plot(y_tmp[:, 0], label='Actual training data')
    training_prediction_graph, = plt.plot(prediction_tr, label='Prediction of training data')
    plt.xlabel('Date (Unix time)')
    plt.ylabel('Open price')
    plt.title('Prediction of Open Price')
    plt.legend(handles=[training_data_graph, training_prediction_graph])
    plt.show()

    plt.figure(2)
    test_data_graph, = plt.plot(y_te.values[:, 0], label='Test set')
    test_prediction_graph, = plt.plot(prediction[:, 0], label='Prediction of test set')
    plt.xlabel('Date (Unix time)')
    plt.ylabel('Open price')
    plt.title('Prediction of Open Price')
    plt.legend(handles=[test_data_graph, test_prediction_graph])
    plt.show()

    print('ELM TEST PRED: \n' + str(prediction_elm_test))
    plt.figure(3)
    test_data_graph, = plt.plot(y_te.values[:, 0], label='Test set')
    elm_test_prediction_graph, = plt.plot(prediction_elm_test[:, 0], label='Prediction of test set')
    plt.xlabel('Date (Unix time)')
    plt.ylabel('Open price')
    plt.title('ELM Prediction of Open Price')
    plt.legend(handles=[test_data_graph, elm_test_prediction_graph])
    plt.show()

    plt.figure(4)
    training_data_graph, = plt.plot(y_tmp[:, 0], label='Actual training data')
    elm_training_prediction_graph, = plt.plot(prediction_elm_tr, label='Prediction of training data')
    plt.xlabel('Date (Unix time)')
    plt.ylabel('Open price')
    plt.title('ELM Prediction of Open Price')
    plt.legend(handles=[training_data_graph, elm_training_prediction_graph])
    plt.show()
    '''
