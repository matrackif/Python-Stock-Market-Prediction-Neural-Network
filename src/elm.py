import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib
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


def convert_to_matplot_dates(dates_dataframe):
    datetimes = pd.to_datetime(dates_dataframe.values.flatten(), unit='s')
    list_datetimes = datetimes.to_pydatetime().tolist()
    return matplotlib.dates.date2num(list_datetimes)


def rand_init(shape, dtype=None):
    epsilon = 0.12
    return (np.random.random_sample(shape) * 2 * epsilon) - epsilon


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
                             name=self.hidden_layer_name, input_shape=self.input_shape, kernel_initializer=rand_init))

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


class ELMMatrixVersion:
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
        self.x_tr = np.mat(self.x_tr.values[1:, :])
        self.x_te = np.mat(self.x_te.values[1:, :])
        self.y_tr = np.mat(self.y_tr.values[:-1, :])
        self.y_te = np.mat(self.y_te.values[:-1, :])

        # Add bias term of ones to test X and train X
        self.x_tr = np.concatenate((np.ones(shape=(self.x_tr.shape[0], 1)), self.x_tr), axis=1)
        self.x_te = np.concatenate((np.ones(shape=(self.x_te.shape[0], 1)), self.x_te), axis=1)

        self.input_shape = (self.num_features + 1,)
        self.input_layer_weights = np.mat(rand_init(shape=(self.num_hidden_layer_neurons, self.num_features + 1)))

        self.beta = None

    def create_h(self):
        h = None
        print('Shape of input_layer_weights: ' + str(self.input_layer_weights.shape))
        print('Shape of training X: ' + str(self.x_tr.shape))
        print('Shape of training Y: ' + str(self.y_tr.shape))
        if self.train_w:
            # Train w but do not train "beta"
            temp_hidden_name = 'temp_hidden'
            temp_model = Sequential()
            # temp_model.add(Dense(self.num_features, input_shape=self.input_shape))
            temp_model.add(Dense(self.num_hidden_layer_neurons, activation='linear', name=temp_hidden_name,
                                 kernel_initializer=rand_init,
                                 input_shape=self.input_shape))
            temp_model.add(Dense(self.num_features, use_bias=False, trainable=False))
            temp_model.compile('adam', 'mse', metrics=['mse'])
            temp_model.fit(self.x_tr, self.y_tr, epochs=10, batch_size=32)
            hidden_layer_only_model = Model(inputs=temp_model.input,
                                            outputs=temp_model.get_layer(temp_hidden_name).output)
            h = np.mat(hidden_layer_only_model.predict(self.x_tr))
        else:
            h = self.x_tr * self.input_layer_weights.T
        print('Created h with shape: ' + str(h.shape) + '\n And values: \n' + str(h))
        return h

    def train(self):
        H = self.create_h()
        T = self.y_tr
        print('Shape of T: ' + str(T.shape) + '\n And values: \n' + str(T))
        self.beta = np.linalg.pinv(H) * np.mat(T)
        print('Finished training ELM')
        print('Beta: \n' + str(self.beta))

    def predict(self):
        print(str(self.y_te[0].shape))
        last_day_in_data = np.hstack(([[1]], self.y_te[0]))
        print('Prediction for last day in data with values:' + str(last_day_in_data) + ' with shape: '
              + str(last_day_in_data.shape))
        print(np.mat(last_day_in_data) * self.input_layer_weights.T * self.beta)
        training_pred_h = self.x_tr * self.input_layer_weights.T
        training_pred = training_pred_h * self.beta
        test_pred_h = self.x_te * self.input_layer_weights.T
        test_pred = test_pred_h * self.beta
        return training_pred, test_pred


class DisjointDataELMMatrixVersion:
    def __init__(self, csv_file: str = '../data/corrected_dates.csv',
                 index_of_plotted_feature: int = 0, train_w: bool = False):
        df = pd.read_csv(csv_file)  # By default header will be read from file
        print('Head of data frame: \n' + str(df.head()))
        print('Dimensions of data frame (row x col)' + str(df.shape))
        self.num_of_rows_in_csv = int(df.shape[0])
        self.num_features = 4
        self.num_hidden_layer_neurons = 512
        self.train_w = train_w
        self.num_prev_timesteps = 8
        self.num_future_timesteps = 1
        self.num_prev_attributes = self.num_prev_timesteps * self.num_features
        self.num_future_attributes = self.num_future_timesteps * self.num_features
        self.df_values = df[['open', 'high', 'low', 'close']].values
        self.x_values = np.zeros(shape=(int(self.num_of_rows_in_csv / self.num_prev_timesteps), self.num_prev_attributes))
        self.y_values = np.zeros(shape=(int(self.num_of_rows_in_csv / self.num_prev_timesteps), self.num_future_attributes))
        dates_df = df[['timestamp']]
        self.plotable_dates = convert_to_matplot_dates(dates_df)
        self.plotable_dates = np.reshape(self.plotable_dates, newshape=(-1, 1)).copy()
        self.plotable_dates_list = []
        print('self.plotable_dates: \n' + str(self.plotable_dates) + str(type(self.plotable_dates)))
        idx = 0
        rows_taken = 0
        # Loop until we have enough remaining rows to add an X-Y pair to our dataset
        while rows_taken <= (self.num_of_rows_in_csv - (self.num_prev_timesteps + self.num_future_timesteps)):
            self.y_values[idx] = self.df_values[rows_taken:rows_taken + self.num_future_timesteps].flatten()
            self.plotable_dates_list.extend(self.plotable_dates[rows_taken:rows_taken + self.num_future_timesteps].flatten().tolist())
            rows_taken += self.num_future_timesteps
            self.x_values[idx] = self.df_values[rows_taken:rows_taken + self.num_prev_timesteps].flatten()
            rows_taken += self.num_prev_timesteps
            idx += 1
        print('self.plotable_dates_list: \n' + str(self.plotable_dates_list) + str(len(self.plotable_dates_list)))
        # Remove all unfilled rows
        self.y_values = self.y_values[np.all(self.y_values != 0, axis=1)]
        self.x_values = self.x_values[np.all(self.x_values != 0, axis=1)]

        # print('self.df_values: \n' + str(self.df_values))
        # print('self.x_values: \n' + str(self.x_values))
        # print('self.y_values: \n' + str(self.y_values))
        print('self.df_values shape: ' + str(self.df_values.shape))
        print('self.x_values shape: ' + str(self.x_values.shape))
        print('self.y_values shape: ' + str(self.y_values.shape))

        self.data_size = self.x_values.shape[0]
        self.training_set_size = int(0.8 * self.data_size)
        self.test_set_size = int(self.data_size - self.training_set_size)
        print('Training set size: ' + str(self.training_set_size))
        print('Test set size: ' + str(self.test_set_size))

        self.x_tr = np.mat(self.x_values[self.test_set_size:])
        self.x_te = np.mat(self.x_values[:self.test_set_size])
        self.y_tr = np.mat(self.y_values[self.test_set_size:])
        self.y_te = np.mat(self.y_values[:self.test_set_size])

        # Each row in y contains num_future_timesteps amount of days, hence the multiplication here
        # The list of dates is a 1D list created above
        self.train_dates = self.plotable_dates_list[self.test_set_size * self.num_future_timesteps:]
        self.test_dates = self.plotable_dates_list[:self.test_set_size * self.num_future_timesteps]

        # Add bias term of ones to test X and train X
        self.x_tr = np.concatenate((np.ones(shape=(self.x_tr.shape[0], 1)), self.x_tr), axis=1)
        self.x_te = np.concatenate((np.ones(shape=(self.x_te.shape[0], 1)), self.x_te), axis=1)

        print('x_tr shape: \n' + str(self.x_tr.shape))
        print('x_te shape: \n' + str(self.x_te.shape))
        print('y_tr shape: \n' + str(self.y_tr.shape))
        print('y_te shape: \n' + str(self.y_te.shape))

        # Our beta will be a weight matrix between the hidden and output layer
        self.input_layer_weights = np.mat(rand_init(shape=(self.num_hidden_layer_neurons, self.num_prev_attributes + 1)))
        self.beta = None
        self.input_shape = (self.num_prev_attributes + 1,)

    def create_h(self):
        h = None
        if self.train_w:
            # Train w but do not train "beta"
            temp_hidden_name = 'temp_hidden'
            temp_model = Sequential()
            temp_model.add(Dense(self.num_hidden_layer_neurons, activation='linear', name=temp_hidden_name,
                                 weights=rand_init,
                                 input_shape=self.input_shape))
            temp_model.add(Dense(self.num_future_attributes, use_bias=False, trainable=False))
            temp_model.compile('adam', 'mse', metrics=['mse'])
            temp_model.fit(self.x_tr, self.y_tr, epochs=10, batch_size=32)
            hidden_layer_only_model = Model(inputs=temp_model.input,
                                            outputs=temp_model.get_layer(temp_hidden_name).output)
            h = np.mat(hidden_layer_only_model.predict(self.x_tr))
        else:
            h = self.x_tr * self.input_layer_weights.T
        print('Created h with shape: ' + str(h.shape) + '\n And values: \n' + str(h))
        return h

    def train(self):
        H = self.create_h()
        T = self.y_tr
        print('Shape of T: ' + str(T.shape) + '\n And values: \n' + str(T))
        self.beta = np.linalg.pinv(H) * np.mat(T)
        print('Finished training ELM')
        print('Beta: \n' + str(self.beta))

    def predict(self):
        last_timeframe_in_data = np.array([self.df_values[0:self.num_prev_timesteps].flatten()])
        last_timeframe_in_data = np.hstack(([[1]], last_timeframe_in_data))
        print('last_timeframe_in_data values:' + str(last_timeframe_in_data) + ' last_timeframe_in_data shape: '
              + str(last_timeframe_in_data.shape))
        print('Prediction for last day is :' + str(np.mat(last_timeframe_in_data) * self.input_layer_weights.T * self.beta))
        training_pred_h = self.x_tr * self.input_layer_weights.T
        training_pred = training_pred_h * self.beta
        test_pred_h = self.x_te * self.input_layer_weights.T
        test_pred = test_pred_h * self.beta
        return training_pred, test_pred


class DisjointDataELM:
    def __init__(self, csv_file: str = '../data/corrected_dates.csv',
                 index_of_plotted_feature: int = 0, train_w: bool = False):
        df = pd.read_csv(csv_file)  # By default header will be read from file
        print('Head of data frame: \n' + str(df.head()))
        print('Dimensions of data frame (row x col)' + str(df.shape))
        self.num_of_rows_in_csv = int(df.shape[0])
        self.num_features = 4
        self.num_hidden_layer_neurons = 600
        self.train_w = train_w
        self.num_prev_timesteps = 5
        self.num_future_timesteps = 1
        self.num_prev_attributes = self.num_prev_timesteps * self.num_features
        self.num_future_attributes = self.num_future_timesteps * self.num_features
        self.df_values = df[['open', 'high', 'low', 'close']].values
        self.x_values = np.zeros(shape=(int(self.num_of_rows_in_csv / self.num_prev_timesteps), self.num_prev_attributes))
        self.y_values = np.zeros(shape=(int(self.num_of_rows_in_csv / self.num_prev_timesteps), self.num_future_attributes))
        dates_df = df[['timestamp']]
        self.plotable_dates = convert_to_matplot_dates(dates_df)
        self.plotable_dates = np.reshape(self.plotable_dates, newshape=(-1, 1)).copy()
        self.plotable_dates_list = []
        print('self.plotable_dates: \n' + str(self.plotable_dates))
        idx = 0
        rows_taken = 0
        # Loop until we have enough remaining rows to add an X-Y pair to our dataset
        while rows_taken <= (self.num_of_rows_in_csv - (self.num_prev_timesteps + self.num_future_timesteps)):
            self.y_values[idx] = self.df_values[rows_taken:rows_taken + self.num_future_timesteps].flatten()
            self.plotable_dates_list.extend(
                self.plotable_dates[rows_taken:rows_taken + self.num_future_timesteps].flatten().tolist())
            rows_taken += self.num_future_timesteps
            self.x_values[idx] = self.df_values[rows_taken:rows_taken + self.num_prev_timesteps].flatten()
            rows_taken += self.num_prev_timesteps
            idx += 1
        # Remove all unfilled rows
        self.y_values = self.y_values[np.all(self.y_values != 0, axis=1)]
        self.x_values = self.x_values[np.all(self.x_values != 0, axis=1)]

        # print('self.df_values: \n' + str(self.df_values))
        # print('self.x_values: \n' + str(self.x_values))
        # print('self.y_values: \n' + str(self.y_values))
        print('self.df_values shape: ' + str(self.df_values.shape))
        print('self.x_values shape: ' + str(self.x_values.shape))
        print('self.y_values shape: ' + str(self.y_values.shape))

        self.data_size = self.x_values.shape[0]
        self.training_set_size = int(0.8 * self.data_size)
        self.test_set_size = int(self.data_size - self.training_set_size)
        print('Training set size: ' + str(self.training_set_size))
        print('Test set size: ' + str(self.test_set_size))

        self.x_tr = self.x_values[self.test_set_size:]
        self.x_te = self.x_values[:self.test_set_size]
        self.y_tr = self.y_values[self.test_set_size:]
        self.y_te = self.y_values[:self.test_set_size]

        # Each row in y contains num_future_timesteps amount of days, hence the multiplication here
        # The list of dates is a 1D list created above
        self.train_dates = self.plotable_dates_list[self.test_set_size * self.num_future_timesteps:]
        self.test_dates = self.plotable_dates_list[:self.test_set_size * self.num_future_timesteps]

        print('x_tr shape: \n' + str(self.x_tr.shape))
        print('x_te shape: \n' + str(self.x_te.shape))
        print('y_tr shape: \n' + str(self.y_tr.shape))
        print('y_te shape: \n' + str(self.y_te.shape))
        # Create Keras NN model to get H
        self.input_shape = (self.num_prev_attributes,)
        self.hidden_layer_name = 'hidden_layer'
        # https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
        self.model = Sequential()
        self.model.add(Dense(self.num_hidden_layer_neurons, activation='linear',
                             name=self.hidden_layer_name, input_shape=self.input_shape, kernel_initializer=rand_init,
                             bias_initializer='ones'))
        self.beta = None

    def create_h(self):
        h = None
        if self.train_w:
            # Train w but do not train "beta"
            temp_hidden_name = 'temp_hidden'
            temp_model = Sequential()
            temp_model.add(Dense(self.num_hidden_layer_neurons, activation='sigmoid', name=temp_hidden_name,
                                 weights=self.model.get_layer(self.hidden_layer_name).get_weights(),
                                 input_shape=self.input_shape))
            temp_model.add(Dense(self.num_future_attributes, use_bias=False, trainable=False))
            temp_model.compile('adam', 'mse', metrics=['mse'])
            temp_model.fit(self.x_tr, self.x_tr, epochs=10, batch_size=32)
            hidden_layer_only_model = Model(inputs=temp_model.input,
                                            outputs=temp_model.get_layer(temp_hidden_name).output)
            h = np.mat(hidden_layer_only_model.predict(self.x_tr))
        else:
            h = np.mat(self.model.predict(self.x_tr))
        print('First weight matrix: \n' + str(self.model.get_layer(self.hidden_layer_name).get_weights()))
        print('Created h with shape: ' + str(h.shape) + '\n And values: \n' + str(h))
        return h

    def train(self):
        H = self.create_h()
        T = np.mat(self.y_tr)
        print('Shape of T: ' + str(T.shape) + '\n And values: \n' + str(T))
        self.beta = np.linalg.pinv(H) * np.mat(T)
        self.model.add(Dense(self.num_future_attributes, use_bias=False, weights=[self.beta]))
        print('Finished training ELM, model summary:')
        self.model.summary()
        print('Beta: \n' + str(self.beta))

    def predict(self):
        last_timeframe_in_data = np.array([self.df_values[0:self.num_prev_timesteps].flatten()])
        print('last_timeframe_in_data values:' + str(last_timeframe_in_data) + ' last_timeframe_in_data shape: '
              + str(last_timeframe_in_data.shape))
        print('Prediction for last day is :' + str(self.model.predict(last_timeframe_in_data)))
        training_pred = self.model.predict(self.x_tr)
        test_pred = self.model.predict(self.x_te)
        return training_pred, test_pred


if __name__ == '__main__':
    disjoint_elm = DisjointDataELMMatrixVersion()
    disjoint_elm.train()
    idx_plotted_feature = 0
    plotted_feature_str = \
        {0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}[idx_plotted_feature]
    print('We will plot: ' + plotted_feature_str)
    num_features = disjoint_elm.num_features
    training_pred, test_pred = disjoint_elm.predict()
    real_tr, real_test = disjoint_elm.y_tr, disjoint_elm.y_te
    # Result of prediction is a 2D matrix,, so we must convert to a 1D list
    plot_y_tr_real = real_tr[:, idx_plotted_feature::num_features].flatten().tolist()[0]
    plot_y_tr_pred = training_pred[:, idx_plotted_feature::num_features].flatten().tolist()[0]
    plot_y_te_real = real_test[:, idx_plotted_feature::num_features].flatten().tolist()[0]
    plot_y_te_pred = test_pred[:, idx_plotted_feature::num_features].flatten().tolist()[0]

    plt.figure(0)
    real_train_graph, = plt.plot_date(disjoint_elm.train_dates, plot_y_tr_real, 'b-',
                                      label='Real training data', color='red')
    pred_train_graph, = plt.plot_date(disjoint_elm.train_dates, plot_y_tr_pred, 'b-',
                                      label='ELM Prediction of training data', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Prediction of ' + plotted_feature_str)
    plt.legend(handles=[real_train_graph, pred_train_graph])
    plt.show()

    plt.figure(1)
    real_test_graph, = plt.plot_date(disjoint_elm.test_dates, plot_y_te_real, 'b-',
                                     label='Real test data', color='red')
    pred_test_graph, = plt.plot_date(disjoint_elm.test_dates, plot_y_te_pred, 'b-',
                                     label='ELM Prediction of test data', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Prediction of ' + plotted_feature_str)
    plt.legend(handles=[real_test_graph, pred_test_graph])
    plt.show()
