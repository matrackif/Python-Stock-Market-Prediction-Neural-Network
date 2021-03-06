import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import utils as utils
import time


class DisjointDataModel:
    def __init__(self, csv_file: str = '../data/daily_MSFT.csv', use_keras: bool = False,
                 index_of_plotted_feature: int = 0, num_of_previous_days: int = 7, num_of_future_days: int = 3,
                 num_of_hidden_neurons: int = 256, train_percentage: int = 80, bias_term: int = 1):
        df = pd.read_csv(csv_file)  # By default header will be read from file
        print('Head of data frame: \n' + str(df.head()))
        print('Dimensions of data frame (row x col)' + str(df.shape))
        self.index_of_plotted_feature = index_of_plotted_feature
        self.plotted_feature_str = \
            {0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}[self.index_of_plotted_feature]
        # print('Model will plot: ' + self.plotted_feature_str)
        self.bias = bias_term
        self.use_keras = use_keras
        self.model_type_str = None
        self.num_of_rows_in_csv = int(df.shape[0])
        self.num_features = 4
        self.num_hidden_layer_neurons = num_of_hidden_neurons
        self.num_prev_timesteps = num_of_previous_days
        self.num_future_timesteps = num_of_future_days
        self.num_prev_attributes = self.num_prev_timesteps * self.num_features
        self.num_future_attributes = self.num_future_timesteps * self.num_features
        self.df_values = df[['open', 'high', 'low', 'close']].values
        self.x_values = np.zeros(
            shape=(int(self.num_of_rows_in_csv / self.num_prev_timesteps), self.num_prev_attributes))
        self.y_values = np.zeros(
            shape=(int(self.num_of_rows_in_csv / self.num_prev_timesteps), self.num_future_attributes))

        self.plotable_dates = utils.convert_to_matplot_dates(df)
        self.plotable_dates = np.reshape(self.plotable_dates, newshape=(-1, 1)).copy()
        self.plotable_dates_list = []
        self.plotable_y_train_real = None
        self.plotable_y_train_pred = None
        self.plotable_y_test_real = None
        self.plotable_y_test_pred = None
        self.plotable_y_future_pred = None
        # print('self.plotable_dates: \n' + str(self.plotable_dates) + str(type(self.plotable_dates)))
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
        # print('self.plotable_dates_list: \n' + str(self.plotable_dates_list) + str(len(self.plotable_dates_list)))
        # Remove all unfilled rows
        self.y_values = self.y_values[np.all(self.y_values != 0, axis=1)]
        self.x_values = self.x_values[np.all(self.x_values != 0, axis=1)]

        # print('self.df_values: \n' + str(self.df_values))
        # print('self.x_values: \n' + str(self.x_values))
        # print('self.y_values: \n' + str(self.y_values))
        # print('self.df_values shape: ' + str(self.df_values.shape))
        # print('self.x_values shape: ' + str(self.x_values.shape))
        # print('self.y_values shape: ' + str(self.y_values.shape))

        self.data_size = self.x_values.shape[0]
        self.training_set_size = int((train_percentage / 100) * self.data_size)
        self.test_set_size = int(self.data_size - self.training_set_size)
        print('Disjoint data model training set size: ' + str(self.training_set_size))
        print('Test set size: ' + str(self.test_set_size))
        self.model = None

        if self.use_keras:
            self.model_type_str = 'Keras'
            self.x_tr = self.x_values[self.test_set_size:]
            self.x_te = self.x_values[:self.test_set_size]
            self.y_tr = self.y_values[self.test_set_size:]
            self.y_te = self.y_values[:self.test_set_size]
            self.plotable_y_train_real = self.y_tr[:, self.index_of_plotted_feature::self.num_features].flatten()
            self.plotable_y_test_real = self.y_te[:, self.index_of_plotted_feature::self.num_features].flatten()
            self.input_shape = (self.num_prev_attributes,)
            self.model = Sequential()
            self.model.add(Dense(self.num_hidden_layer_neurons, input_shape=self.input_shape, activation='linear'))
            self.model.add(Dense(self.num_future_attributes))
            self.model.compile(loss='mse', optimizer='adam')
            print('Created a keras model for disjoint X and Y stock data, summary: ')
            self.model.summary()
        else:
            self.model_type_str = 'ELM'
            self.x_tr = np.mat(self.x_values[self.test_set_size:])
            self.x_te = np.mat(self.x_values[:self.test_set_size])
            self.y_tr = np.mat(self.y_values[self.test_set_size:])
            self.y_te = np.mat(self.y_values[:self.test_set_size])
            self.plotable_y_train_real = \
                self.y_tr[:, self.index_of_plotted_feature::self.num_features].flatten().tolist()[0]
            self.plotable_y_test_real = \
                self.y_te[:, self.index_of_plotted_feature::self.num_features].flatten().tolist()[0]
            # Add bias term of ones to test X and train X
            self.x_tr = np.concatenate((np.ones(shape=(self.x_tr.shape[0], 1)) * self.bias, self.x_tr), axis=1)
            self.x_te = np.concatenate((np.ones(shape=(self.x_te.shape[0], 1)) * self.bias, self.x_te), axis=1)
            # Our beta will be a weight matrix between the hidden and output layer
            self.input_layer_weights = np.mat(
                utils.rand_init(shape=(self.num_hidden_layer_neurons, self.num_prev_attributes + 1)))
            self.beta = None

        # Each row in y contains num_future_timesteps amount of days, hence the multiplication here
        # The list of dates is a 1D list created above
        self.train_dates = self.plotable_dates_list[self.test_set_size * self.num_future_timesteps:]
        self.test_dates = self.plotable_dates_list[:self.test_set_size * self.num_future_timesteps]

        # print('x_tr shape: \n' + str(self.x_tr.shape))
        # print('x_te shape: \n' + str(self.x_te.shape))
        # print('y_tr shape: \n' + str(self.y_tr.shape))
        # print('y_te shape: \n' + str(self.y_te.shape))

    def create_h(self):
        h = None
        if not self.use_keras:
            h = self.x_tr * self.input_layer_weights.T
            print('Disjoint data model: created h with shape: ' + str(h.shape))
            # + '\n And values: \n' + str(h))
        return h

    def train(self):
        # Our model is trained to predict the stock prices at t+n, t+n-1, ..., t given the prices at t-1, t-2, ..., t-k
        # Where n is num_future_timesteps and k is num_prev_timesteps
        start_time = time.time()
        if self.use_keras:
            self.model.fit(self.x_tr, self.y_tr, epochs=50, batch_size=16, validation_data=(self.x_te, self.y_te),
                           verbose=2)
            print('Keras disjoint data model: finished training in %s seconds' % (time.time() - start_time))
        else:
            H = self.create_h()
            T = self.y_tr
            print('Disjoint data model: Shape of T: ' + str(T.shape))
            # + '\n And values: \n' + str(T))
            self.beta = np.linalg.pinv(H) * np.mat(T)
            print('Disjoint data model: Finished training ELM in %s seconds' % (time.time() - start_time))
            # print('Disjoint data model: Beta: \n' + str(self.beta))

    def predict_and_plot(self, do_plot: bool = True):
        training_pred, test_pred = None, None
        future_pred = None
        last_timeframe_in_data = np.array([self.df_values[0:self.num_prev_timesteps].flatten()])
        # print('Disjoint data model: last_timeframe_in_data values:' + str(last_timeframe_in_data) + ' last_timeframe_in_data shape: '
        #       + str(last_timeframe_in_data.shape))
        if self.use_keras:
            training_pred = self.model.predict(self.x_tr)
            test_pred = self.model.predict(self.x_te)
            future_pred = self.model.predict(last_timeframe_in_data)
            # Convert 2D numpy array to plotable 1D python list of values
            self.plotable_y_train_pred = training_pred[:, self.index_of_plotted_feature::self.num_features].flatten()
            self.plotable_y_test_pred = test_pred[:, self.index_of_plotted_feature::self.num_features].flatten()
            self.plotable_y_future_pred = future_pred[:,
                                          self.index_of_plotted_feature::self.num_features].flatten()
        else:
            last_timeframe_in_data = np.array([self.df_values[0:self.num_prev_timesteps].flatten()])
            last_timeframe_in_data = np.hstack(([[1]], last_timeframe_in_data))
            future_pred = np.mat(last_timeframe_in_data) * self.input_layer_weights.T * self.beta
            training_pred_h = self.x_tr * self.input_layer_weights.T
            training_pred = training_pred_h * self.beta
            test_pred_h = self.x_te * self.input_layer_weights.T
            test_pred = test_pred_h * self.beta
            # Convert 2D numpy matrix to plotable 1D python list of values
            self.plotable_y_train_pred = \
                training_pred[:, self.index_of_plotted_feature::self.num_features].flatten().tolist()[0]
            self.plotable_y_test_pred = \
                test_pred[:, self.index_of_plotted_feature::self.num_features].flatten().tolist()[0]
            self.plotable_y_future_pred = future_pred[:,
                                          self.index_of_plotted_feature::self.num_features].flatten().tolist()[0]

        if do_plot:
            # Plot real training data vs. the prediction of training data
            plt.figure(0)
            real_train_graph, = plt.plot_date(self.train_dates, self.plotable_y_train_real, 'b-',
                                              label='Real training data', color='red')
            pred_train_graph, = plt.plot_date(self.train_dates, self.plotable_y_train_pred, 'b-',
                                              label=self.model_type_str + ' prediction of training data', color='blue')
            plt.xlabel('Date')
            plt.ylabel('Prediction of ' + self.plotted_feature_str)
            plt.legend(handles=[real_train_graph, pred_train_graph])
            plt.show()

            # Plot real training data vs. the prediction of training data
            plt.figure(1)
            real_test_graph, = plt.plot_date(self.test_dates, self.plotable_y_test_real, 'b-',
                                             label='Real test data', color='red')
            pred_test_graph, = plt.plot_date(self.test_dates, self.plotable_y_test_pred, 'b-',
                                             label=self.model_type_str + ' prediction of test data', color='blue')
            plt.xlabel('Date')
            plt.ylabel('Prediction of ' + self.plotted_feature_str)
            plt.legend(handles=[real_test_graph, pred_test_graph])
            plt.show()

            # Plot the future prediction
            plt.figure(2)
            # We reverse the array plotable_y_future_pred because our prediction is in decreasing order
            # So when reversing we get increasing order
            # When we do plot_date that step is unnecessary
            # ..because the plot_date() function automatically formats the dates to be in increasing order
            future_graph, = plt.plot(range(1, len(self.plotable_y_future_pred) + 1), self.plotable_y_future_pred[::-1],
                                     'b-',
                                     label=self.model_type_str + ' future prediction', color='red')
            plt.xlabel('Number of days since last day in data')
            plt.ylabel('Prediction of ' + self.plotted_feature_str)
            plt.legend(handles=[future_graph])
            plt.show()

        # print('Found ' + self.model_type_str + ' prediction for training_pred, shape is: ' + str(training_pred.shape),
        #       ' and type: ' + str(type(training_pred)))
        # print('Found ' + self.model_type_str + ' prediction for test_pred, shape is: ' + str(test_pred.shape),
        #       ' and type: ' + str(type(test_pred)))
        # print('Found ' + self.model_type_str + ' prediction for future time frame, shape is: ' + str(future_pred.shape),
        #       ' and type: ' + str(type(future_pred)))
        # print('Found ' + self.model_type_str + ' prediction for future time frame, values are: ' + str(
        #     self.plotable_y_future_pred),
        #       ' and len: ' + str(len(self.plotable_y_future_pred)))

        return training_pred, test_pred, future_pred


if __name__ == '__main__':
    disjoint_model = DisjointDataModel(use_keras=True, index_of_plotted_feature=1)
    disjoint_model.train()
    disjoint_model.predict_and_plot(do_plot=True)
