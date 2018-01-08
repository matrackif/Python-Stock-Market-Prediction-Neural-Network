import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import utils as utils
import time


class RollingWindowModel:
    """Class represents an ELM or Keras model that predicts X future days given Y days"""
    def __init__(self, csv_file: str = '../data/daily_MSFT.csv', use_keras: bool = False,
                 index_of_plotted_feature: int = 0, num_of_previous_days: int = 7, num_of_future_days: int = 3,
                 num_of_hidden_neurons: int = 256, train_percentage: int = 80, bias_term: int = 1):
        """
        Constructor parses the CSV file and initializes the training and test data

        Parameters
        ----------
        csv_file: str
            The path to the CSV file (from AlphaVantage API) to be parsed
       use_keras: bool
            If true, the model will be a "standard" neural network implemented using the Keras API, if false,
            the model will be an ELM
        index_of_plotted_feature: int
            The class after training and predicting generates an array that can be plotted. This number is between 0 and 4.
            0: Open, 1: High, 2: Low, 3: Close, 4: Volume. Note: 4 (volume) is currently not supported.
        num_of_previous_days: int
            The number of previous days needed in order to make a single prediction
         num_of_future_days: int
            Given num_of_previous_days, we predict num_of_future_days number of future days
         num_of_hidden_neurons: int
            Number of neurons in the hidden layer
        train_percentage: int
            Number between 0 and 100 that represents the percentage of the CSV file data that will be training data
            The remainder will be test data
        bias_term: int = 1
            Column that is prepended to the X data matrix (bias term)
        """
        df = pd.read_csv(csv_file)  # By default header will be read from file
        # print('Head of data frame: \n' + str(df.head()))
        # print('Dimensions of data frame (row x col)' + str(df.shape))
        self.index_of_plotted_feature = index_of_plotted_feature
        self.plotted_feature_str = \
            {0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}[self.index_of_plotted_feature]
        # Stores the bias (an integer)
        self.bias = bias_term
        self.use_keras = use_keras
        self.model_type_str = None
        self.num_of_rows_in_csv = int(df.shape[0])
        # Assume 4 features because volume is ignored
        self.num_features = 4
        self.num_hidden_layer_neurons = num_of_hidden_neurons
        self.num_prev_timesteps = num_of_previous_days
        self.num_future_timesteps = num_of_future_days
        # Stores the amount of columns in a row that are needed to make a future prediction
        self.num_prev_attributes = self.num_prev_timesteps * self.num_features
        # Stores the amount of columns in a row that represent a future prediction
        self.num_future_attributes = self.num_future_timesteps * self.num_features
        # Initialize a NumPy array of values that will store our stock data
        self.df_values = df[['open', 'high', 'low', 'close']].values
        # Frame our data as rolling window series prediction using series_to_supervised() method
        self.reframed_x_y = utils.series_to_supervised(self.df_values, self.num_prev_timesteps,
                                                       self.num_future_timesteps)
        # Convert the Pandas DataFrame to NumPy array
        self.x_y_values = self.reframed_x_y.values
        self.plotable_dates = utils.convert_to_matplot_dates(df)
        self.plotable_dates = np.reshape(self.plotable_dates, newshape=(-1, 1)).copy()
        self.reframed_dates = utils.series_to_supervised(self.plotable_dates, self.num_prev_timesteps,
                                                         self.num_future_timesteps)
        # Initialize variables that will store the Mean Squared Error
        self.mse_train_cost = -1
        self.mse_test_cost = -1
        # All of the variables that have the word "plotable" are standard python lists that contain
        # Values of a specific attribute (for example "Close") as specified by index_of_plotted_feature
        self.plotable_y_train_real = None
        self.plotable_y_train_pred = None
        self.plotable_y_test_real = None
        self.plotable_y_test_pred = None
        self.plotable_y_future_pred = None
        # The amount of rows in our matrix
        self.data_size = self.x_y_values.shape[0]
        # Separate our data into training and test sets
        self.training_set_size = int((train_percentage / 100) * self.data_size)
        self.test_set_size = int(self.data_size - self.training_set_size)
        print('Rolling window model: Training set size: ' + str(self.training_set_size))
        print('Rolling window model: Test set size: ' + str(self.test_set_size))
        # Below we store integers that matplotlib uses for the plot_date() method so we can plot
        # Dates on the X-axis in a neat way
        self.train_dates = self.reframed_dates.values[self.test_set_size:, -1:].flatten()
        self.test_dates = self.reframed_dates.values[:self.test_set_size, -1:].flatten()
        self.model = None

        # In our y, each row contains more than 1 time step worth of data (due to rolling window prediction)
        # However it makes sense only to plot 1 y in each row
        # Therefore we only plot the last prediction of y given a sequence of values
        # For example if num_future_timesteps is 5, then from each row plot the selected feature at t+4
        # (Since then our y would contain predictions for t, t+1, t+2, t+3, t+4)

        # If we predict 2 future days given 1 day, then one row in  self.x_y_values would have the following format:
        # [Open(t-1), High(t-1), Low(t-1), Close(t-1), Open(t), High(t), Low(t), Close(t), Open(t+1), High(t+1), Low(t+1), Close(t+1)]
        # self.num_prev_attributes in this example is 4, because the first 4 values in the row represent the  data needed to predict the future
        # self.num_future_attributes would be 8, because the last 8 values represent the all the future data
        if self.use_keras:
            self.model_type_str = 'Keras'
            self.x_tr = self.x_y_values[self.test_set_size:, :self.num_prev_attributes]
            self.x_te = self.x_y_values[:self.test_set_size, :self.num_prev_attributes]
            self.y_tr = self.x_y_values[self.test_set_size:, -self.num_future_attributes:]
            self.y_te = self.x_y_values[:self.test_set_size, -self.num_future_attributes:]
            self.plotable_y_train_real = self.y_tr[:, -(self.num_features + self.index_of_plotted_feature)].flatten()
            self.plotable_y_test_real = self.y_te[:, -(self.num_features + self.index_of_plotted_feature)].flatten()
            self.input_shape = (self.num_prev_attributes,)
            self.model = Sequential()
            self.model.add(Dense(self.num_hidden_layer_neurons, input_shape=self.input_shape, activation='linear'))
            self.model.add(Dense(self.num_future_attributes))
            self.model.compile(loss='mse', optimizer='adam')
            print('Created a keras model for disjoint X and Y stock data, summary: ')
            self.model.summary()
        else:
            self.model_type_str = 'ELM'
            self.x_tr = np.mat(self.x_y_values[self.test_set_size:, :self.num_prev_attributes])
            self.x_te = np.mat(self.x_y_values[:self.test_set_size, :self.num_prev_attributes])
            self.y_tr = np.mat(self.x_y_values[self.test_set_size:, -self.num_future_attributes:])
            self.y_te = np.mat(self.x_y_values[:self.test_set_size, -self.num_future_attributes:])
            self.plotable_y_train_real = \
                self.y_tr[:, -(self.num_features + self.index_of_plotted_feature)].flatten().tolist()[0]
            self.plotable_y_test_real = \
                self.y_te[:, -(self.num_features + self.index_of_plotted_feature)].flatten().tolist()[0]
            # Add bias term of ones to test X and train X
            self.x_tr = np.concatenate((np.ones(shape=(self.x_tr.shape[0], 1)) * self.bias, self.x_tr), axis=1)
            self.x_te = np.concatenate((np.ones(shape=(self.x_te.shape[0], 1)) * self.bias, self.x_te), axis=1)
            # Our beta will be a weight matrix between the hidden and output layer
            self.input_layer_weights = np.mat(
                utils.rand_init(shape=(self.num_hidden_layer_neurons, self.num_prev_attributes + 1)))
            self.beta = None

    def create_h(self):
        """
        Creates the "H" matrix which represents the hidden layer in the ELM
        This is only executed if use_keras is true
        We take our input matrix X (obtained in our constructor) which is of size m x (n + 1)
        Where m is the number of samples and n is the number of previous attributes (one added for the bias)
        H is obtained by one step of forward propagation, multiplying X by the weights connecting X to the hidden layer

        Returns
        ----------
        numpy.matrixlib.defmatrix.matrix (NumPy Matrix)
            The NumPy matrix H obtained by forward propagation of size m x (num_hidden_layer_neurons)
        """
        h = None
        if not self.use_keras:
            h = self.x_tr * self.input_layer_weights.T
            # print('Rolling window model: created h with shape: ' + str(h.shape) + '\n And values: \n' + str(h))
            print('Rolling window model: created h with shape: ' + str(h.shape))
        return h

    def train(self, plot_history: bool = True):
        """
        If use_keras is true then this method trains the keras model, otherwise we find the output layer of the ELM

        Parameters
        ----------
        plot_history: bool
            If use_keras is true then this bool specifies whether or not to plot the history of errors.
            Keras trains its model in "epochs" where it takes batch_size amount of samples and runs gradient descent
            on them. Keras records the mean squared error obtained in each epoch. If plot_history is true then we
            plot the MSE obtained in each epoch

        """
        # Our model is trained to predict the stock prices at t, t+1, ..., t+n given the prices at t-k, t-k+1, ..., t-1
        # Where n is num_future_timesteps and k is num_prev_timesteps
        start_time = time.time()
        if self.use_keras:
            history = self.model.fit(self.x_tr, self.y_tr, epochs=50, batch_size=50, validation_data=(self.x_te, self.y_te),
                           verbose=2)
            print('Keras Rolling window model: finished training in %s seconds' % (time.time() - start_time))
            # plot history
            if plot_history:
                plt.figure(0)
                plt.plot(history.history['loss'], label='Training loss (error)')
                plt.plot(history.history['val_loss'], label='Test loss (error)')
                plt.title('Mean Squared Error For Keras Rolling Window Model')
                plt.xlabel('Epoch')
                plt.legend()
                plt.show()
        else:
            H = self.create_h()
            T = self.y_tr
            print('Rolling window model: shape of T: ' + str(T.shape))
            # We aim to solve the Equation H * beta = T for beta
            self.beta = np.linalg.pinv(H) * np.mat(T)
            print('Rolling window model: finished training ELM in %s seconds' % (time.time() - start_time))
            # print('Rolling window model: Beta: \n' + str(self.beta))

    def predict_and_plot(self, do_plot: bool = True):
        """
        If use_keras is false then we use the beta obatined by the train() method and
        predict by doing forward propagation. Otherwise we  predict using Keras model
        This method predicts using the training X and the test X, the last row in
        self.x_y_values contains the data we need in order to make a "real" future prediction

        Parameters
        ----------
        do_plot: bool
            If use_keras is true then this bool specifies whether or not to plot the prediction result

        Returns
        ----------
        Tuple[numpy.matrixlib.defmatrix.matrix, numpy.matrixlib.defmatrix.matrix, numpy.matrixlib.defmatrix.matrix]
            A tuple that contains 3 matrices, the first being the training prediction, the second is the test prediction,
            and the third is the future prediction
        """
        training_pred, test_pred = None, None
        future_pred = None

        # last_timeframe_in_data stores the data we need in order to make a "real" future prediction
        last_timeframe_in_data = np.array([self.x_y_values[0, -self.num_prev_attributes:].flatten()])
        # real_train contains the training data of only the variable we wish to plot
        real_train = self.y_tr[:, -(self.num_features + self.index_of_plotted_feature)]
        # real_test contains the training data of only the variable we wish to plot
        real_test = self.y_te[:, -(self.num_features + self.index_of_plotted_feature)]
        if self.use_keras:
            training_pred = self.model.predict(self.x_tr)
            test_pred = self.model.predict(self.x_te)
            future_pred = self.model.predict(last_timeframe_in_data)
            # Because our prediction contains all attributes (high, low, etc..),
            # We must extract the attribute that is plotted from training_pred
            mse_train = training_pred[:, -(self.num_features + self.index_of_plotted_feature)]
            mse_test = test_pred[:, -(self.num_features + self.index_of_plotted_feature)]
            # Apply the MSE formula
            mse_train = np.sum(np.square(np.subtract(real_train, mse_train)))
            mse_test = np.sum(np.square(np.subtract(real_test, mse_test)))

            self.mse_train_cost = mse_train / (self.training_set_size)
            self.mse_test_cost = mse_test / (self.test_set_size)
            print('Rolling window Keras Training Set Mean Squared Error Cost: ' + str(self.mse_train_cost))
            print('Rolling window Keras Test Set Mean Squared Error Cost: ' + str(self.mse_test_cost))
            # Convert 2D numpy array to plotable 1D python list of values
            self.plotable_y_train_pred = training_pred[:,
                                         -(self.num_features + self.index_of_plotted_feature)].flatten()
            self.plotable_y_test_pred = test_pred[:, -(self.num_features + self.index_of_plotted_feature)].flatten()
            # When predicting the future we have num_future_timesteps amount of future days
            self.plotable_y_future_pred = future_pred[:,
                                          self.index_of_plotted_feature::self.num_features].flatten()
        else:
            last_timeframe_in_data = np.array([self.df_values[0:self.num_prev_timesteps].flatten()])
            # Prepend bias term to the row
            last_timeframe_in_data = np.hstack(([[self.bias]], last_timeframe_in_data))
            # Apply the forward propagation algorithm to find the output layer matrix
            future_pred = np.mat(last_timeframe_in_data) * self.input_layer_weights.T * self.beta
            training_pred_h = self.x_tr * self.input_layer_weights.T
            training_pred = training_pred_h * self.beta
            test_pred_h = self.x_te * self.input_layer_weights.T
            test_pred = test_pred_h * self.beta
            # Convert 2D numpy matrix to plotable 1D python list of values

            mse_train = training_pred[:, -(self.num_features + self.index_of_plotted_feature)]
            mse_test = test_pred[:, -(self.num_features + self.index_of_plotted_feature)]

            mse_train = np.sum(np.square(np.subtract(real_train, mse_train)))
            mse_test = np.sum(np.square(np.subtract(real_test, mse_test)))

            self.mse_train_cost = mse_train / (self.training_set_size)
            self.mse_test_cost = mse_test / (self.test_set_size)
            print('Rolling window ELM Training Set Mean Squared Error Cost: ' + str(self.mse_train_cost))
            print('Rolling window ELM Test Set Mean Squared Error Cost: ' + str(self.mse_test_cost))
            # Create the standard Python lists that store the feature we want to plot
            # We have to convert a 2D NumPy Array to a 1D Python array
            self.plotable_y_train_pred = \
                training_pred[:, -(self.num_features + self.index_of_plotted_feature)].flatten().tolist()[0]
            self.plotable_y_test_pred = \
                test_pred[:, -(self.num_features + self.index_of_plotted_feature)].flatten().tolist()[0]
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
            future_graph, = plt.plot(range(1, len(self.plotable_y_future_pred) + 1), self.plotable_y_future_pred, 'b-',
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
    rolling_window_model = RollingWindowModel(use_keras=True)
    rolling_window_model.train()
    rolling_window_model.predict_and_plot()
