import matplotlib.pyplot as plt
from disjoint_data_model import DisjointDataModel
from rolling_window_model import RollingWindowModel
from lstm_sequence_predictor import LSTMSequencePredictor
import argparse


def train_and_predict(plotted_feature_str: str = 'Open', num_previous_days: int = 10, num_future_days: int = 5,
                      num_hidden_neurons: int = 512, data_source_file: str = '../data/daily_MSFT.csv'):
    plotted_feature_str = plotted_feature_str.lower()
    index_of_plotted_data = {'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 4}[plotted_feature_str]
    disjoint_data_model = DisjointDataModel(use_keras=True, index_of_plotted_feature=index_of_plotted_data,
                                            num_of_previous_days=num_previous_days, num_of_future_days=num_future_days,
                                            num_of_hidden_neurons=num_hidden_neurons, csv_file=data_source_file)
    disjoint_data_model_elm = DisjointDataModel(use_keras=False, index_of_plotted_feature=index_of_plotted_data,
                                                num_of_previous_days=num_previous_days,
                                                num_of_future_days=num_future_days,
                                                num_of_hidden_neurons=num_hidden_neurons, csv_file=data_source_file)
    rolling_window_model = RollingWindowModel(use_keras=True, index_of_plotted_feature=index_of_plotted_data,
                                              num_of_previous_days=num_previous_days,
                                              num_of_future_days=num_future_days,
                                              num_of_hidden_neurons=num_hidden_neurons, csv_file=data_source_file)
    lstm = LSTMSequencePredictor(index_of_plotted_feature=index_of_plotted_data, num_of_previous_days=num_previous_days,
                                 num_of_future_days=num_future_days, num_of_hidden_neurons=num_hidden_neurons,
                                 csv_file=data_source_file)
    rolling_window_model_elm = RollingWindowModel(use_keras=False, index_of_plotted_feature=index_of_plotted_data,
                                                  num_of_previous_days=num_previous_days,
                                                  num_of_future_days=num_future_days,
                                                  num_of_hidden_neurons=num_hidden_neurons, csv_file=data_source_file)
    disjoint_data_model.train()
    disjoint_data_model_elm.train()
    rolling_window_model.train()
    lstm.train()
    rolling_window_model_elm.train()

    disjoint_data_model.predict_and_plot(do_plot=False)
    disjoint_data_model_elm.predict_and_plot(do_plot=False)
    rolling_window_model.predict_and_plot(do_plot=False)
    lstm.predict_all_and_plot(do_plot=False)
    rolling_window_model_elm.predict_and_plot(do_plot=False)

    plt.figure(0)
    real_training_data, = plt.plot_date(rolling_window_model.train_dates,
                                        rolling_window_model.plotable_y_train_real, 'b-',
                                        label='Real training data', color='pink')
    rolling_window_model_train_pred, = plt.plot_date(rolling_window_model.train_dates,
                                                     rolling_window_model.plotable_y_train_pred, 'b-',
                                                     label='Keras rolling window train prediction', color='blue')
    lstm_train_pred, = plt.plot_date(lstm.train_dates,
                                     lstm.plotable_y_train_pred, 'b-',
                                     label='Keras LSTM train prediction', color='orange')
    rolling_window_model_elm_train_pred, = plt.plot_date(rolling_window_model_elm.train_dates,
                                                         rolling_window_model_elm.plotable_y_train_pred, 'b-',
                                                         label='ELM rolling window train prediction', color='green')
    plt.xlabel('Date')
    plt.ylabel('Prediction of ' + plotted_feature_str)
    plt.legend(
        handles=[real_training_data, rolling_window_model_train_pred, lstm_train_pred,
                 rolling_window_model_elm_train_pred])
    plt.show()

    plt.figure(1)
    real_test_data, = plt.plot_date(rolling_window_model.test_dates,
                                    rolling_window_model.plotable_y_test_real, 'b-',
                                    label='Real test data', color='pink')
    rolling_window_model_test_pred, = plt.plot_date(rolling_window_model.test_dates,
                                                    rolling_window_model.plotable_y_test_pred, 'b-',
                                                    label='Keras rolling window test prediction', color='blue')
    lstm_test_pred, = plt.plot_date(lstm.test_dates,
                                    lstm.plotable_y_test_pred, 'b-',
                                    label='Keras LSTM test prediction', color='orange')
    rolling_window_model_elm_test_pred, = plt.plot_date(rolling_window_model_elm.test_dates,
                                                        rolling_window_model_elm.plotable_y_test_pred, 'b-',
                                                        label='ELM rolling window test prediction', color='green')
    plt.xlabel('Date')
    plt.ylabel('Prediction of ' + plotted_feature_str)
    plt.legend(
        handles=[real_test_data, rolling_window_model_test_pred, lstm_test_pred,
                 rolling_window_model_elm_test_pred])
    plt.show()

    plt.figure(2)
    future_disjoint_data_model, = plt.plot(range(1, len(disjoint_data_model.plotable_y_future_pred) + 1),
                                           disjoint_data_model.plotable_y_future_pred, 'b-',
                                           label='Keras disjoint data future prediction', color='red')
    future_disjoint_data_model_elm, = plt.plot(range(1, len(disjoint_data_model_elm.plotable_y_future_pred) + 1),
                                               disjoint_data_model_elm.plotable_y_future_pred, 'b-',
                                               label='ELM disjoint data future prediction', color='pink')
    future_rolling_window_model, = plt.plot(range(1, len(rolling_window_model.plotable_y_future_pred) + 1),
                                            rolling_window_model.plotable_y_future_pred, 'b-',
                                            label='Keras rolling window future prediction', color='blue')
    future_lstm, = plt.plot(range(1, len(lstm.plotable_y_future_pred) + 1),
                            lstm.plotable_y_future_pred, 'b-',
                            label='Keras LSTM future prediction', color='orange')
    future_rolling_window_model_elm, = plt.plot(range(1, len(rolling_window_model_elm.plotable_y_future_pred) + 1),
                                                rolling_window_model_elm.plotable_y_future_pred, 'b-',
                                                label='ELM rolling window future prediction', color='green')
    plt.xlabel('Number of days since last day in data')
    plt.ylabel('Prediction of ' + plotted_feature_str)
    plt.legend(
        handles=[future_disjoint_data_model, future_disjoint_data_model_elm, future_rolling_window_model, future_lstm,
                 future_rolling_window_model_elm])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # The argument is of the form -f or --file.
    # If -f or --file is given... for ex: "main.py -f" but no file is given then the "const" argument specifies the file
    # If no -f or --file option is given at all then the "default" argument specifies the file
    parser.add_argument('-f', '--file', nargs='?', type=str, default='../data/daily_MSFT.csv',
                        const='../data/daily_MSFT.csv', help='Path to input CSV file to be read')
    parser.add_argument('-hc', '--hidden-count', nargs='?', type=int, default=256,
                        const=256, help='Number of hidden layer neurons')
    parser.add_argument('-pd', '--previous-days', nargs='?', type=int, default=14,
                        const=14, help='For a given prediction use this many previous days')
    parser.add_argument('-fd', '--future-days', nargs='?', type=int, default=3,
                        const=3, help='For a given amount of previous days predict a sequence of this many future days')
    parser.add_argument('-plot', '--plotted-feature', nargs='?', type=str, default='Open',
                        const='Open', help='The feature to be plotted')
    program_args = vars(parser.parse_args())
    print('Program args: ' + str(program_args))
    num_prev_days = program_args['previous_days']
    num_future_days = program_args['future_days']
    num_hidden_neurons = program_args['hidden_count']
    csv_file_path = program_args['file']
    plotted_feature = program_args['plotted_feature']

    train_and_predict(num_hidden_neurons=num_hidden_neurons, data_source_file=csv_file_path, num_previous_days=num_prev_days, plotted_feature_str=plotted_feature)
