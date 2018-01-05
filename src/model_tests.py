import matplotlib.pyplot as plt
from disjoint_data_model import DisjointDataModel
from rolling_window_model import RollingWindowModel
from lstm_sequence_predictor import LSTMSequencePredictor
from main import train_and_predict
import argparse


def run_neuron_count_test(neuron_count_step: int = 25, max_neuron_count: int = 1024, plotted_feature_str: str = 'Open',
                          num_previous_days: int = 10, num_future_days: int = 5,
                          init_num_hidden_neurons: int = 512, data_source_file: str = '../data/daily_MSFT.csv',
                          use_keras: bool = False, bias_term: int = 1, train_percentage: int = 80):
    num_hidden_neurons = init_num_hidden_neurons
    csv_file_path = data_source_file
    rolling_window_elm_train_errors = []
    rolling_window_elm_test_errors = []
    keras_rolling_window_train_errors = []
    keras_rolling_window_test_errors = []
    lstm_train_errors = []
    lstm_test_errors = []
    neuron_counts = []
    while num_hidden_neurons < max_neuron_count:
        neuron_counts.append(num_hidden_neurons)
        return_val = train_and_predict(num_hidden_neurons=num_hidden_neurons, data_source_file=csv_file_path,
                                       num_previous_days=num_previous_days, num_future_days=num_future_days,
                                       plotted_feature_str=plotted_feature_str,
                                       train_percentage=train_percentage, bias_term=bias_term, use_keras=use_keras,
                                       do_plot=False)
        if use_keras:
            rolling_window_elm_train_errors.append(return_val[0])
            rolling_window_elm_test_errors.append(return_val[1])
            keras_rolling_window_train_errors.append(return_val[2])
            keras_rolling_window_test_errors.append(return_val[3])
            lstm_train_errors.append(return_val[4])
            lstm_test_errors.append(return_val[5])
        else:
            rolling_window_elm_train_errors.append(return_val[0])
            rolling_window_elm_test_errors.append(return_val[1])

        num_hidden_neurons += neuron_count_step
    plt.title('Error analysis with varying number of hidden layer neuron count')
    plt.xlabel('Number of hidden Neurons')
    plt.ylabel('Mean Squared Error Cost')
    elm_train_err_plt, = plt.plot(neuron_counts, rolling_window_elm_train_errors, label='ELM Train')
    elm_test_err_plt, = plt.plot(neuron_counts, rolling_window_elm_test_errors, label='ELM Test')
    plt.legend(
        handles=[elm_train_err_plt, elm_test_err_plt])
    plt.show()


def run_prev_days_test(prev_days_step: int = 1, max_prev_days: int = 30, plotted_feature_str: str = 'Open',
                       init_num_previous_days: int = 1, num_future_days: int = 5,
                       num_hidden_neurons: int = 512, data_source_file: str = '../data/daily_MSFT.csv',
                       use_keras: bool = False, bias_term: int = 1, train_percentage: int = 80):
    num_previous_days = init_num_previous_days
    csv_file_path = data_source_file
    rolling_window_elm_train_errors = []
    rolling_window_elm_test_errors = []
    keras_rolling_window_train_errors = []
    keras_rolling_window_test_errors = []
    lstm_train_errors = []
    lstm_test_errors = []
    prev_days_list = []
    while num_previous_days < max_prev_days:
        prev_days_list.append(num_previous_days)
        return_val = train_and_predict(num_hidden_neurons=num_hidden_neurons, data_source_file=csv_file_path,
                                       num_previous_days=num_previous_days, num_future_days=num_future_days,
                                       plotted_feature_str=plotted_feature_str,
                                       train_percentage=train_percentage, bias_term=bias_term, use_keras=use_keras,
                                       do_plot=False)
        if use_keras:
            rolling_window_elm_train_errors.append(return_val[0])
            rolling_window_elm_test_errors.append(return_val[1])
            keras_rolling_window_train_errors.append(return_val[2])
            keras_rolling_window_test_errors.append(return_val[3])
            lstm_train_errors.append(return_val[4])
            lstm_test_errors.append(return_val[5])
        else:
            rolling_window_elm_train_errors.append(return_val[0])
            rolling_window_elm_test_errors.append(return_val[1])

        num_previous_days += prev_days_step
    plt.title('Error analysis with varying number of previous days')
    plt.xlabel('Number of previous days')
    plt.ylabel('Mean Squared Error Cost')
    elm_train_err_plt, = plt.plot(prev_days_list, rolling_window_elm_train_errors, label='ELM Train')
    elm_test_err_plt, = plt.plot(prev_days_list, rolling_window_elm_test_errors, label='ELM Test')
    plt.legend(
        handles=[elm_train_err_plt, elm_test_err_plt])
    plt.show()


def run_future_days_test(future_days_step: int = 1, max_future_days: int = 30, plotted_feature_str: str = 'Open',
                         num_previous_days: int = 7, init_num_future_days: int = 1,
                         num_hidden_neurons: int = 512, data_source_file: str = '../data/daily_MSFT.csv',
                         use_keras: bool = False, bias_term: int = 1, train_percentage: int = 80):
    num_future_days = init_num_future_days
    csv_file_path = data_source_file
    rolling_window_elm_train_errors = []
    rolling_window_elm_test_errors = []
    keras_rolling_window_train_errors = []
    keras_rolling_window_test_errors = []
    lstm_train_errors = []
    lstm_test_errors = []
    future_days_list = []
    while num_future_days < max_future_days:
        future_days_list.append(num_future_days)
        return_val = train_and_predict(num_hidden_neurons=num_hidden_neurons, data_source_file=csv_file_path,
                                       num_previous_days=num_previous_days, num_future_days=num_future_days,
                                       plotted_feature_str=plotted_feature_str,
                                       train_percentage=train_percentage, bias_term=bias_term, use_keras=use_keras,
                                       do_plot=False)
        if use_keras:
            rolling_window_elm_train_errors.append(return_val[0])
            rolling_window_elm_test_errors.append(return_val[1])
            keras_rolling_window_train_errors.append(return_val[2])
            keras_rolling_window_test_errors.append(return_val[3])
            lstm_train_errors.append(return_val[4])
            lstm_test_errors.append(return_val[5])
        else:
            rolling_window_elm_train_errors.append(return_val[0])
            rolling_window_elm_test_errors.append(return_val[1])

            num_future_days += future_days_step
    plt.title('Error analysis with varying number of future days')
    plt.xlabel('Number of future days')
    plt.ylabel('Mean Squared Error Cost')
    elm_train_err_plt, = plt.plot(future_days_list, rolling_window_elm_train_errors, label='ELM Train')
    elm_test_err_plt, = plt.plot(future_days_list, rolling_window_elm_test_errors, label='ELM Test')
    plt.legend(
        handles=[elm_train_err_plt, elm_test_err_plt])
    plt.show()


if __name__ == '__main__':
    run_neuron_count_test(use_keras=False, neuron_count_step=1, init_num_hidden_neurons=1, max_neuron_count=40)
    run_prev_days_test(use_keras=False, prev_days_step=1, init_num_previous_days=1, max_prev_days=30)
    run_future_days_test(use_keras=False, future_days_step=1, init_num_future_days=1, max_future_days=30)
