import matplotlib.pyplot as plt
from src.disjoint_data_model import DisjointDataModel
from src.rolling_window_model import RollingWindowModel
from src.lstm_sequence_predictor import LSTMSequencePredictor


def train_and_predict(index_of_plotted_data: int = 0, num_previous_days:int = 10, num_future_days:int = 5, num_hidden_neurons:int = 512):

    plotted_feature_str = {0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}[index_of_plotted_data]
    disjoint_data_model = DisjointDataModel(use_keras=True, index_of_plotted_feature=index_of_plotted_data,
                                            num_of_previous_days=num_previous_days, num_of_future_days=num_future_days)
    disjoint_data_model_elm = DisjointDataModel(use_keras=False, index_of_plotted_feature=index_of_plotted_data,
                                                num_of_previous_days=num_previous_days,
                                                num_of_future_days=num_future_days)
    rolling_window_model = RollingWindowModel(use_keras=True, index_of_plotted_feature=index_of_plotted_data,
                                              num_of_previous_days=num_previous_days,
                                              num_of_future_days=num_future_days)
    lstm = LSTMSequencePredictor(index_of_plotted_feature=index_of_plotted_data, num_of_previous_days=num_previous_days,
                                 num_of_future_days=num_future_days)
    rolling_window_model_elm = RollingWindowModel(use_keras=False, index_of_plotted_feature=index_of_plotted_data,
                                                  num_of_previous_days=num_previous_days,
                                                  num_of_future_days=num_future_days)
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
        handles=[real_training_data, rolling_window_model_train_pred, lstm_train_pred, rolling_window_model_elm_train_pred])
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
    train_and_predict()