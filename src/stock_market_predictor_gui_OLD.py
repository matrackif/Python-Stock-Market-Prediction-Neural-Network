import sys
import os
from queue import Queue
from main import train_and_predict
from resc.smp_gui import Ui_SMPMainWindow
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from disjoint_data_model import DisjointDataModel
from rolling_window_model import RollingWindowModel
from lstm_sequence_predictor import LSTMSequencePredictor


# The new Stream Object which replaces the default stream associated with sys.stdout
# This object just puts data in a queue!
class WriteStream(object):
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)


# A QObject (to be run in a QThread) which sits waiting for data to come through a Queue.Queue().
# It blocks until data is available, and one it has got something from the queue, it sends
# it to the "MainThread" by emitting a Qt Signal
class DataReceiver(QObject):

    guiSignal = pyqtSignal(str)

    def __init__(self, queue, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.queue = queue

    @pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.guiSignal.emit(text)


class PredictorThread(QThread):
    def __init__(self, num_of_prev_days: int = 7, num_of_future_days: int = 3, num_of_hidden_neurons: int = 256,
                 csv_data_file: str = '../data/daily_MSFT.csv'):
        QThread.__init__(self)
        self.num_prev_days = num_of_prev_days
        self.num_of_future_days = num_of_future_days
        self.num_hidden_neurons = num_of_hidden_neurons
        self.csv_file_path = csv_data_file

    def __del__(self):
        self.quit()

    def run(self):
        index_of_plotted_data = 0
        plotted_feature_str = {0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}[index_of_plotted_data]
        disjoint_data_model = DisjointDataModel(use_keras=True, index_of_plotted_feature=index_of_plotted_data,
                                                num_of_previous_days=self.num_prev_days,
                                                num_of_future_days=self.num_of_future_days,
                                                num_of_hidden_neurons=self.num_hidden_neurons,
                                                csv_file=self.csv_file_path)
        disjoint_data_model_elm = DisjointDataModel(use_keras=False, index_of_plotted_feature=index_of_plotted_data,
                                                    num_of_previous_days=self.num_prev_days,
                                                    num_of_future_days=self.num_of_future_days,
                                                    num_of_hidden_neurons=self.num_hidden_neurons,
                                                    csv_file=self.csv_file_path)
        rolling_window_model = RollingWindowModel(use_keras=True, index_of_plotted_feature=index_of_plotted_data,
                                                  num_of_previous_days=self.num_prev_days,
                                                  num_of_future_days=self.num_of_future_days,
                                                  num_of_hidden_neurons=self.num_hidden_neurons,
                                                  csv_file=self.csv_file_path)
        lstm = LSTMSequencePredictor(index_of_plotted_feature=index_of_plotted_data,
                                     num_of_previous_days=self.num_prev_days,
                                     num_of_future_days=self.num_of_future_days,
                                     num_of_hidden_neurons=self.num_hidden_neurons,
                                     csv_file=self.csv_file_path)
        rolling_window_model_elm = RollingWindowModel(use_keras=False, index_of_plotted_feature=index_of_plotted_data,
                                                      num_of_previous_days=self.num_prev_days,
                                                      num_of_future_days=self.num_of_future_days,
                                                      num_of_hidden_neurons=self.num_hidden_neurons,
                                                      csv_file=self.csv_file_path)
        disjoint_data_model.train()
        disjoint_data_model_elm.train()
        rolling_window_model.train()
        lstm.train(plot_history=False)
        rolling_window_model_elm.train()

        disjoint_data_model.predict_and_plot(do_plot=False)
        disjoint_data_model_elm.predict_and_plot(do_plot=True)
        rolling_window_model.predict_and_plot(do_plot=False)
        lstm.predict_all_and_plot(do_plot=True)
        rolling_window_model_elm.predict_and_plot(do_plot=True)

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
            handles=[future_disjoint_data_model, future_disjoint_data_model_elm, future_rolling_window_model,
                     future_lstm,
                     future_rolling_window_model_elm])
        plt.show()
        # train_and_predict(num_previous_days=int(self.num_prev_days), num_future_days=int(self.num_future_days),
        #                   num_hidden_neurons=int(self.num_hidden_neurons), data_source_file=self.csv_file_path)


class StockMarketPredictorMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_SMPMainWindow()
        self.ui.setupUi(self)
        self.ui.trainAndPredictButton.clicked.connect(self.handle_train_and_predict_button)
        self.predictor_runnable = None
        self.worker_thread = None

    @pyqtSlot(str)
    def append_text(self, text):
        self.ui.stdOutputTextEdit.moveCursor(QTextCursor.End)
        self.ui.stdOutputTextEdit.insertPlainText(text)

    @pyqtSlot()
    def handle_train_and_predict_button(self):
        # TODO run in a separate thread because now it's blocking the UI thread
        # TODO handle select CSV button click
        num_prev_days = self.ui.numPrevDaysEdit.text()
        num_future_days = self.ui.numFutureDaysEdit.text()
        num_hidden_neurons = self.ui.numHiddenNeuronsEdit.text()
        csv_file_path = self.ui.csvPathEdit.text()
        print('num_prev_days: ' + num_prev_days)
        print('num_future_days: ' + num_future_days)
        print('num_hidden_neurons: ' + num_hidden_neurons)
        print('csv_file_path: ' + csv_file_path)
        if self.worker_thread is not None:
            print('worker thread is finished: ' + str(self.worker_thread.isFinished()))
            # self.worker_thread.__del__()
        self.worker_thread = PredictorThread(num_of_prev_days=int(num_prev_days),
                                             num_of_future_days=int(num_future_days),
                                             num_of_hidden_neurons=int(num_hidden_neurons))
        self.worker_thread.started.connect(lambda: self.ui.trainAndPredictButton.setEnabled(False))
        self.worker_thread.finished.connect(lambda: self.ui.trainAndPredictButton.setEnabled(True))
        self.worker_thread.start()


if __name__ == '__main__':
    # Create Queue and redirect sys.stdout to this queue
    standard_output_queue = Queue()
    sys.stdout = WriteStream(queue=standard_output_queue)

    app = QApplication(sys.argv)
    smp_window = StockMarketPredictorMainWindow()
    smp_window.show()

    # Create thread that will listen on the other end of the queue, and send the text to the textedit in our application
    gui_updater_thread = QThread()
    data_receiver = DataReceiver(queue=standard_output_queue)
    data_receiver.guiSignal.connect(smp_window.append_text)
    data_receiver.moveToThread(gui_updater_thread)
    gui_updater_thread.started.connect(data_receiver.run)
    gui_updater_thread.start()

    print('Current working directory: ' + os.getcwd())
    app.exec()