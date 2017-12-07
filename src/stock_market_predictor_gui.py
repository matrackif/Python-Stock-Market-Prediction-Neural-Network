import sys
import os
print('Current working directory: ' + os.getcwd())
from src.main import train_and_predict
from resc.smp_gui import Ui_SMPMainWindow
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class StockMarketPredictorMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_SMPMainWindow()
        self.ui.setupUi(self)
        self.ui.trainAndPredictButton.clicked.connect(self.handle_train_and_predict_button)

    def handle_train_and_predict_button(self):
        # TODO run in a separate thread because now it's blocking the UI thread
        # TODO handle select CSV button click
        num_prev_days = self.ui.numPrevDaysEdit.text()
        num_future_days = self.ui.numFutureDaysEdit.text()
        num_hidden_neurons = self.ui.numHiddenNeuronsEdit.text()
        print('num_prev_days: ' + num_prev_days)
        print('num_future_days: ' + num_future_days)
        print('num_hidden_neurons: ' + num_hidden_neurons)
        train_and_predict(num_previous_days=int(num_prev_days), num_future_days=int(num_future_days), num_hidden_neurons=int(num_hidden_neurons))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    smp_window = StockMarketPredictorMainWindow()
    smp_window.show()
    sys.exit(app.exec())