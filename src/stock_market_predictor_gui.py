import sys
import os
# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# print('parentdir: ' + parentdir)
# os.sys.path.insert(0, parentdir + '/resc')
from smp_gui import Ui_SMPMainWindow
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class StockMarketPredictorMainWindow(QMainWindow):
    """Class that inherits from QMainWindow and handles the logic and interacting with the GUI"""
    def __init__(self):
        super().__init__()
        self.ui = Ui_SMPMainWindow()
        self.ui.setupUi(self)
        self.ui.trainAndPredictButton.clicked.connect(self.handle_train_and_predict_button)
        self.ui.csvButton.clicked.connect(self.handle_csv_button)
        self.command = 'py -3 main.py'
        self.predicting_process = QProcess()
        self.predicting_process.readyReadStandardOutput.connect(self.stdout_ready)
        self.predicting_process.readyReadStandardError.connect(self.stderr_ready)
        self.predicting_process.started.connect(lambda: self.ui.trainAndPredictButton.setEnabled(False))
        self.predicting_process.started.connect(lambda: self.ui.stopButton.setEnabled(True))
        self.predicting_process.finished.connect(lambda: self.ui.trainAndPredictButton.setEnabled(True))
        self.predicting_process.finished.connect(lambda: self.ui.stopButton.setEnabled(False))
        self.ui.stopButton.clicked.connect(self.handle_kill_button)
        self.ui.stopButton.setEnabled(False)

    def handle_train_and_predict_button(self):
        """
        When the "Train and Predict" button is pressed we execute a QProcess that runs our main python script
        With our specific arguments passed from the GUI
        """
        num_prev_days = self.ui.numPrevDaysEdit.text()
        num_future_days = self.ui.numFutureDaysEdit.text()
        num_hidden_neurons = self.ui.numHiddenNeuronsEdit.text()
        csv_file_path = self.ui.csvPathEdit.text()
        use_keras = str(self.ui.useKerasCheckBox.isChecked())
        bias = self.ui.biasLineEdit.text()
        train_percentage = self.ui.trainingPercentageEdit.text()
        plotted_feature = self.ui.plottedFeatureComboBox.currentText()
        print('num_prev_days: ' + num_prev_days)
        print('num_future_days: ' + num_future_days)
        print('num_hidden_neurons: ' + num_hidden_neurons)
        print('csv_file_path: ' + csv_file_path)
        print('bias: ' + bias)
        print('train_percentage: ' + train_percentage)
        print('plotted_feature: ' + plotted_feature)
        print('use_keras: ' + use_keras)
        python_interpreter = 'py -3 '
        program_name = 'main.exe '
        # TODO add plotted feature arg
        self.command = program_name \
                       + '-hc ' + num_hidden_neurons \
                       + ' -pd ' + num_prev_days \
                       + ' -fd ' + num_future_days \
                       + ' -f ' + '\"' + csv_file_path + '\"' \
                       + ' -trp ' + train_percentage \
                       + ' -b ' + bias \
                       + ' -k ' + use_keras \
                       + ' -plot ' + plotted_feature
        print('About to execute command: ' + self.command)
        self.predicting_process.start(self.command)

    def handle_csv_button(self):
        files, _ = QFileDialog.getOpenFileNames(None, "QFileDialog.getOpenFileNames()", "",
                                                "CSV Files (*.csv)")
        if len(files) > 0:
            self.ui.csvPathEdit.setText(files[0])

    def handle_kill_button(self):
        self.predicting_process.kill()

    def append_to_stdout_textedit(self, text):
        self.ui.stdOutputTextEdit.moveCursor(QTextCursor.End)
        self.ui.stdOutputTextEdit.insertPlainText(text)

    def stdout_ready(self):
        qbyte_array = self.predicting_process.readAllStandardOutput()
        text = bytearray(qbyte_array).decode(encoding="utf-8")
        print(text)
        self.append_to_stdout_textedit(text)

    def stderr_ready(self):
        qbyte_array = self.predicting_process.readAllStandardError()
        text = bytearray(qbyte_array).decode(encoding="utf-8")
        print(text)
        self.append_to_stdout_textedit(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    smp_window = StockMarketPredictorMainWindow()
    smp_window.show()
    print('Current working directory: ' + os.getcwd())
    sys.exit(app.exec())
