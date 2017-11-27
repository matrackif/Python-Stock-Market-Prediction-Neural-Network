import matplotlib.pyplot as plt
from src.elm import ELMMatrixVersion
from src.regular_model import RegularModel

if __name__ == '__main__':
    reg_model = RegularModel()
    elm = ELMMatrixVersion()
    reg_model.train()
    elm.train()
    real_train = elm.y_tr
    real_test = elm.y_te
    elm_train_pred, elm_test_pred = elm.predict()
    reg_train_pred, reg_test_pred = reg_model.predict()
    index_of_plotted_feature = elm.index_of_plotted_feature

    plt.figure(0)
    training_data_graph, = plt.plot(real_train[:, index_of_plotted_feature], label='Actual training data')
    training_prediction_graph, = plt.plot(reg_train_pred[:, index_of_plotted_feature],
                                          label='Prediction of training data')
    training_prediction_elm_graph, = plt.plot(elm_train_pred[:, index_of_plotted_feature],
                                              label='ELM Prediction of training data')

    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.title('ELM vs Reg model vs Real training data')
    plt.legend(handles=[training_data_graph, training_prediction_graph, training_prediction_elm_graph])
    plt.show()

    plt.figure(1)
    test_data_graph, = plt.plot(real_test[:, index_of_plotted_feature], label='Actual test data')
    test_prediction_graph, = plt.plot(reg_test_pred[:, index_of_plotted_feature],
                                      label='Prediction of test data')
    test_prediction_elm_graph, = plt.plot(elm_test_pred[:, index_of_plotted_feature],
                                          label='ELM Prediction of test data')

    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.title('ELM vs Reg model vs Real test data')
    plt.legend(handles=[test_data_graph, test_prediction_graph, test_prediction_elm_graph])
    plt.show()
