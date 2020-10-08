import sys
import numpy as np
from PyQt5.QtWidgets import QApplication,QMainWindow,QMessageBox,QPushButton
from functools import partial
import settings
import os
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path)


def input_parameter(ui):

    ls=[]
    
    ls.append(int(ui.lineEdit.text())) #m_RFrameWidth
    ls.append(int(ui.lineEdit_2.text())) #seed
    ls.append(int(ui.lineEdit_4.text())) #m_HLUnitNum
    
    ls.append(int(ui.lineEdit_5.text())) #m_YJieV
    ls.append(int(ui.lineEdit_6.text())) #m_YXingV
    ls.append(int(ui.lineEdit_7.text())) #m_YShouV
   
    ls.append(int(ui.lineEdit_8.text())) #m_OJieV
    ls.append(int(ui.lineEdit_9.text())) #m_OXingV
    ls.append(int(ui.lineEdit_10.text())) #m_OShouV

    ls=np.array(ls)
    np.save(path + "/parameters/10_parameters.npy",ls)

def study(ui):
    import learn
    learn.learn_protein(int(ui.lineEdit_3.text()))

def forecast(ui):
    import predict
    predict.predict(int(ui.lineEdit_11.text()))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = settings.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.pushButton.clicked.connect(partial(input_parameter,ui))
    ui.pushButton_2.clicked.connect(partial(study,ui))
    ui.pushButton_3.clicked.connect(partial(forecast,ui))
    sys.exit(app.exec_())


