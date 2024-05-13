from Ui_interface import *
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from Custom_Widgets.Widgets import *

class MainWindow(QMainWindow):
    def __init__(self, parent=None) :
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        ###########################################################
        # APPLY JSON STYLESHEET
        ###########################################################
        # self = QMainWindow class
        # self.ui = Ui_Main
        loadJsonStyle(self, self.ui)
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())