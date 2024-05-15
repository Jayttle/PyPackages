# -*- coding: UTF-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QIcon  # 导入QIcon类
from Ui_CommonSoftware import Ui_MainWindow

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        # 设置信号与槽   
        self.initMenu()  # 初始化菜单
        self.initConnections()  # 初始化信号与槽连接
        
    def initMenu(self):
        # 初始化菜单状态为收起
        self.isMenuExpanded = False

    def initConnections(self):
        # 设置信号与槽
        # 连接按钮点击事件与槽函数
        self.menuBtn.clicked.connect(self.toggleMenu)
         # 点击左侧按钮页面1切换到第一个页面
        self.pushButton.clicked.connect(self.display_page1)
        # 点击左侧按钮页面2切换到第二个页面
        self.pushButton_2.clicked.connect(self.display_page2)
        # 点击左侧按钮页面3切换到第三个页面
        self.pushButton_3.clicked.connect(self.display_page3)

    def toggleMenu(self):
        # 切换菜单展开状态
        self.isMenuExpanded = not self.isMenuExpanded
        if self.isMenuExpanded:
            # 展开菜单
            self.side_menu_body.setFixedWidth(150)
            self.menuBtn.setIcon(QIcon(":/icon_black/icon_black/menu-fold.svg"))
        else:
            # 收起菜单
            self.side_menu_body.setFixedWidth(0)
            self.menuBtn.setIcon(QIcon(":/icon_black/icon_black/menu-unfold.svg"))

    def display_page1(self):
        self.stackedWidget.setCurrentIndex(0)

    def display_page2(self):
        self.stackedWidget.setCurrentIndex(1)

    def display_page3(self):
        self.stackedWidget.setCurrentIndex(2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()
    sys.exit(app.exec_())