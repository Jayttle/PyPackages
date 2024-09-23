# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'DataProcessSoftware.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QRadioButton, QSizePolicy, QSpacerItem,
    QSpinBox, QStackedWidget, QStatusBar, QTextEdit,
    QVBoxLayout, QWidget)
import icon_black_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(876, 657)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"*{\n"
"	border:none;\n"
"	background-color:transparent;\n"
"}\n"
"#centralwidget{\n"
"	background-color:#e3f6f5;\n"
"}\n"
"#header{\n"
"	background-color:#ffffff;\n"
"	border-radius:10px\n"
"}\n"
"#main_body{\n"
"	border:none;\n"
"	background-color:transparent;\n"
"}\n"
"#side_main_body{\n"
"	background-color:#ffffff;\n"
"	border-radius:10px\n"
"}\n"
"#side_menu_body{\n"
"	background-color:#ffffff;\n"
"	border-radius:10px\n"
"}\n"
"QPushButton{\n"
"	background-color:#e3f5f6;\n"
"    text-align: center;\n"
"	font-family: \"Microsoft YaHei\";\n"
"	font-size: 12px;\n"
"    font-weight: bold;\n"
"	border-radius: 6px;\n"
"	padding: 5px 10px;\n"
"	border: 2px solid #9fabac;\n"
"\n"
"}\n"
"QPushButton:hover{\n"
"	background-color:#e3f6e4;\n"
"}\n"
"QPushButton.disabled {\n"
"    opacity: 0.6;\n"
"    cursor: not-allowed;\n"
"}\n"
"QRadioButton{\n"
"    font-family: \"Microsoft YaHei\";\n"
"	font-weight: bold;\n"
"    font-size: 10px;\n"
"    color: black;\n"
"    font-style: italic;\n"
"\n"
"    spacing: 5px;\n"
"   "
                        " padding-left: 3px;\n"
"    padding-top: 0px;\n"
"\n"
"    border-style: solid;\n"
"    border-width: 2px;\n"
"    border-color: #9fabac;\n"
"\n"
"    border-radius: 20;\n"
"\n"
"    background-color: #e3e8f6;\n"
"    background-repeat: no-repeat;\n"
"    background-position: right center;\n"
"}\n"
"QRadioButton:hover{\n"
"    background-color:#e3f6e4;\n"
"}\n"
"QRadioButton:pressed{\n"
"	color: green;\n"
"	border-color: blueviolet;\n"
"    background-color: black;\n"
"}\n"
"QRadioButton:disabled{\n"
"	color: blue;\n"
"	border-color: brown;\n"
"    background-color: aqua;\n"
"}\n"
"QLabel{\n"
"	font-family: \"Microsoft YaHei\";\n"
"    font-size: 16px;\n"
"}\n"
"QLineEdit {\n"
"	border: 1px solid #9fabac;\n"
"	border-radius: 3px; /* \u8fb9\u6846\u5706\u89d2 */\n"
"	padding-left: 5px; /* \u6587\u672c\u8ddd\u79bb\u5de6\u8fb9\u754c\u67095px */\n"
"	background-color: #e3e8f6; /* \u80cc\u666f\u989c\u8272 */\n"
"	color: #A0A0A0; /* \u6587\u672c\u989c\u8272 */\n"
"	selection-background-color: #A0A0A0; /* \u9009\u4e2d"
                        "\u6587\u672c\u7684\u80cc\u666f\u989c\u8272 */\n"
"	selection-color: #F2F2F2; /* \u9009\u4e2d\u6587\u672c\u7684\u989c\u8272 */\n"
"	font-family: \"Microsoft YaHei\"; /* \u6587\u672c\u5b57\u4f53\u65cf */\n"
"	font-size: 10pt; /* \u6587\u672c\u5b57\u4f53\u5927\u5c0f */\n"
"}\n"
"QLineEdit:hover { /* \u9f20\u6807\u60ac\u6d6e\u5728QLineEdit\u65f6\u7684\u72b6\u6001 */\n"
"	border: 1px solid #298DFF;\n"
"	border-radius: 3px;\n"
"	background-color: #F2F2F2;\n"
"	color: #298DFF;\n"
"	selection-background-color: #298DFF;\n"
"	selection-color: #F2F2F2;\n"
"}\n"
"\n"
"QLineEdit[echoMode=\"2\"] { /* QLineEdit\u6709\u8f93\u5165\u63a9\u7801\u65f6\u7684\u72b6\u6001 */\n"
"	lineedit-password-character: 9679;\n"
"	lineedit-password-mask-delay: 2000;\n"
"}\n"
"\n"
"QLineEdit:disabled { /* QLineEdit\u5728\u7981\u7528\u65f6\u7684\u72b6\u6001 */\n"
"	border: 1px solid #CDCDCD;\n"
"	background-color: #CDCDCD;\n"
"	color: #B4B4B4;\n"
"}\n"
"\n"
"QLineEdit:read-only { /* QLineEdit\u5728\u53ea\u8bfb\u65f6\u7684\u72b6\u6001 */\n"
"	back"
                        "ground-color: #CDCDCD;\n"
"	color: #F2F2F2;\n"
"}\n"
"QSpinBox{\n"
"	border: 1px solid #9fabac;\n"
"	border-radius: 3px; /* \u8fb9\u6846\u5706\u89d2 */\n"
"	padding-left: 5px; /* \u6587\u672c\u8ddd\u79bb\u5de6\u8fb9\u754c\u67095px */\n"
"	color: #A0A0A0; /* \u6587\u672c\u989c\u8272 */\n"
"	font-family: \"Microsoft YaHei\"; /* \u6587\u672c\u5b57\u4f53\u65cf */\n"
"	font-size: 10pt; /* \u6587\u672c\u5b57\u4f53\u5927\u5c0f */\n"
"}\n"
"QCheckBox {\n"
"    padding: 2px;\n"
"	font-family: \"Microsoft YaHei\";\n"
"    font-size: 14px;\n"
"}\n"
"QCheckBox:disabled, QRadioButton:disabled {\n"
"    color: #808086;\n"
"    padding: 2px;\n"
"}\n"
"\n"
"QCheckBox:hover {\n"
"    border-radius:4px;\n"
"    border-style:solid;\n"
"    padding-left: 1px;\n"
"    padding-right: 1px;\n"
"    padding-bottom: 1px;\n"
"    padding-top: 1px;\n"
"    border-width:1px;\n"
"    border-color: transparent;\n"
"}\n"
"QCheckBox::indicator:checked {\n"
"    image: url(:/icon_black/icon_black/checkbox3_blue.svg);\n"
"    height: 20px;\n"
""
                        "    width: 20px;\n"
"}\n"
"QCheckBox::indicator:unchecked {\n"
"    image: url( :/icon_black/icon_black/checkbox1.svg);\n"
"    height: 20px;\n"
"    width: 20px;\n"
"    background-color: #fbfdfa;\n"
"}\n"
"QCheckBox::indicator:unchecked:hover {\n"
"        image: url(:/icon_black/icon_black/checkbox3_gray.svg);\n"
"}")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.header = QFrame(self.centralwidget)
        self.header.setObjectName(u"header")
        self.header.setMinimumSize(QSize(0, 50))
        self.header.setMaximumSize(QSize(16777215, 50))
        self.header.setFrameShape(QFrame.Shape.NoFrame)
        self.horizontalLayout_2 = QHBoxLayout(self.header)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.side_menu = QFrame(self.header)
        self.side_menu.setObjectName(u"side_menu")
        self.menuBtn = QPushButton(self.side_menu)
        self.menuBtn.setObjectName(u"menuBtn")
        self.menuBtn.setGeometry(QRect(0, 0, 100, 31))
        self.menuBtn.setMinimumSize(QSize(100, 0))
        self.menuBtn.setMaximumSize(QSize(100, 16777215))
        self.menuBtn.setStyleSheet(u"#menuBtn{\n"
"	background-color:#e3f5f6;\n"
"    text-align: center;\n"
"	font-size: 12px;\n"
"	border-radius: 6px;\n"
"	padding: 5px 10px;\n"
"	border: 2px solid #9fabac;\n"
"\n"
"}\n"
"#menuBtn:hover{\n"
"	background-color:#e3f6e4;\n"
"}\n"
"#menuBtn.disabled {\n"
"    opacity: 0.6;\n"
"    cursor: not-allowed;\n"
"}")
        icon = QIcon()
        icon.addFile(u":/icon_black/icon_black/menu-unfold.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.menuBtn.setIcon(icon)

        self.horizontalLayout_2.addWidget(self.side_menu)

        self.frame_6 = QFrame(self.header)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.Shape.NoFrame)

        self.horizontalLayout_2.addWidget(self.frame_6)


        self.verticalLayout.addWidget(self.header)

        self.main_body = QFrame(self.centralwidget)
        self.main_body.setObjectName(u"main_body")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.main_body.sizePolicy().hasHeightForWidth())
        self.main_body.setSizePolicy(sizePolicy)
        self.main_body.setFrameShape(QFrame.Shape.NoFrame)
        self.horizontalLayout = QHBoxLayout(self.main_body)
        self.horizontalLayout.setSpacing(9)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.side_menu_body = QFrame(self.main_body)
        self.side_menu_body.setObjectName(u"side_menu_body")
        self.side_menu_body.setMinimumSize(QSize(0, 0))
        self.side_menu_body.setMaximumSize(QSize(150, 16777215))
        self.side_menu_body.setStyleSheet(u"")
        self.verticalLayout_2 = QVBoxLayout(self.side_menu_body)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.menu_btn_frame = QFrame(self.side_menu_body)
        self.menu_btn_frame.setObjectName(u"menu_btn_frame")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.menu_btn_frame.sizePolicy().hasHeightForWidth())
        self.menu_btn_frame.setSizePolicy(sizePolicy1)
        self.menu_btn_frame.setMinimumSize(QSize(0, 200))
        self.menu_btn_frame.setMaximumSize(QSize(16777215, 200))
        self.menu_btn_frame.setStyleSheet(u"#classBtn{\n"
"	background-color:#e3f5f6;\n"
"    text-align: center;\n"
"	font-size: 12px;\n"
"	border-radius: 6px;\n"
"	padding: 5px 10px;\n"
"	border: 2px solid #9fabac;\n"
"\n"
"}\n"
"#classBtn:hover{\n"
"	background-color:#e3f6e4;\n"
"}\n"
"#classBtn.disabled {\n"
"    opacity: 0.6;\n"
"    cursor: not-allowed;\n"
"}\n"
"#downloadBtn{\n"
"	background-color:#e3f5f6;\n"
"    text-align: center;\n"
"	font-size: 12px;\n"
"	border-radius: 6px;\n"
"	padding: 5px 10px;\n"
"	border: 2px solid #9fabac;\n"
"\n"
"}\n"
"#downloadBtn:hover{\n"
"	background-color:#e3f6e4;\n"
"}\n"
"#downloadBtn.disabled {\n"
"    opacity: 0.6;\n"
"    cursor: not-allowed;\n"
"}\n"
"#processBtn{\n"
"	background-color:#e3f5f6;\n"
"    text-align: center;\n"
"	font-size: 12px;\n"
"	border-radius: 6px;\n"
"	padding: 5px 10px;\n"
"	border: 2px solid #9fabac;\n"
"\n"
"}\n"
"#processBtn:hover{\n"
"	background-color:#e3f6e4;\n"
"}\n"
"#processBtn.disabled {\n"
"    opacity: 0.6;\n"
"    cursor: not-allowed;\n"
"}\n"
"#otherBtn{\n"
"	background"
                        "-color:#e3f5f6;\n"
"    text-align: center;\n"
"	font-size: 12px;\n"
"	border-radius: 6px;\n"
"	padding: 5px 10px;\n"
"	border: 2px solid #9fabac;\n"
"\n"
"}\n"
"#otherBtn:hover{\n"
"	background-color:#e3f6e4;\n"
"}\n"
"#otherBtn.disabled {\n"
"    opacity: 0.6;\n"
"    cursor: not-allowed;\n"
"}")
        self.menu_btn_frame.setFrameShape(QFrame.Shape.NoFrame)
        self.verticalLayout_3 = QVBoxLayout(self.menu_btn_frame)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.classBtn = QPushButton(self.menu_btn_frame)
        self.classBtn.setObjectName(u"classBtn")
        icon1 = QIcon()
        icon1.addFile(u":/icon_black/icon_black/icon-test4.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.classBtn.setIcon(icon1)

        self.verticalLayout_3.addWidget(self.classBtn)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.downloadBtn = QPushButton(self.menu_btn_frame)
        self.downloadBtn.setObjectName(u"downloadBtn")
        icon2 = QIcon()
        icon2.addFile(u":/icon_black/icon_black/icon-test7.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.downloadBtn.setIcon(icon2)

        self.verticalLayout_3.addWidget(self.downloadBtn)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_2)

        self.processBtn = QPushButton(self.menu_btn_frame)
        self.processBtn.setObjectName(u"processBtn")
        icon3 = QIcon()
        icon3.addFile(u":/icon_black/icon_black/icon-test6.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.processBtn.setIcon(icon3)

        self.verticalLayout_3.addWidget(self.processBtn)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_3)

        self.otherBtn = QPushButton(self.menu_btn_frame)
        self.otherBtn.setObjectName(u"otherBtn")
        icon4 = QIcon()
        icon4.addFile(u":/icon_black/icon_black/gengduo.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.otherBtn.setIcon(icon4)

        self.verticalLayout_3.addWidget(self.otherBtn)


        self.verticalLayout_2.addWidget(self.menu_btn_frame, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.frame_2 = QFrame(self.side_menu_body)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet(u".button{\n"
"	background-color:#e3f5f6;\n"
"    text-align: center;\n"
"	font-size: 12px;\n"
"	border-radius: 6px;\n"
"	padding: 5px 10px;\n"
"	border: 2px solid #9fabac;\n"
"\n"
"}\n"
".button:hover{\n"
"	background-color:#e3f6e4;\n"
"}\n"
".button.disabled {\n"
"    opacity: 0.6;\n"
"    cursor: not-allowed;\n"
"}")
        self.frame_2.setFrameShape(QFrame.Shape.NoFrame)

        self.verticalLayout_2.addWidget(self.frame_2)


        self.horizontalLayout.addWidget(self.side_menu_body)

        self.side_main_body = QFrame(self.main_body)
        self.side_main_body.setObjectName(u"side_main_body")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.side_main_body.sizePolicy().hasHeightForWidth())
        self.side_main_body.setSizePolicy(sizePolicy2)
        self.side_main_body.setFrameShape(QFrame.Shape.NoFrame)
        self.gridLayout_2 = QGridLayout(self.side_main_body)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.stackedWidget = QStackedWidget(self.side_main_body)
        self.stackedWidget.setObjectName(u"stackedWidget")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy3)
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.horizontalLayout_3 = QHBoxLayout(self.page)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.frame = QFrame(self.page)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(0, 100))
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.frame)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.GNSSRdo = QRadioButton(self.frame)
        self.GNSSRdo.setObjectName(u"GNSSRdo")

        self.verticalLayout_5.addWidget(self.GNSSRdo)

        self.verticalSpacer_4 = QSpacerItem(20, 14, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_4)

        self.DataPointRdo = QRadioButton(self.frame)
        self.DataPointRdo.setObjectName(u"DataPointRdo")

        self.verticalLayout_5.addWidget(self.DataPointRdo)

        self.verticalSpacer_5 = QSpacerItem(20, 14, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_5)

        self.RinexRdo = QRadioButton(self.frame)
        self.RinexRdo.setObjectName(u"RinexRdo")

        self.verticalLayout_5.addWidget(self.RinexRdo)


        self.horizontalLayout_3.addWidget(self.frame, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.frame_3 = QFrame(self.page)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy2.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy2)
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.classTxt = QTextEdit(self.frame_3)
        self.classTxt.setObjectName(u"classTxt")
        font = QFont()
        font.setFamilies([u"\u65b0\u5b8b\u4f53"])
        font.setPointSize(12)
        font.setKerning(True)
        self.classTxt.setFont(font)

        self.horizontalLayout_4.addWidget(self.classTxt)


        self.horizontalLayout_3.addWidget(self.frame_3)

        self.stackedWidget.addWidget(self.page)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.verticalLayout_6 = QVBoxLayout(self.page_2)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.page2_main_frame = QFrame(self.page_2)
        self.page2_main_frame.setObjectName(u"page2_main_frame")
        sizePolicy1.setHeightForWidth(self.page2_main_frame.sizePolicy().hasHeightForWidth())
        self.page2_main_frame.setSizePolicy(sizePolicy1)
        self.page2_main_frame.setMaximumSize(QSize(500, 16777215))
        self.page2_main_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.page2_main_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.page2_main_frame)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.frame_4 = QFrame(self.page2_main_frame)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label = QLabel(self.frame_4)
        self.label.setObjectName(u"label")

        self.horizontalLayout_5.addWidget(self.label)

        self.le_folder = QLineEdit(self.frame_4)
        self.le_folder.setObjectName(u"le_folder")
        self.le_folder.setMinimumSize(QSize(300, 0))

        self.horizontalLayout_5.addWidget(self.le_folder)

        self.horizontalSpacer_2 = QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_2)

        self.OpenFolderBtn = QPushButton(self.frame_4)
        self.OpenFolderBtn.setObjectName(u"OpenFolderBtn")

        self.horizontalLayout_5.addWidget(self.OpenFolderBtn)


        self.verticalLayout_7.addWidget(self.frame_4)

        self.frame_5 = QFrame(self.page2_main_frame)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_2 = QLabel(self.frame_5)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_6.addWidget(self.label_2)

        self.spb_start = QSpinBox(self.frame_5)
        self.spb_start.setObjectName(u"spb_start")
        self.spb_start.setMinimumSize(QSize(60, 0))
        self.spb_start.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_6.addWidget(self.spb_start)

        self.label_5 = QLabel(self.frame_5)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_6.addWidget(self.label_5)

        self.spb_end = QSpinBox(self.frame_5)
        self.spb_end.setObjectName(u"spb_end")
        self.spb_end.setMinimumSize(QSize(60, 0))
        self.spb_end.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_6.addWidget(self.spb_end)

        self.horizontalSpacer_6 = QSpacerItem(70, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_6)


        self.verticalLayout_7.addWidget(self.frame_5)

        self.frame_7 = QFrame(self.page2_main_frame)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout = QGridLayout(self.frame_7)
        self.gridLayout.setObjectName(u"gridLayout")
        self.chk_R032 = QCheckBox(self.frame_7)
        self.chk_R032.setObjectName(u"chk_R032")

        self.gridLayout.addWidget(self.chk_R032, 1, 1, 1, 1)

        self.chk_R081 = QCheckBox(self.frame_7)
        self.chk_R081.setObjectName(u"chk_R081")

        self.gridLayout.addWidget(self.chk_R081, 2, 2, 1, 1)

        self.chk_R052 = QCheckBox(self.frame_7)
        self.chk_R052.setObjectName(u"chk_R052")

        self.gridLayout.addWidget(self.chk_R052, 1, 3, 1, 1)

        self.label_6 = QLabel(self.frame_7)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 1)

        self.chk_R071 = QCheckBox(self.frame_7)
        self.chk_R071.setObjectName(u"chk_R071")

        self.gridLayout.addWidget(self.chk_R071, 2, 0, 1, 1)

        self.chk_R072 = QCheckBox(self.frame_7)
        self.chk_R072.setObjectName(u"chk_R072")

        self.gridLayout.addWidget(self.chk_R072, 2, 1, 1, 1)

        self.chk_R051 = QCheckBox(self.frame_7)
        self.chk_R051.setObjectName(u"chk_R051")

        self.gridLayout.addWidget(self.chk_R051, 1, 2, 1, 1)

        self.chk_R082 = QCheckBox(self.frame_7)
        self.chk_R082.setObjectName(u"chk_R082")

        self.gridLayout.addWidget(self.chk_R082, 2, 3, 1, 1)

        self.chk_R031 = QCheckBox(self.frame_7)
        self.chk_R031.setObjectName(u"chk_R031")

        self.gridLayout.addWidget(self.chk_R031, 1, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 4, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 2, 4, 1, 1)


        self.verticalLayout_7.addWidget(self.frame_7)


        self.verticalLayout_6.addWidget(self.page2_main_frame, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.page2_mid_frame = QFrame(self.page_2)
        self.page2_mid_frame.setObjectName(u"page2_mid_frame")
        sizePolicy.setHeightForWidth(self.page2_mid_frame.sizePolicy().hasHeightForWidth())
        self.page2_mid_frame.setSizePolicy(sizePolicy)
        self.page2_mid_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.page2_mid_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.page2_mid_frame)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.frame_8 = QFrame(self.page2_mid_frame)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_8)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.chk_p1 = QCheckBox(self.frame_8)
        self.chk_p1.setObjectName(u"chk_p1")

        self.verticalLayout_4.addWidget(self.chk_p1)

        self.chk_p2 = QCheckBox(self.frame_8)
        self.chk_p2.setObjectName(u"chk_p2")

        self.verticalLayout_4.addWidget(self.chk_p2)

        self.chk_p3 = QCheckBox(self.frame_8)
        self.chk_p3.setObjectName(u"chk_p3")

        self.verticalLayout_4.addWidget(self.chk_p3)

        self.chk_p4 = QCheckBox(self.frame_8)
        self.chk_p4.setObjectName(u"chk_p4")

        self.verticalLayout_4.addWidget(self.chk_p4)

        self.chk_p5 = QCheckBox(self.frame_8)
        self.chk_p5.setObjectName(u"chk_p5")

        self.verticalLayout_4.addWidget(self.chk_p5, 0, Qt.AlignmentFlag.AlignLeft)

        self.btn_runp = QPushButton(self.frame_8)
        self.btn_runp.setObjectName(u"btn_runp")
        icon5 = QIcon()
        icon5.addFile(u":/icon_black/icon_black/icon-test8.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_runp.setIcon(icon5)

        self.verticalLayout_4.addWidget(self.btn_runp)


        self.horizontalLayout_7.addWidget(self.frame_8, 0, Qt.AlignmentFlag.AlignLeft)

        self.frame_9 = QFrame(self.page2_mid_frame)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.frame_9)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.downloadTxt = QTextEdit(self.frame_9)
        self.downloadTxt.setObjectName(u"downloadTxt")

        self.horizontalLayout_8.addWidget(self.downloadTxt)


        self.horizontalLayout_7.addWidget(self.frame_9)


        self.verticalLayout_6.addWidget(self.page2_mid_frame)

        self.processbar_frame = QFrame(self.page_2)
        self.processbar_frame.setObjectName(u"processbar_frame")
        self.processbar_frame.setMinimumSize(QSize(0, 50))
        self.processbar_frame.setMaximumSize(QSize(16777215, 50))
        self.processbar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.processbar_frame.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout_6.addWidget(self.processbar_frame)

        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        self.label_3 = QLabel(self.page_3)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 20, 54, 12))
        self.stackedWidget.addWidget(self.page_3)
        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.horizontalLayout_9 = QHBoxLayout(self.page_4)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.frame_11 = QFrame(self.page_4)
        self.frame_11.setObjectName(u"frame_11")
        self.frame_11.setMinimumSize(QSize(200, 0))
        self.frame_11.setMaximumSize(QSize(200, 16777215))
        self.frame_11.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_11.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame_11)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.frame_12 = QFrame(self.frame_11)
        self.frame_12.setObjectName(u"frame_12")
        self.frame_12.setMinimumSize(QSize(0, 130))
        self.frame_12.setMaximumSize(QSize(16777215, 130))
        self.frame_12.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_12.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_3 = QGridLayout(self.frame_12)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.le_mouse = QLineEdit(self.frame_12)
        self.le_mouse.setObjectName(u"le_mouse")

        self.gridLayout_3.addWidget(self.le_mouse, 1, 0, 1, 1)

        self.le_mouse_info = QLineEdit(self.frame_12)
        self.le_mouse_info.setObjectName(u"le_mouse_info")

        self.gridLayout_3.addWidget(self.le_mouse_info, 3, 0, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_5, 1, 1, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(31, 18, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_4, 0, 1, 1, 1)

        self.label_4 = QLabel(self.frame_12)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(16777215, 50))

        self.gridLayout_3.addWidget(self.label_4, 0, 0, 1, 1)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_7, 3, 1, 1, 1)

        self.label_8 = QLabel(self.frame_12)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_3.addWidget(self.label_8, 2, 0, 1, 1)


        self.verticalLayout_8.addWidget(self.frame_12, 0, Qt.AlignmentFlag.AlignHCenter)

        self.frame_13 = QFrame(self.frame_11)
        self.frame_13.setObjectName(u"frame_13")
        self.frame_13.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_13.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.frame_13)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.frame_14 = QFrame(self.frame_13)
        self.frame_14.setObjectName(u"frame_14")
        self.frame_14.setMinimumSize(QSize(0, 120))
        self.frame_14.setMaximumSize(QSize(16777215, 120))
        self.frame_14.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_14.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_10 = QVBoxLayout(self.frame_14)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.btn_turn_mouse = QPushButton(self.frame_14)
        self.btn_turn_mouse.setObjectName(u"btn_turn_mouse")

        self.verticalLayout_10.addWidget(self.btn_turn_mouse)

        self.label_7 = QLabel(self.frame_14)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_10.addWidget(self.label_7)

        self.btn_save_mouse = QPushButton(self.frame_14)
        self.btn_save_mouse.setObjectName(u"btn_save_mouse")

        self.verticalLayout_10.addWidget(self.btn_save_mouse)


        self.verticalLayout_9.addWidget(self.frame_14)

        self.frame_15 = QFrame(self.frame_13)
        self.frame_15.setObjectName(u"frame_15")
        sizePolicy.setHeightForWidth(self.frame_15.sizePolicy().hasHeightForWidth())
        self.frame_15.setSizePolicy(sizePolicy)
        self.frame_15.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_15.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout_9.addWidget(self.frame_15)


        self.verticalLayout_8.addWidget(self.frame_13)


        self.horizontalLayout_9.addWidget(self.frame_11)

        self.frame_10 = QFrame(self.page_4)
        self.frame_10.setObjectName(u"frame_10")
        sizePolicy2.setHeightForWidth(self.frame_10.sizePolicy().hasHeightForWidth())
        self.frame_10.setSizePolicy(sizePolicy2)
        self.frame_10.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_10.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_10 = QHBoxLayout(self.frame_10)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.txt_mouse = QTextEdit(self.frame_10)
        self.txt_mouse.setObjectName(u"txt_mouse")

        self.horizontalLayout_10.addWidget(self.txt_mouse)


        self.horizontalLayout_9.addWidget(self.frame_10)

        self.stackedWidget.addWidget(self.page_4)

        self.gridLayout_2.addWidget(self.stackedWidget, 0, 0, 1, 1)


        self.horizontalLayout.addWidget(self.side_main_body)


        self.verticalLayout.addWidget(self.main_body)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(3)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.menuBtn.setText(QCoreApplication.translate("MainWindow", u"\u83dc\u5355", None))
        self.classBtn.setText(QCoreApplication.translate("MainWindow", u"\u67e5\u770b\u6570\u636e\u7ed3\u6784", None))
        self.downloadBtn.setText(QCoreApplication.translate("MainWindow", u"FTP\u6587\u4ef6\u4e0b\u8f7d", None))
        self.processBtn.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u5904\u7406", None))
        self.otherBtn.setText(QCoreApplication.translate("MainWindow", u"\u5176\u4ed6", None))
        self.GNSSRdo.setText(QCoreApplication.translate("MainWindow", u"MyGNSSCSVModel", None))
        self.DataPointRdo.setText(QCoreApplication.translate("MainWindow", u"DataPoint", None))
        self.RinexRdo.setText(QCoreApplication.translate("MainWindow", u"RinexFileInfo", None))
        self.classTxt.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'\u65b0\u5b8b\u4f53'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u6587\u4ef6\u5939", None))
        self.le_folder.setText("")
        self.OpenFolderBtn.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u65f6\u95f4\u6bb5", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"~", None))
        self.chk_R032.setText(QCoreApplication.translate("MainWindow", u"R032", None))
        self.chk_R081.setText(QCoreApplication.translate("MainWindow", u"R081", None))
        self.chk_R052.setText(QCoreApplication.translate("MainWindow", u"R052", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"\u6d4b\u7ad9\uff1a", None))
        self.chk_R071.setText(QCoreApplication.translate("MainWindow", u"R071", None))
        self.chk_R072.setText(QCoreApplication.translate("MainWindow", u"R072", None))
        self.chk_R051.setText(QCoreApplication.translate("MainWindow", u"R051", None))
        self.chk_R082.setText(QCoreApplication.translate("MainWindow", u"R082", None))
        self.chk_R031.setText(QCoreApplication.translate("MainWindow", u"R031", None))
        self.chk_p1.setText(QCoreApplication.translate("MainWindow", u"\u83b7\u53d6To_download", None))
        self.chk_p2.setText(QCoreApplication.translate("MainWindow", u"\u4e0b\u8f7d\u6587\u4ef6", None))
        self.chk_p3.setText(QCoreApplication.translate("MainWindow", u"\u89e3\u538b\u7f29\u4e0e\u8f6c\u6362", None))
        self.chk_p4.setText(QCoreApplication.translate("MainWindow", u"\u5408\u5e76\u6587\u4ef6", None))
        self.chk_p5.setText(QCoreApplication.translate("MainWindow", u"\u6587\u4ef6markername\u4fee\u6539", None))
        self.btn_runp.setText(QCoreApplication.translate("MainWindow", u"\u6267\u884c", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u5904\u7406", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u9f20\u6807\u5750\u6807", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"\u5907\u6ce8", None))
        self.btn_turn_mouse.setText(QCoreApplication.translate("MainWindow", u"\u5f00/\u5173\u76d1\u6d4b\u9f20\u6807", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"\u5feb\u6377\u952e\uff1actrl+1", None))
        self.btn_save_mouse.setText(QCoreApplication.translate("MainWindow", u"\u5b58\u50a8\u5750\u6807", None))
    # retranslateUi

