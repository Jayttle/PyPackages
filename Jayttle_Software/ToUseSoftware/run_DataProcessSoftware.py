import icon_black_rc
from collections import defaultdict
from DataProcessSoftware_ui import *
import sys
import pyautogui
import time
from threading import Thread
from PySide6 import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from Custom_Widgets.Widgets import *


def messageHandler(type, context, message):
    if "Negative sizes" in message:
        # 这里不做任何操作，即忽略包含 "Negative sizes" 的消息
        pass
    else:
        # 对于其他类型的消息，可以选择打印到控制台或者记录到日志文件中
        print(message)

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.initProperties() # 初始化变量属性
        self.initSpinBox() # 初始化SpinBox
        self.initConnections()  # 初始化信号与槽连接  

# region 初始化
    def initProperties(self):
        # 初始化菜单状态为展开
        self.isMenuExpanded = True
        self.spb_start_value = 16  # 存储 spinbox1 的值的属性
        self.spb_end_value = 19  # 存储 spinbox2 的值的属性
        self.root_folder = f"D:\Ropeway" # 存储软件运行的根目录
        self.le_folder.setText(self.root_folder)
        self.base_marker_names = ['B011', 'B021']
        self.selected_checkboxes = []  # 用于记录被选中的复选框名称后缀的列表
        self.selected_chk_process = []  # 处理流程字典，默认值为列表
        self.isMouseRecord = False # 是否开始检测鼠标位置
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_mouse_position)

        # 映射处理流程到操作的字典
        self.process_mapping = {
            'p1': self.perform_operation_p1,
            'p2': self.perform_operation_p2,
            'p3': self.perform_operation_p3,
            'p4': self.perform_operation_p4,
            'p5': self.perform_operation_p5,
            # 添加其他处理流程的映射...
        }
# region 处理流程映射
    def perform_operation_p1(self):
        # 执行处理流程 p1 对应的操作
        print("Performing operation for process p1")
        if self.toDownload_folder:
            print(f"{self.toDownload_folder}")
        return ['Result for p1']  # 返回处理流程 p1 的结果

    def perform_operation_p2(self):
        # 执行处理流程 p2 对应的操作
        print("Performing operation for process p2")
        if self.FTP_folder:
            print(f"{self.FTP_folder}")
        return ['Result for p2']  # 返回处理流程 p2 的结果

    def perform_operation_p3(self):
        # 执行处理流程 p3 对应的操作
        print("Performing operation for process p3")
        return ['Result for p3']  # 返回处理流程 p3 的结果

    def perform_operation_p4(self):
        # 执行处理流程 p4 对应的操作
        print("Performing operation for process p4")
        return ['Result for p4']  # 返回处理流程 p4 的结果

    def perform_operation_p5(self):
        # 执行处理流程 p5 对应的操作
        print("Performing operation for process p5")
        return ['Result for p5']  # 返回处理流程 p5 的结果
# endregion

    def initSpinBox(self):
        # spinbox1
        self.spb_start.setRange(0, 23)
        self.spb_start.setSingleStep(1)
        self.spb_start.setWrapping(True)
        # spinbox2
        self.spb_end.setRange(0, 23)
        self.spb_end.setSingleStep(1)
        self.spb_end.setWrapping(True)

        # 设置初始值
        self.spb_start.setValue(self.spb_start_value)
        self.spb_end.setValue(self.spb_end_value)

        self.spb_start.valueChanged.connect(self.spinBox_cb)
        self.spb_end.valueChanged.connect(self.spinBox2_cb)

    def spinBox_cb(self, value):
        print("spinbox1's current value is ", value)

    def spinBox2_cb(self, value):
        print("spinbox2's current value is ", value)
        
# endregion
    
# region 信号
    def initConnections(self):
        # 设置信号与槽
        # 连接按钮点击事件与槽函数
        self.menuBtn.clicked.connect(self.toggleMenu)
        self.classBtn.clicked.connect(self.display_page1) # 点击左侧按钮页面1切换到第一个页面
        self.downloadBtn.clicked.connect(self.display_page2) # 点击左侧按钮页面2切换到第二个页面
        self.processBtn.clicked.connect(self.display_page3) #  点击左侧按钮页面3切换到第三个页面
        self.otherBtn.clicked.connect(self.display_page4) # 第四个页面
        self.GNSSRdo.clicked.connect(self.display_GNSS) # 点击rdo按钮显示对应数据结构情况
        self.DataPointRdo.clicked.connect(self.display_DPt) # 点击rdo按钮显示对应数据结构情况
        self.RinexRdo.clicked.connect(self.display_Rnx) # 点击rdo按钮显示对应数据结构情况
        self.btn_runp.clicked.connect(self.run_download_process)  # 执行运行下载功能
        self.OpenFolderBtn.clicked.connect(self.open_folder) # 打开要创建的目录
        self.chk_R031.stateChanged.connect(self.chk_receive_StateChanged) # chk绑定
        self.chk_R032.stateChanged.connect(self.chk_receive_StateChanged) # chk绑定
        self.chk_R051.stateChanged.connect(self.chk_receive_StateChanged) # chk绑定
        self.chk_R052.stateChanged.connect(self.chk_receive_StateChanged) # chk绑定
        self.chk_R071.stateChanged.connect(self.chk_receive_StateChanged) # chk绑定
        self.chk_R072.stateChanged.connect(self.chk_receive_StateChanged) # chk绑定
        self.chk_R081.stateChanged.connect(self.chk_receive_StateChanged) # chk绑定
        self.chk_R082.stateChanged.connect(self.chk_receive_StateChanged) # chk绑定
        self.chk_p1.stateChanged.connect(self.chk_process_StateChanged) # chk处理流程复选框绑定
        self.chk_p2.stateChanged.connect(self.chk_process_StateChanged) # chk处理流程复选框绑定
        self.chk_p3.stateChanged.connect(self.chk_process_StateChanged) # chk处理流程复选框绑定
        self.chk_p4.stateChanged.connect(self.chk_process_StateChanged) # chk处理流程复选框绑定
        self.chk_p5.stateChanged.connect(self.chk_process_StateChanged) # chk处理流程复选框绑定
        self.btn_turn_mouse.clicked.connect(self.turn_off_on_mouse) # 绑定监测鼠标开关
        self.btn_save_mouse.clicked.connect(self.print_mouse_position) 

        shortcut = QShortcut(QKeySequence('Ctrl+1'), self)        # 添加键盘快捷键 Ctrl+1
        shortcut.activated.connect(self.print_mouse_position)

# endregion

# region 槽函数已完成
    def toggleMenu(self):
        # 切换菜单展开状态
        self.isMenuExpanded = not self.isMenuExpanded
        if self.isMenuExpanded:
            # 展开菜单
            self.expandMenu()
        else:
            # 收起菜单
            self.collapseMenu()

    def expandMenu(self):
        # 创建属性动画，将菜单宽度从0渐变到150
        self.animation = QPropertyAnimation(self.side_menu_body, b"maximumWidth")
        self.animation.setDuration(1000)  # 设置动画持续时间
        self.animation.setEasingCurve(QEasingCurve.OutBack)  # 设置动画缓和曲线
        self.animation.setStartValue(0)  # 动画起始值
        self.animation.setEndValue(150)  # 动画结束值
        self.animation.start()

        # 更改按钮图标
        self.menuBtn.setIcon(QIcon(":/icon_black/icon_black/menu-fold.svg"))

    def collapseMenu(self):
        # 获取当前最大宽度
        current_max_width = self.side_menu_body.maximumWidth()

        # 如果当前最大宽度为负数，将其设置为正数
        if current_max_width < 0:
            self.side_menu_body.setMaximumWidth(abs(current_max_width))

        # 创建属性动画，将菜单宽度从150渐变到0
        self.animation = QPropertyAnimation(self.side_menu_body, b"maximumWidth")
        self.animation.setDuration(1000)  # 设置动画持续时间
        self.animation.setEasingCurve(QEasingCurve.OutBack)  # 设置动画缓和曲线
        self.animation.setStartValue(150)  # 动画起始值
        self.animation.setEndValue(0)  # 动画结束值
        self.animation.start()

        # 更改按钮图标
        self.menuBtn.setIcon(QIcon(":/icon_black/icon_black/menu-unfold.svg"))

    def display_page1(self):
        self.stackedWidget.setCurrentIndex(0)

    def display_page2(self):
        self.stackedWidget.setCurrentIndex(1)

    def display_page3(self):
        self.stackedWidget.setCurrentIndex(2)

    def display_page4(self):
        self.stackedWidget.setCurrentIndex(3)

    def display_GNSS(self):
        # 在 QTexEdit 中显示文本信息
        self.classTxt.clear()
        description = (
            "起始时间(start_time): 开始记录的时间\n"
            "结束时间(end_time): 结束记录的时间\n"
            "点ID(point_id): 点的唯一标识\n"
            "北坐标(north_coordinate): 北向坐标\n"
            "东坐标(east_coordinate): 东向坐标\n"
            "高程(altitude): 海拔高度\n"
            "纬度(latitude): 纬度坐标\n"
            "经度(longitude): 经度坐标\n"
            "PDOP(pdop): 位置精度因子\n"
            "RMS(rms): 均方根误差\n"
            "水平精度(horizontal_accuracy): 水平精度\n"
            "垂直精度(vertical_accuracy): 垂直精度\n"
            "起始点ID(start_point_id): 起始点的唯一标识\n"
            "结束点ID(end_point_id): 结束点的唯一标识\n"
            "X增量(x_increment): X轴增量\n"
            "Y增量(y_increment): Y轴增量\n"
            "Z增量(z_increment): Z轴增量\n"
            "矢量长度(vector_length): 矢量长度\n"
            "解算类型(solution_type): 解算类型\n"
            "状态(status): 状态"
        )
        self.classTxt.setText(description)

    def display_DPt(self):
        # 在 QTexEdit 中显示文本信息
        self.classTxt.clear()
        description = (
            "ID(point_id): 唯一标识\n"
            "北坐标(north_coordinate): 数据点在北向上的坐标\n"
            "东坐标(east_coordinate): 数据点在东向上的坐标\n"
            "高程(elevation): 海拔高度\n"
            "纬度(latitude): 纬度坐标\n"
            "经度(longitude): 经度坐标\n"
            "椭球高度(ellipsoid_height): 椭球高度\n"
            "开始时间(start_time): 数据点记录的开始时间\n"
            "结束时间(end_time): 数据点记录的结束时间\n"
            "持续时间(duration): 数据点记录的持续时间\n"
            "位置精度衰减因子(PDOP)(pdop): 位置精度衰减因子\n"
            "均方根(RMS)(rms): 均方根误差\n"
            "水平精度(horizontal_accuracy): 水平精度\n"
            "垂直精度(vertical_accuracy): 垂直精度\n"
            "北坐标误差(north_coordinate_error): 北坐标误差\n"
            "东坐标误差(east_coordinate_error): 东坐标误差\n"
            "高程误差(elevation_error): 高程误差\n"
            "高度误差(height_error): 高度误差"
        )
        self.classTxt.setText(description)

    def display_Rnx(self):
        # 在 QTexEdit 中显示文本信息
        self.classTxt.clear()
        description = (
            "来自FTP(ftp_file): 指示文件是否来自FTP\n"
            "输入路径(input_path): 文件的输入路径\n"
            "文件信息对象(file_info): 表示文件的路径信息对象\n"
            "文件名(file_name): 文件的名称\n"
            "站点名称(station_name): 文件名的前三个字符，代表站点名称\n"
            "标记名称(marker_name): 文件名的前四个字符，代表标记名称\n"
            "站点ID(station_id): 从站点名称中第三个字符转换而来的整数ID\n"
            "站点类型(station_type): 根据文件名的第一个字符确定的站点类型，如果未知则为 Unknown\n"
            "接收器ID(receiver_id): 从文件名第四个字符转换而来的整数ID\n"
            "起始GPS时间字符串(start_gps_time_str): 从文件名中获取的起始GPS时间字符串\n"
            "起始GPS时间(start_gps_time): 从起始GPS时间字符串转换得到的日期时间对象\n"
            "持续时间字符串(duration_str): 从文件名中获取的持续时间字符串\n"
            "持续时间(duration): 从持续时间字符串转换得到的时间间隔对象\n"
            "时间字符串(time_str): 包含起始GPS时间和持续时间的字符串表示\n"
            "文件类型(file_type): 根据文件名中的第二个字符确定的文件类型，如果未知则为 Unknown\n"
            "信息字符串(info_str): 从文件名中获取的信息字符串\n"
            "是否为压缩文件(compressed): 指示文件是否为压缩文件\n"
            "文件格式(format): 根据文件扩展名确定的文件格式，如果未知则为 Unknown"
        )
        self.classTxt.setText(description)

    def open_folder(self):
        # 打开文件夹对话框
        self.root_folder = QFileDialog.getExistingDirectory(self, "选择文件夹", "\\")
        if self.root_folder:
            # 如果用户选择了文件夹，则将路径显示在lineEdit控件中
            self.le_folder.setText(self.root_folder)
# endregion
    # 在其他函数中使用 spinBox 的值
    def run_download_process(self):
        if self.root_folder:
            if self.selected_checkboxes:
                for item in self.selected_checkboxes:
                    folder_name = f"{item}_{self.spb_start_value}{self.spb_end_value}"
                    folder_path = os.path.join(self.root_folder, folder_name)
                    os.makedirs(folder_path, exist_ok=True)
                    self.toDownload_folder = os.path.join(folder_path, 'toDownload')
                    os.makedirs(self.toDownload_folder, exist_ok=True)
                    self.FTP_folder = os.path.join(folder_path, 'FTP')
                    os.makedirs(self.FTP_folder, exist_ok=True)
                    self.FTPMerge_folder = os.path.join(folder_path, 'FTPMerge')
                    os.makedirs(self.FTPMerge_folder, exist_ok=True)
                    if self.selected_chk_process:
                        for suffix in self.selected_chk_process:
                            if suffix in self.process_mapping:
                                process = self.process_mapping[suffix]
                                process()
            else: 
                print("selected_checkboxes is not set.")
        else:
            print("Root folder is not set.")

    def chk_process_StateChanged(self, state):
        # 处理流程复选框
        checkbox = self.sender()
        if isinstance(checkbox, QCheckBox):
            if state == 2: # chk选中state是2
                checkbox_name = checkbox.objectName()
                suffix = checkbox_name.split("_")[-1]
                self.selected_chk_process.append(suffix)
                self.selected_chk_process.sort()
                print("Selected checkboxes:", self.selected_chk_process)
            elif state == 0:# chk未选中state是0
                checkbox_name = checkbox.objectName()
                suffix = checkbox_name.split("_")[-1]
                if suffix in self.selected_chk_process:
                    self.selected_chk_process.remove(suffix)
                    print("Selected checkboxes:", self.selected_chk_process)

    def chk_receive_StateChanged(self, state):
        # 接收机复选框
        checkbox = self.sender()
        if isinstance(checkbox, QCheckBox):
            if state == 2: # chk选中state是2
                checkbox_name = checkbox.objectName()
                suffix = checkbox_name.split("_")[-1]
                self.selected_checkboxes.append(suffix)
                self.selected_checkboxes.sort()
                print("Selected checkboxes:", self.selected_checkboxes)
            elif state == 0:# chk未选中state是0
                checkbox_name = checkbox.objectName()
                suffix = checkbox_name.split("_")[-1]
                if suffix in self.selected_checkboxes:
                    self.selected_checkboxes.remove(suffix)
                    print("Selected checkboxes:", self.selected_checkboxes)
    
    def turn_off_on_mouse(self):
        if self.isMouseRecord:
            self.isMouseRecord = False
            self.timer.stop()  # 停止计时器
            self.btn_turn_mouse.setText('打开监测')
        else:
            self.isMouseRecord = True
            self.btn_turn_mouse.setText('关闭监测')
            self.timer.start(100)  # 每100ms更新一次

    def update_mouse_position(self):
        # 获取鼠标位置
        x, y = pyautogui.position()

        # 更新 QLabel 中的文本
        text = f'({x}, {y})'
        self.le_mouse.setText(text)

    def print_mouse_position(self):
        text = f"{self.le_mouse.text()} {self.le_mouse_info.text()}"
        self.txt_mouse.append(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 将消息处理器设置为自定义的函数
    qInstallMessageHandler(messageHandler)
    main_window = MyMainWindow()
    main_window.show()

    sys.exit(app.exec_())