import pyautogui
import os

def move_and_click( x, y):
    """移动鼠标到指定坐标并单击"""
    pyautogui.moveTo(x, y)
    pyautogui.click()

def move_and_Twoclick( x, y):
    """移动鼠标到指定坐标并双击"""
    pyautogui.moveTo(x, y)
    pyautogui.click()
    pyautogui.click()

def type_string( string):
    """输入字符串"""
    pyautogui.typewrite(string)

def move_and_click_with_shift( x, y):
    """按住 Shift 键的鼠标移动单击"""
    pyautogui.keyDown('shift')  # 按下 Shift 键
    pyautogui.moveTo(x, y)
    pyautogui.click()
    pyautogui.keyUp('shift')  # 释放 Shift 键

def press_delete_key():
    """按下键盘上的 Delete 键"""
    pyautogui.press('delete')

def press_ctrl_space():
    """按下键盘上的 Ctrl 和空格键"""
    pyautogui.hotkey('ctrl', 'space')

def right_click_and_press_D():
    """右键并按下键盘的D"""
    pyautogui.rightClick()  # 右键
    pyautogui.press('d')    # 按下键盘的D键   

def get_subdirectories_with_no_csv(folder_path):
    """获取目标文件夹中所有没有 CSV 文件的文件夹的名字"""
    subdirectories = []
    # 遍历目标文件夹中的所有文件和文件夹
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # 检查是否为文件夹
        if os.path.isdir(item_path):
            # 检查文件夹内是否有 CSV 文件
            if not any(file.lower().endswith('.csv') for file in os.listdir(item_path)):
                subdirectories.append(item)
    return subdirectories

def get_subdirectories_with_no_csv_without03(folder_path):
    """获取目标文件夹中所有没有 CSV 文件的文件夹的名字，排除第5~6位是'03'的文件夹，并且文件夹内文件数等于6"""
    subdirectories = []
    # 遍历目标文件夹中的所有文件和文件夹
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # 检查是否为文件夹
        if os.path.isdir(item_path):
            # 检查文件夹名是否满足条件
            if (len(item) >= 6) and (item[4:6] != '03'):
                # 检查文件夹内是否有 CSV 文件
                if not any(file.lower().endswith('.csv') for file in os.listdir(item_path)):
                    # 检查文件夹内文件数量是否为6
                    if len(os.listdir(item_path)) == 6:
                        subdirectories.append(item)
    return subdirectories

def get_csv_file_paths(folder_path):
    """获取目标文件夹中所有CSV文件的路径"""
    csv_file_paths = []

    # 遍历目标文件夹中的所有文件和文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(root, file)
                csv_file_paths.append(file_path)

    return csv_file_paths


def move_up_with_right_click(distance):
    """按住右键的鼠标向上移动一定距离"""
    pyautogui.mouseDown(button='right')  # 按住右键
    pyautogui.move(0, -distance, duration=0.5)  # 向上移动指定距离
    pyautogui.mouseUp(button='right')  # 松开右键

def move_right_with_right_click(distance):
    """按住右键的鼠标向右移动一定距离"""
    pyautogui.mouseDown(button='right')  # 按住右键
    pyautogui.move(distance, 0, duration=0.5)  # 向右移动指定距离
    pyautogui.mouseUp(button='right')  # 松开右键

def moveTo_up_with_right_click(x,y,distance):
    """移动到指定位置再按住右键的鼠标向上移动一定距离"""
    pyautogui.moveTo(x,y)
    move_up_with_right_click(distance)

def moveTo_right_with_right_click(x,y,distance):
    """移动到指定位置再按住右键的鼠标向右移动一定距离"""
    pyautogui.moveTo(x,y)
    move_right_with_right_click(distance)


def get_mouse_position():
    """获取鼠标位置"""
    x, y = pyautogui.position()
    # 返回鼠标位置
    return x, y

def list_folders(directory):
    """获取目标文件夹下的所有文件夹folder名字"""
    folder_names = []
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_names.append(os.path.join(root, folder))
    return folder_names

def list_files_in_folders(folder_names):
    """获取目标文件夹下的所有文件file名字"""
    file_paths = []
    for folder in folder_names:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_paths.append(os.path.join(root, file))
    return file_paths