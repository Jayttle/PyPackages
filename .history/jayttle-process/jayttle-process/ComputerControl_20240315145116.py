import pyautogui
import os
from PIL import Image 
import pytesseract
import pyautogui
import time

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
    pyautogui.moveTo(x,y)
    move_up_with_right_click(distance)

def moveTo_right_with_right_click(x,y,distance):
    pyautogui.moveTo(x,y)
    move_right_with_right_click(distance)


def read_text_from_window(window_region):
    """
    从指定的窗口区域中读取文本
    Args:
        window_region: 窗口区域的坐标 (x, y, width, height)
    Returns:
        识别到的文本
    """
    # 截取窗口区域的截图
    screenshot = pyautogui.screenshot(region=window_region)
    # 示例：灰度化和二值化
    screenshot = screenshot.convert('L')  # 转为灰度图
    threshold = 200  # 阈值，根据需要调整
    screenshot = screenshot.point(lambda p: p > threshold and 255)
    # 将截图保存为临时文件
    temp_image_path = "temp_screenshot.png"
    screenshot.save(temp_image_path)
    # 使用 pytesseract 识别文本
    text = pytesseract.image_to_string(Image.open(temp_image_path), lang='chi_sim')
    # 删除临时文件
    os.remove(temp_image_path)
    return text

def ensure_no_input_data():
    # 确保tbc输入的文件为空
    window_region_输入= (43,305,300,20)
    while True:
        # 识别窗口中的文本
        recognized_text = read_text_from_window(window_region_输入)
        if recognized_text:
            move_and_click(150, 314) #单击第一个文件
            right_click_and_press_D() #删除文件
        else:
            break
    # 请替换为您希望识别的窗口区域的坐标
    window_region_输入= (142,290,300,20)
    # 识别窗口中的文本
    recognized_text = read_text_from_window(window_region_输入)

    if recognized_text:
        move_and_click(192, 300) #单击第一个文件
        right_click_and_press_D() #删除文件
    
def get_mouse_position():
    x, y = pyautogui.position()
    # 返回鼠标位置
    return x, y


pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
window_region_保存 = (1731, 507, 65, 25)
window_region_输入= (1704,986,34,21)

def main():
    # 目标文件夹路径
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    # 每次运行之前清除终端屏幕

    main()