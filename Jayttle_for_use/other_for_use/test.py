import os

# 指定文件夹路径
folder_path = r'C:\Users\juntaox\Desktop\工作'

# 存储所有文件夹名称的列表
list_names = []

# 遍历指定文件夹下的所有目录
for root, dirs, files in os.walk(folder_path):
    # 将当前目录下的文件夹名称添加到 list_names
    list_names.extend(dirs)
    # 如果只想获取第一层子目录，可以加上 break
    break  # 只获取文件夹名称，不递归进入子目录

# 打印文件夹名列表
print(list_names)
