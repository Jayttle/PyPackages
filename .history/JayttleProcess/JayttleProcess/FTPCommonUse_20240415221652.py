import os
from ftplib import FTP
import json
import CommonDecorator as ComD
import RinexCommonManage


@ComD.log_function_call
def text(i:int = 1):
    return i

@ComD.log_function_call
def load_ftp_files():
    # 从 options.json 文件读取配置信息
    with open(r'D:\Program Files (x86)\Software\OneDrive\PyPackages\options.json', 'r') as f:
        options = json.load(f)

    # FTP服务器的地址和端口
    ftp_address = options['FtpOptions']['Address']
    ftp_port = options['FtpOptions']['Port']
    ftp_username = options['FtpOptions']['UserName']
    ftp_password = options['FtpOptions']['Password']
    root_folder = options['RootFolder']
    local_save_path = options['LocalSavePath']

    # 连接到FTP服务器
    ftp = FTP()
    ftp.connect(ftp_address, ftp_port)
    ftp.login(ftp_username, ftp_password)

    try:
        # 切换到指定的根目录
        ftp.cwd(root_folder)

        # 获取FTP服务器上的文件列表
        file_list = ftp.nlst()

        # 输出文件数量
        print(f"FTP服务器中有 {len(file_list)} 个文件夹和文件:")
        for item in file_list:
            print(item)

        # 如果 "tower3" 文件夹存在，进一步处理其中的文件
        if 'tower3' in file_list:
            ftp.cwd('tower3')  # 切换到 tower3 文件夹
            tower3_files = ftp.nlst()  # 获取 tower3 文件夹下的文件列表

            # 输出 tower3 文件夹下的文件数量
            print(f"tower3 文件夹中有 {len(tower3_files)} 个文件。")

            # 将 tower3 文件夹下的文件名逐行保存到本地的 txt 文件中
            txt_file_path = os.path.join(local_save_path, 'tower3_files.txt')
            with open(txt_file_path, 'w') as txt_file:
                for file_name in tower3_files:
                    txt_file.write(file_name + '\n')

            print(f"tower3 文件夹中的文件名已保存到本地: {txt_file_path}")

        return True

    except Exception as e:
        print(f"发生错误: {e}")
        return False

    finally:
        # 关闭FTP连接
        ftp.quit()

# 调用函数连接FTP并获取文件数量
success = load_ftp_files()
if success:
    print("FTP连接成功！")
else:
    print("FTP连接失败。")
