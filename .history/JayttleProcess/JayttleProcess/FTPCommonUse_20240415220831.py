import os
from ftplib import FTP
import json

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
        print(f"FTP服务器中有 {len(file_list)} 个文件。")

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
