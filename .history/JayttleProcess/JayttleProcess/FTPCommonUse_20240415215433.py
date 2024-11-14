import os
from ftplib import FTP
import json

def load_ftp_files():
    # 从 options.json 文件读取配置信息
    with open('options.json', 'r') as f:
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

        # 输出文件列表
        print("FTP服务器文件列表:")
        for file_name in file_list:
            print(file_name)

        # 下载文件到本地保存路径
        if not os.path.exists(local_save_path):
            os.makedirs(local_save_path)

        for file_name in file_list:
            remote_file_path = os.path.join(root_folder, file_name)
            local_file_path = os.path.join(local_save_path, file_name)
            with open(local_file_path, 'wb') as local_file:
                ftp.retrbinary('RETR ' + remote_file_path, local_file.write)

        print(f"文件已下载到本地路径: {local_save_path}")
        return True

    except Exception as e:
        print(f"发生错误: {e}")
        return False

    finally:
        # 关闭FTP连接
        ftp.quit()

# 调用函数加载FTP文件
success = load_ftp_files()
if success:
    print("FTP文件加载成功！")
else:
    print("FTP文件加载失败。")