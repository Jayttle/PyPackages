import os
from ftplib import FTP
import json
import CommonDecorator as ComD

class FTPConfig:
    def __init__(self, json_file_path: str):
        # 确保 JSON 文件存在
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"指定的 JSON 文件不存在: {json_file_path}")
        
        # 从 JSON 文件中读取配置信息
        with open(json_file_path, 'r') as f:
            self.options = json.load(f)
        
        # 从配置信息中获取 FTP 配置
        ftp_options = self.options.get('FtpOptions', {})
        self.ftp_address = ftp_options.get('Address')
        self.ftp_port = ftp_options.get('Port')
        self.ftp_username = ftp_options.get('UserName')
        self.ftp_password = ftp_options.get('Password')
        self.root_folder = self.options.get('RootFolder')


    def __str__(self):
        # 返回对象的字符串表示
        return f"FTPConfig(Address={self.ftp_address}, Port={self.ftp_port}, UserName={self.ftp_username}, RootFolder={self.root_folder})"




@ComD.log_function_call
def load_ftp_files() -> None:
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

        # 如果 "tower5" 文件夹存在，进一步处理其中的文件
        if 'tower3' in file_list:
            ftp.cwd('tower3')  # 切换到 tower5 文件夹
            tower3_files = ftp.nlst()  # 获取 tower5 文件夹下的文件列表

            # 输出 tower5 文件夹下的文件数量
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


def download_files_from_ftp(ftp_config: FTPConfig, 
                            remote_folder: str, 
                            files_to_download: list[str], 
                            local_save_path: str) -> bool:
    try:
        # 连接到FTP服务器
        ftp = FTP()
        ftp.connect(ftp_config.ftp_address, ftp_config.ftp_port)
        ftp.login(ftp_config.ftp_username, ftp_config.ftp_password)

        # 切换到指定的根目录
        ftp.cwd(ftp_config.root_folder)

        # 切换到指定的远程文件夹
        ftp.cwd(remote_folder)

        # 确保本地保存路径存在
        os.makedirs(local_save_path, exist_ok=True)

        # 下载指定的文件
        for file_name in files_to_download:
            local_file_path = os.path.join(local_save_path, file_name)
            with open(local_file_path, 'wb') as local_file:
                ftp.retrbinary(f'RETR {file_name}', local_file.write)

            print(f"已下载文件: {file_name} 到本地路径: {local_file_path}")

        return True

    except Exception as e:
        print(f"发生错误: {e}")
        return False

    finally:
        # 关闭FTP连接
        ftp.quit()



def download_files_from_ftp_by_txt(ftp_config: FTPConfig, txt_file_path: str, local_save_path: str) -> bool:
    # 读取指定的 txt 文件
    with open(txt_file_path, 'r') as txt_file:
        files_to_download = txt_file.readlines()
    # 去除每行末尾的换行符
    files_to_download = [file.strip() for file in files_to_download]

    # 使用 FTPConfig 对象下载文件
    success = download_files_from_ftp(ftp_config, remote_folder='towerbase1', files_to_download=files_to_download, local_save_path=local_save_path)
    return success

# 示例使用
json_file_path = r'D:\Program Files (x86)\Software\OneDrive\PyPackages\options.json'
ftp_config = FTPConfig(json_file_path)
txt_file_path = r'D:\Ropeway\GNSS\towerbase1_files.txt'
local_save_path = r'D:\Ropeway\GNSS\FTP\B011'

if download_files_from_ftp_by_txt(ftp_config, txt_file_path, local_save_path):
    print("文件下载成功！")
else:
    print("文件下载失败。")