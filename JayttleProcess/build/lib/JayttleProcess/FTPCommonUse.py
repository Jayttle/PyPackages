import os
from ftplib import FTP
import json
import datetime
from dataclasses import dataclass
import JayttleProcess.CommonDecorator as ComD
from JayttleProcess import RinexCommonManage
from JayttleProcess import ComputerControl
from JayttleProcess.RinexCommonManage import *


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
def check_FTP_file() -> None:
    # 从 options.json 文件读取配置信息
    with open(r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\options.json', 'r') as f:
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

    all_files = []  # 存储所有文件名

    try:
        # 切换到指定的根目录
        ftp.cwd(root_folder)

        # 获取FTP服务器上的文件列表
        file_list = ftp.nlst()

        # 输出文件数量
        print(f"FTP服务器中有 {len(file_list)} 个文件夹和文件:")
        for item in file_list:
            print(item)

        # 遍历文件夹
        for folder_name in ['tower3', 'tower5', 'tower7', 'tower8', 'towerbase1', 'towerbase2']:
            if folder_name in file_list:

                ftp.cwd(folder_name)  # 切换到对应文件夹
                folder_files = ftp.nlst()  # 获取文件夹下的文件列表


                print(f"{folder_name} 文件夹中有 {len(folder_files)} 个文件。")

                # 将文件夹下的文件名逐行保存到本地的 txt 文件中
                txt_file_path = os.path.join(local_save_path, f'{folder_name}_files.txt')
                with open(txt_file_path, 'w') as txt_file:
                    for file_name in folder_files:
                        txt_file.write(file_name + '\n')
                        all_files.append(file_name)  # 添加到 all_files 列表中

                print(f"{folder_name} 文件夹中的文件名已保存到本地: {txt_file_path}")
                # 返回上一级文件夹
                ftp.cwd('..')

        # 将所有文件名写入 all_files.txt 文件
        all_files_path = os.path.join(local_save_path, 'all_files.txt')
        with open(all_files_path, 'w') as all_files_txt:
            for file_name in all_files:
                all_files_txt.write(file_name + '\n')

        print(f"所有文件名已保存到本地: {all_files_path}")

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
            # 检查文件是否已经存在于本地路径中
            # 检查文件是否已经存在于本地路径中，且替换后的文件也不存在
            if not os.path.exists(local_file_path) and not os.path.exists(local_file_path.replace(".crx.Z", ".rnx").replace(".rnx.Z", ".rnx")):
                with open(local_file_path, 'wb') as local_file:
                    ftp.retrbinary(f'RETR {file_name}', local_file.write)

                print(f"已下载文件: {file_name} 到本地路径: {local_file_path}")
            else:
                print(f"文件 {file_name} 已存在于本地路径: {local_file_path}, 跳过下载")

        return True

    except Exception as e:
        print(f"发生错误: {e}")
        return False

    finally:
        # 关闭FTP连接
        ftp.quit()



def process_string(s: str) -> str:
    if len(s) == 4:
        if s.startswith('B'):
            return 'towerbase' + s[2]
        elif s.startswith('R'):
            return 'tower' + s[2]
    return None



def download_files_from_ftp_by_txt(ftp_config: FTPConfig, txt_file_path: str, local_save_path: str) -> bool:
    # 读取指定的 txt 文件
    with open(txt_file_path, 'r') as txt_file:
        files_to_download = txt_file.readlines()
    # 去除每行末尾的换行符
    files_to_download = [file.strip() for file in files_to_download]
    # 获取文件名的前四个字符并处理
    target_folder = process_string(os.path.basename(txt_file_path)[:4])
    # 使用 FTPConfig 对象下载文件
    success = download_files_from_ftp(ftp_config, remote_folder=target_folder, files_to_download=files_to_download, local_save_path=local_save_path)
    return success

def copy_files(from_folder: str, to_folder: str, prefix: str) -> None:
    """将指定文件夹中以指定前缀开头的文件复制到目标文件夹中."""
    for file_name in os.listdir(from_folder):
        if file_name.startswith(prefix):
            source_file = os.path.join(from_folder, file_name)
            destination_file = os.path.join(to_folder, file_name)
            # 检查目标文件夹中是否已经存在相同的文件
            if not os.path.exists(destination_file):
                shutil.copy(source_file, destination_file)

@ComD.log_function_call
def Process_Copy(isFirst: bool,from_copy_merge_folder_list: list[str] , toDownload_path: str, merge_path: str) -> None:
    """处理第一部分数据."""
    if not isFirst:
        for filename in os.listdir(toDownload_path):
            # 获取文件名的前四个字符
            first_four_characters = filename[:4]
            if first_four_characters in ("B011", "B021"):  # 使用 in 检查多个条件
                toDownload_file = os.path.join(toDownload_path, filename)
                # 读取指定的 txt 文件
                with open(toDownload_file, 'r') as txt_file:
                    files_to_download: list[str] = txt_file.readlines()
                # 去除每行末尾的换行符
                files_to_download = [file.strip() for file in files_to_download]
                for from_copy_folder in from_copy_merge_folder_list:
                    print(f"from_copy_folder:{from_copy_folder}") 
                    merge_folders: list[str] = os.listdir(from_copy_folder)
                    file_data = {}
                    # 遍历 files_to_download 列表并检查是否需要移除元素
                    for file_to_download in files_to_download.copy():
                        third_part: str = file_to_download.split('_')[2]  # 获取文件名中的第三部分
                        first_part: str = file_to_download.split('_')[0][:4]
                        isNeedRemove: bool = False
                        if third_part in merge_folders and first_part == first_four_characters:
                            isNeedRemove = True
                            from_copy_merge_folder: str = os.path.join(from_copy_folder, third_part)
                            to_merge_folder = os.path.join(merge_path, third_part)
                            os.makedirs(to_merge_folder, exist_ok=True)
                            copy_files(from_copy_merge_folder, to_merge_folder, first_four_characters)
                        if not isNeedRemove:
                            date_part: str = file_to_download.split('_')[2][:7]
                            if date_part not in file_data:
                                file_data[date_part] = []
                            file_data[date_part].append(file_to_download.strip())
                    
                    # 删除长度不满指定个数的条目
                    specified_length = 8  # 指定的长度
                    to_remove = [date_part for date_part, files_list in file_data.items() if len(files_list) < specified_length]
                    for date_part in to_remove:
                        del file_data[date_part]

                    # 将更新后的 file_data 保存回原始的文本文件中
                    with open(toDownload_file, 'w') as txt_file:
                        for files_list in file_data.values():
                            txt_file.write('\n'.join(files_list) + '\n')
# TODO: COPY逻辑改善



@ComD.log_function_call
def Process_Part1(output_file_path, specified_marker_names, start_hour, end_hour):
    """
    1.完成  _toDownload.txt中存放需下载文件
    """
    # 指定文件夹
    directory_path = r"D:\Ropeway\GNSS\FTP_File_Situation"
    # 要读取得所有FTP文件的情况
    file_list_path = os.path.join(directory_path, "all_files.txt")
    # 读取rnx文件信息 list[RinexFileInfo]
    rinex_files_info: list[RinexFileInfo] = RinexCommonManage.read_rinex_files_info(file_list_path)

    # # 指定时间范围
    # start_hour= 16
    # end_hour = 19
    # 统计每天指定时间范围内文件的数量，并按日期和标记站名返回文件数的字典。
    files_in_hour_range: dict[datetime.date, dict[str, int]] = RinexCommonManage.count_files_in_hour_range(rinex_files_info, start_hour, end_hour)
    # 要检查的文件数
    specified_count = (end_hour - start_hour + 1) * 2
    # 查找指定标记站名文件数同时为8的日期列表
    dates_with_specific_count:list[datetime.date] = RinexCommonManage.find_dates_with_specific_file_count(files_in_hour_range, specified_marker_names, specified_count)
    """
    dates_with_specific_count 就是想要下载的日期列表
    """
    # 要下载的文件名放到 _toDownload.txt中
    for item in specified_marker_names:
        to_download_file_name = os.path.join(output_file_path, f"{item}_toDownload.txt")
        marker_name = item
        RinexCommonManage.write_rinex_files_to_txt(rinex_files_info, marker_name, dates_with_specific_count, to_download_file_name, start_hour, end_hour)


@ComD.log_function_call
def Process_Part2(toDownload_path, local_save_path):
    """
    2.完成指定文件的下载工作
    """
    # json文件
    json_file_path = r'D:\Program Files (x86)\Software\OneDrive\C#\windows_C#\Cableway.Net7\Cableway.Download\options.json'
    # 读取json文件里的FTPConfig
    ftp_config = FTPConfig(json_file_path)
    # 下载的目的地
    # 遍历文件夹中的文件
    for filename in os.listdir(toDownload_path):
        # 获取文件名的前四个字符
        first_four_characters = filename[:4]
        # 构建要保存的本地路径
        save_folder_path = os.path.join(local_save_path, first_four_characters)
        # 检查本地路径是否存在，如果不存在，则创建
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        RinexCommonManage.delete_small_files(save_folder_path, 0.1)
        if download_files_from_ftp_by_txt(ftp_config, os.path.join(toDownload_path, filename), save_folder_path):
            print("文件下载成功！")
        else:
            print("文件下载失败。")


@ComD.log_function_call
def Process_Part3(local_save_path):
    """
    3.解压缩 .Z格式文件
    4.文件转换 CRX2RNX.exe
    """
    for foldername in os.listdir(local_save_path):
        save_folder_path = os.path.join(local_save_path, foldername)
        # 解压缩文件
        RinexCommonManage.unzip_folder_path(folder_path = save_folder_path)
        print("文件解压成功！")
        # 读取文件夹中crx格式的文件 并且没有同名的rnx格式的文件
        to_convert_crx_files: list[RinexFileInfo] = RinexCommonManage.get_rnx_files_crx(save_folder_path)
        # CRX2RNX 文件转换
        RinexCommonManage.crx_to_rnx(to_convert_crx_files)
        print("文件转换成功！")

@ComD.log_function_call
def Process_Part4(local_save_path, merge_path):
    """
    5.合并rnx文件
    6.对MO文件 markername的修改
    executed in 3999.1432s
    """
    for foldername in os.listdir(local_save_path):
        save_folder_path = os.path.join(local_save_path, foldername)    
        # 用日期来作为key值,将同一天的数据存放在一起
        rnx_files: dict[datetime.date, list[RinexFileInfo]] = RinexCommonManage.get_rnx_files_dict_date(save_folder_path)
        to_process_rnx_file: list[list[RinexFileInfo]] = []
        for start_date, rinex_list in rnx_files.items():
            o_files: list[RinexFileInfo] = [rnx_file for rnx_file in rinex_list if rnx_file.file_type == RinexFileType.O]
            n_files: list[RinexFileInfo] = [rnx_file for rnx_file in rinex_list if rnx_file.file_type == RinexFileType.N]
            to_process_rnx_file.append(o_files)
            to_process_rnx_file.append(n_files)
        # 合并文件
        RinexCommonManage.merge_files_threadpool(to_process_rnx_file, merge_path, 4)
        # 遍历合并文件的保存地址 下的所有文件夹里面的 _MO.rnx文件预处理
        RinexCommonManage.process_rnx_files(merge_path)
    

@ComD.log_function_call
def Process_Check(toDownload_path, local_save_path):
    file_lines_mapping = RinexCommonManage.count_lines_in_each_txt_file(toDownload_path)
    for file_name, num_lines in file_lines_mapping.items():
        print(f"文件 '{file_name}' 的行数为: {num_lines}")
    folder_file_count = RinexCommonManage.count_files_in_folders(local_save_path)
    for folder_name, file_count in folder_file_count.items():
        print(f"文件夹 '{folder_name}' 中包含 {file_count} 个文件.")

def process_rnx_files1(folder_path: str) -> None:
    """
    对每个文件夹中末尾是 _MO.rnx 的文件执行指定操作，将第四行数据的前四个字符修改为文件名的前四个字符。
    在修改之前检查第四行数据的前四个字符是否已经等于文件名的前四个字符。如果是，则跳过该文件
    """
    # 获取目标文件夹中所有文件夹的名字
    subdirectories = os.listdir(folder_path)
    # 存储符合条件的文件名的字典，键是目录路径，值是文件名列表
    rnx_files = {}
    
    for directory in subdirectories:
        # 构建文件夹的完整路径
        directory_path = os.path.join(folder_path, directory)
        # 检查路径是否是文件夹
        if os.path.isdir(directory_path):
            # 初始化当前目录的文件名列表
            rnx_files[directory_path] = []
            # 遍历文件夹中的文件
            for file_name in os.listdir(directory_path):
                # 检查文件是否是以 "_MO.rnx" 结尾的文件
                if file_name.endswith("_MO.rnx"):
                    if int(file_name[16:19]) < 216 and int(file_name[12:16]) == 2023:
                        # 添加符合条件的文件名到文件名列表
                        rnx_files[directory_path].append(file_name)

    # 对字典中每个目录的文件名列表进行排序
    for directory, files in rnx_files.items():
        files.sort()

    # 遍历字典，处理每个目录中的文件
    for directory, files in rnx_files.items():
        for file in files:
            file_path = os.path.join(directory, file)
            print(f"directory={directory}\tfile={file}")

@ComD.log_function_call
def Process_in_one_step():
    base_marker_names = ['B011', 'B021']
    root_folder = r"D:\Ropeway\MySQL"
    # to_process_marker_names = ['R052', 'R071']
    to_process_marker_names = ['R032', 'R051', 'R052', 'R071', 'R072', 'R081', 'R082']
    # 指定时间范围
    start_hour= 4
    end_hour = 7
    TBC_Process = False
    isFirstProcess = True
    from_copy_merge_folder_list = []
    for item in to_process_marker_names:
        folder_name = f"{item}_{start_hour}{end_hour}"
        folder_path = os.path.join(root_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        toDownload_folder = os.path.join(folder_path, 'toDownload')
        os.makedirs(toDownload_folder, exist_ok=True)
        FTP_folder = os.path.join(folder_path, 'FTP')
        os.makedirs(FTP_folder, exist_ok=True)
        FTPMerge_folder = os.path.join(folder_path, 'FTPMerge')
        os.makedirs(FTPMerge_folder, exist_ok=True)
        TBC_folder = os.path.join(folder_path, 'TBC')
        os.makedirs(TBC_folder, exist_ok=True)
        base_marker_names.append(item)
        Process_Part1(toDownload_folder, base_marker_names, start_hour, end_hour)

        from_copy_merge_folder_list.append(FTPMerge_folder)
        Process_Copy(isFirstProcess, from_copy_merge_folder_list, toDownload_folder ,FTPMerge_folder)

        Process_Part2(toDownload_folder, FTP_folder)
        # Process_Part3(FTP_folder)
        # Process_Part4(FTP_folder, FTPMerge_folder)
        # Process_Check(toDownload_folder, FTP_folder)
        if TBC_Process:
            ComputerControl.TBC_auto_Process(FTPMerge_folder, folder_name)
        base_marker_names.remove(item)
        isFirstProcess = False

#Process_in_one_step()

if __name__=='__main__':
    print('-----------run-----------')