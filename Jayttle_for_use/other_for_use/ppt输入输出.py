from pptx import Presentation
import os

def read_ppt_file(file_path, output_file_path):
    """
    读取指定的 PowerPoint 文件并原样输出其内容
    :param file_path: PowerPoint 文件的路径
    """
    # 打开 PowerPoint 文件
    presentation = Presentation(file_path)
    
    # 保存为新文件
    presentation.save(output_file_path)

def process_all_ppt_files(directory, output_dir):
    """
    处理指定文件夹下的所有 PowerPoint 文件，并生成新的文件。
    :param directory: 文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        # 判断文件是否为 PowerPoint 文件 (.pptx)
        if filename.endswith('.pptx'):
            file_path = os.path.join(directory, filename)

            # 生成新的输出文件路径，文件名加上 '_new'
            new_filename = filename.replace('.pptx', '_1127.pptx')
            output_file_path = os.path.join(output_dir, new_filename)

            # 读取并保存 PowerPoint 文件
            read_ppt_file(file_path, output_file_path)
            print(f"Processed: {file_path} -> {output_file_path}")

if __name__ == '__main__':
    # 指定文件夹路径
    folder_path = r"C:\Users\juntaox\Desktop\工作\30.电能计量诉求-营销任务单工作总结\原始材料"  # 你可以修改为目标文件夹路径
    output_dir = r"C:\Users\juntaox\Desktop\工作\30.电能计量诉求-营销任务单工作总结\工作总结拆分" 
    
    # 处理文件夹下的所有 PowerPoint 文件
    process_all_ppt_files(folder_path, output_dir)
