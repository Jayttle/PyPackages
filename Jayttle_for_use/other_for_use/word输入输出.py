from docx import Document
import pandas as pd
from pptx import Presentation
import os
import zipfile
import win32com.client as win32

def is_valid_docx(file_path):
    """
    检查文件是否是有效的 Word 文件 (.docx)
    :param file_path: 文件路径
    :return: 如果是有效的 .docx 文件返回 True，否则返回 False
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zipf:
            zipf.testzip()
        return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile):
        return False

def is_valid_doc(file_path):
    """
    检查文件是否是有效的 Word 97-2003 格式 (.doc)
    :param file_path: 文件路径
    :return: 如果是有效的 .doc 文件返回 True，否则返回 False
    """
    try:
        # 使用 pywin32 打开 .doc 文件检查是否有效
        word = win32.Dispatch("Word.Application")
        doc = word.Documents.Open(file_path)
        doc.Close()
        return True
    except Exception:
        return False

def read_docx_file(file_path, output_file_path):
    """
    读取 .docx 文件并原样输出其内容
    :param file_path: Word 文件的路径
    :param output_file_path: 输出 Word 文件的路径
    """
    if is_valid_docx(file_path):
        try:
            # 打开并保存 Word 文件
            doc = Document(file_path)
            doc.save(output_file_path)
            print(f"Word 文件（.docx）已处理并保存: {file_path} -> {output_file_path}")
        except Exception as e:
            print(f"处理文件时出错: {file_path}，错误信息：{e}")
    else:
        print(f"无效的 .docx 文件: {file_path}")

def read_doc_file(file_path, output_file_path):
    """
    读取 .doc 文件并原样输出其内容
    :param file_path: Word 文件的路径
    :param output_file_path: 输出 Word 文件的路径
    """
    if is_valid_doc(file_path):
        try:
            # 使用 pywin32 打开 .doc 文件
            word = win32.Dispatch("Word.Application")
            doc = word.Documents.Open(file_path)
            # 保存为 .docx 格式
            doc.SaveAs(output_file_path, FileFormat=16)  # FileFormat=16 对应 .docx 格式
            doc.Close()
            print(f"Word 文件（.doc）已处理并保存: {file_path} -> {output_file_path}")
        except Exception as e:
            print(f"处理文件时出错: {file_path}，错误信息：{e}")
    else:
        print(f"无效的 .doc 文件: {file_path}")

def process_all_word_files(directory, output_dir):
    """
    处理指定文件夹下的所有 Word 文件，并生成新的文件。
    :param directory: 文件夹路径
    :param output_dir: 输出文件夹路径
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # 判断文件是否为 .docx 文件
        if filename.endswith('.docx'):
            new_filename = filename.replace('.docx', '_new.docx')
            output_file_path = os.path.join(output_dir, new_filename)
            read_docx_file(file_path, output_file_path)
        
        # 判断文件是否为 .doc 文件
        elif filename.endswith('.doc'):
            new_filename = filename.replace('.doc', '_new.docx')  # 将 .doc 文件转换为 .docx
            output_file_path = os.path.join(output_dir, new_filename)
            read_doc_file(file_path, output_file_path)

def read_ppt_file(file_path, output_file_path):
    """
    读取指定的 PowerPoint 文件并原样输出其内容
    :param file_path: PowerPoint 文件的路径
    :param output_file_path: 输出 PowerPoint 文件的路径
    """
    # 打开 PowerPoint 文件
    presentation = Presentation(file_path)
    
    # 保存为新文件
    presentation.save(output_file_path)
    print(f"PowerPoint文件已处理并保存: {file_path} -> {output_file_path}")

def process_all_ppt_files(directory, output_dir):
    """
    处理指定文件夹下的所有 PowerPoint 文件，并生成新的文件。
    :param directory: 文件夹路径
    :param output_dir: 输出文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        # 判断文件是否为 PowerPoint 文件 (.pptx)
        if filename.endswith('.pptx'):
            file_path = os.path.join(directory, filename)
            new_filename = filename.replace('.pptx', '_new.pptx')  # 修改文件名
            output_file_path = os.path.join(output_dir, new_filename)

            # 读取并保存 PowerPoint 文件
            read_ppt_file(file_path, output_file_path)

def read_and_save_excel(file_path, output_path):
    """
    读取指定的 Excel 文件并将内容保存到一个新的 Excel 文件
    :param file_path: 输入 Excel 文件的路径
    :param output_path: 输出 Excel 文件的路径
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name=None)  # sheet_name=None 会读取所有的 sheet

    # 将读取到的数据保存到新的 Excel 文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, sheet_data in df.items():
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Excel文件已处理并保存到：{output_path}")

def process_all_excel_files(directory, output_dir):
    """
    处理指定文件夹下的所有 Excel 文件，并生成新的文件。
    :param directory: 文件夹路径
    :param output_dir: 输出文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        # 判断文件是否为 Excel 文件 (.xlsx)
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            new_filename = filename.replace('.xlsx', '_new.xlsx')  # 修改文件名
            output_file_path = os.path.join(output_dir, new_filename)

            # 读取并保存 Excel 文件
            read_and_save_excel(file_path, output_file_path)

if __name__ == '__main__':
    # 指定文件夹路径
    base_path = r'C:\Users\juntaox\Desktop\工作'
    # 存储所有文件夹名称的列表
    list_names = ['25.GDP与售电量比值预测']

    # 指定文件夹路径
    for item in list_names:
        folder_path = os.path.join(base_path, item) # 原始文件夹路径
        output_dir = rf"C:\Users\juntaox\Desktop\工作out\{item}"  # 新文件夹路径

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 处理文件夹下的所有 Word 文件
        process_all_word_files(folder_path, output_dir)

        # 处理文件夹下的所有 PowerPoint 文件
        process_all_ppt_files(folder_path, output_dir)

        # 处理文件夹下的所有 Excel 文件
        process_all_excel_files(folder_path, output_dir)

        print(f"{item} 所有文件处理完成。")
