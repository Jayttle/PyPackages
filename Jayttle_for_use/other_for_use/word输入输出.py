from docx import Document
import os

def read_word_file(file_path):
    """
    读取指定的 Word 文件并原样输出其内容
    :param file_path: Word 文件的路径
    """
    # 打开 Word 文件
    doc = Document(file_path)
    doc.save(r"test.docx")
if __name__ == '__main__':
    # 获取当前工作目录
    current_directory = os.getcwd()

    # 定义文件路径为当前目录下的 'word_file.docx'
    file_path = os.path.join(current_directory, r"C:\Users\juntaox\Desktop\工作\23.浦东四大重点区域负荷分析\原始材料\浦东新区重点经济圈用电需求专题研究10.28-修改.docx")

    # 读取并输出 Word 文件的内容
    read_word_file(file_path)