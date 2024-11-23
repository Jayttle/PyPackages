from docx import Document
import os

def read_word_file(file_path, output_file_path):
    """
    读取指定的 Word 文件并原样输出其内容
    :param file_path: Word 文件的路径
    """
    # 打开 Word 文件
    doc = Document(file_path)
    doc.save(output_file_path)

    
if __name__ == '__main__':
    # 获取当前工作目录
    current_directory = os.getcwd()

    # 定义文件路径为当前目录下的 'word_file.docx'
    file_path = os.path.join(current_directory, r"C:\Users\juntaox\Desktop\硬件IT基础知识培训.docx")

    # 读取并输出 Word 文件的内容
    read_word_file(file_path,  r"C:\Users\juntaox\Desktop\硬件IT基础知识培训_1.docx")