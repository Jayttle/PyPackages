import subprocess
import re

# 指定文件夹路径
folder_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\JayttleProcess"

# 构造命令
command = "python setup.py sdist bdist_wheel"

# 在指定文件夹中执行命令
try:
    # cwd参数用于指定工作目录
    subprocess.run(command, shell=True, cwd=folder_path, check=True)
    print("命令执行成功！")
except subprocess.CalledProcessError as e:
    print("命令执行失败:", e)



# 设置setup.py文件路径
setup_file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\JayttleProcess\setup.py"

# 读取setup.py文件内容
with open(setup_file_path, 'r', encoding='utf-8') as file:
    setup_content = file.read()

# 使用正则表达式匹配版本号
version_pattern = r"version='([\d.]+)'"
match = re.search(version_pattern, setup_content)

if match:
    version = match.group(1)
    print("setup.py中的version为:", version)
else:
    print("未找到版本号信息。")


# 读取 PyPI API 密钥
with open(r"D:\Program Files (x86)\Software\OneDrive\密令与账号\pypl_api.txt", "r") as api_file:
    pypl_api = api_file.read().strip()

# 构造命令
command = f"twine upload dist\JayttleProcess-{version}-py3-none-any.whl -u __token__ -p {pypl_api}"

# 在指定文件夹中执行命令
try:
    # cwd参数用于指定工作目录
    subprocess.run(command, shell=True, cwd=folder_path, check=True)
    print("命令执行成功！")
except subprocess.CalledProcessError as e:
    print("命令执行失败:", e)