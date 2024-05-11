import subprocess
import re

def execute_command(cmd: str) -> bool:
    """执行命令并返回是否成功"""
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()

        # 打印标准输出
        if stdout:
            print(stdout.decode())

        if stderr:
            print(stderr.decode())

        # 检查命令是否执行成功
        if process.returncode == 0:
            print("Command executed successfully")
            return True
        else:
            print(f"Command execution failed")
            return False
    except Exception as e:
        print(f"Error occurred while executing command: {e}")
        return False
    
# 读取 setup.py 文件
with open("setup.py", "r", encoding="utf-8") as file:
    setup_code = file.read()

# 使用正则表达式提取版本信息
match = re.search(r"version=['\"]([^'\"]+)['\"]", setup_code)
if match:
    version = match.group(1)
    print(f"Version: {version}")

# 执行命令: python setup.py sdist bdist_wheel
cmd = "python setup.py sdist bdist_wheel"
success = execute_command(cmd)


# 读取 PyPI API 密钥
with open(r"D:\Program Files (x86)\Software\OneDrive\密令与账号\pypl_api.txt", "r") as api_file:
    pypl_api = api_file.read().strip()

# 执行上传操作
if success:
    cmd = rf"twine upload dist\JayttleProcess-{version}-py3-none-any.whl -u __token__ -p {pypl_api}" 
    print(cmd)
