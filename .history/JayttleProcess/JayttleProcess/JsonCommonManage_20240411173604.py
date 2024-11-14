import json
from datetime import datetime

# 指定保存路径
file_path = r"D:\Program Files (x86)\Software\OneDrive\设备清单.json"

# 将数据写入 JSON 文件
with open(file_path, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)