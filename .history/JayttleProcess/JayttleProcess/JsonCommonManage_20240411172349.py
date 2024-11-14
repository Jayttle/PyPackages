import json
from datetime import datetime

# 数据
data = [
    {
        "型号": "ROG枪神7plus",
        "时间": datetime(2023, 2, 26).strftime("%Y-%m-%d"),
        "价格": 14999.00,
        "保修": "NULL",
        "备注": "NULL",
        "类别": "电脑"
    },
    {
        "型号": "海力士内存条16GB",
        "时间": datetime(2023, 11, 21).strftime("%Y-%m-%d"),
        "价格": 327.00,
        "保修": "NULL",
        "备注": "NULL",
        "类别": "电脑"
    },
    {
        "型号": "致态长江存储1T",
        "时间": datetime(2023, 2, 26).strftime("%Y-%m-%d"),
        "价格": 509.00,
        "保修": "NULL",
        "备注": "NULL",
        "类别": "电脑"
    }
]

# 写入 JSON 文件
with open('data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("JSON 文件已创建成功。")
