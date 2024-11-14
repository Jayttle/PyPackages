import json
from datetime import datetime


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)
    
# 数据
data = {
    "电脑": [
        {
            "型号": "ROG枪神7plus",
            "时间": datetime(2023, 2, 26),
            "价格": 14999.00,
            "保修": None,
            "备注": None
        },
        {
            "型号": "海力士内存条16GB",
            "时间": datetime(2023, 11, 21),
            "价格": 327.00,
            "保修": None,
            "备注": None
        },
        {
            "型号": "致态长江存储1T",
            "时间": datetime(2023, 2, 26),
            "价格": 509.00,
            "保修": None,
            "备注": None
        }
    ],
    "手机": [
        {
            "型号": "小米11Ultra",
            "时间": datetime(2022, 8, 4),
            "价格": 3500,
            "保修": None,
            "备注": "碎屏2024-03-24"
        }
    ],
    "平板": [
        {
            "型号": "ipad2018",
            "时间": "约2019-07",
            "价格": "约7000",
            "保修": None,
            "备注": None
        },
        {
            "型号": "平板套",
            "时间": datetime(2020, 11, 1),
            "价格": 53.00,
            "保修": None,
            "备注": None
        },
        {
            "型号": "pencil二代",
            "时间": datetime(2022, 3, 15),
            "价格": 768.00,
            "保修": None,
            "备注": None
        },
        {
            "型号": "pencil二代",
            "时间": datetime(2019, 8, 22),
            "价格": 880.00,
            "保修": None,
            "备注": "已经换新"
        }
    ],
    "耳机": [
        {
            "型号": "Sony WF-1000XM5",
            "时间": datetime(2024, 4, 11),
            "价格": 1399.00,
            "保修": None,
            "备注": None
        },
        {
            "型号": "Sony WH-1000XM3",
            "时间": datetime(2022, 9, 22),
            "价格": 937.00,
            "保修": None,
            "备注": None
        },
        {
            "型号": "Sony WF-1000XM3",
            "时间": "约2023-02-10",
            "价格": "约300.00",
            "保修": None,
            "备注": "电量不足"
        }
    ],
    "键鼠": [
        {
            "型号": "新盟M75",
            "时间": datetime(2024, 4, 7),
            "价格": 211.00,
            "保修": None,
            "备注": None
        },
        {
            "型号": "鼠标垫盗版卓威",
            "时间": datetime(2024, 2, 28),
            "价格": 24.90,
            "保修": None,
            "备注": None
        },
        {
            "型号": "罗技M650L",
            "时间": datetime(2023, 8, 30),
            "价格": 183.00,
            "保修": None,
            "备注": None
        },
        {
            "型号": "罗技M221",
            "时间": datetime(2023, 5, 27),
            "价格": 60.05,
            "保修": None,
            "备注": "几乎不用"
        },
        {
            "型号": "小熊卡通键盘手托",
            "时间": datetime(2022, 10, 15),
            "价格": 39.8,
            "保修": None,
            "备注": "几乎不用"
        },
        {
            "型号": "牧马人键盘",
            "时间": datetime(2022, 2, 5),
            "价格": 149.00,
            "保修": None,
            "备注": "坏了基本"
        }
    ],
    "显示器": [
        {
            "型号": "AOC27寸2k",
            "时间": datetime(2023, 7, 20),
            "价格": 1491.5,
            "保修": None,
            "备注": None
        },
        {
            "型号": "DELL S2721DGF",
            "时间": "约2022-08-04",
            "价格": "约2800",
            "保修": None,
            "备注": None
        },
        {
            "型号": "ULT-unite HDMI线(2米)",
            "时间": datetime(2023, 10, 5),
            "价格": 25.32,
            "保修": None,
            "备注": None
        },
        {
            "型号": "ULT-unite HDMI线(1米)",
            "时间": datetime(2023, 7, 20),
            "价格": 15.00,
            "保修": None,
            "备注": None
        }
    ],
    "其他": [
        {
            "型号": "飞利浦音响",
            "时间": datetime(2022, 7, 2),
            "价格": 195.00,
            "保修": None,
            "备注": "几乎不用"
        },
        {
            "型号": "小米MI显示器挂灯",
            "时间": datetime(2022, 6, 30),
            "价格": 179,
            "保修": None,
            "备注": "几乎不用"
        },
        {
            "型号": "笔记本支架绿巨能",
            "时间": datetime(2022, 3, 18),
            "价格": 39.9,
            "保修": None,
            "备注": "几乎不用"
        }
    ]
}
# 使用自定义编码器将数据转换为 JSON 格式
json_data = json.dumps(data, cls=DateTimeEncoder, ensure_ascii=False)