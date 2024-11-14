import math
import random
from datetime import datetime, timedelta

class TimeSeriesData:
    def __init__(self, value, datetime):
        self.value = value
        self.datetime = datetime

    def __str__(self):
        return f"Value: {self.value}, Datetime: {self.datetime}"

    def kendall_change_point_detection(self):
        """时序数据的kendall突变点检测"""
        n = len(self.value)
        Sk = [0]
        UFk = [0]
        s = 0
        Exp_value = [0]
        Var_value = [0]

        for i in range(1, n):
            for j in range(i):
                if self.value[i] > self.value[j]:
                    s += 1
            Sk.append(s)
            Exp_value.append((i + 1) * (i + 2) / 4.0)
            Var_value.append((i + 1) * i * (2 * (i + 1) + 5) / 72.0)
            UFk.append((Sk[i] - Exp_value[i]) / math.sqrt(Var_value[i]))

        Sk2 = [0]
        UBk = [0]
        UBk2 = [0]
        s2 = 0
        Exp_value2 = [0]
        Var_value2 = [0]
        input_data_t = list(reversed(self.value))

        for i in range(1, n):
            for j in range(i):
                if input_data_t[i] > input_data_t[j]:
                    s2 += 1
            Sk2.append(s2)
            Exp_value2.append((i + 1) * (i + 2) / 4.0)
            Var_value2.append((i + 1) * i * (2 * (i + 1) + 5) / 72.0)
            UBk.append((Sk2[i] - Exp_value2[i]) / math.sqrt(Var_value2[i]))
            UBk2.append(-UBk[i])

        UBkT = list(reversed(UBk2))
        diff = [x - y for x, y in zip(UFk, UBkT)]
        K = []

        for k in range(1, n):
            if diff[k - 1] * diff[k] < 0:
                K.append(self.value[k])

        return K








def kendall_change_point_detection(input_data):
    """时序数据的kendall突变点检测"""
    n = len(input_data)
    Sk = [0]
    UFk = [0]
    s = 0
    Exp_value = [0]
    Var_value = [0]

    for i in range(1, n):
        for j in range(i):
            if input_data[i].value > input_data[j].value:
                s += 1
        Sk.append(s)
        Exp_value.append((i + 1) * (i + 2) / 4.0)
        Var_value.append((i + 1) * i * (2 * (i + 1) + 5) / 72.0)
        UFk.append((Sk[i] - Exp_value[i]) / math.sqrt(Var_value[i]))

    Sk2 = [0]
    UBk = [0]
    UBk2 = [0]
    s2 = 0
    Exp_value2 = [0]
    Var_value2 = [0]
    input_data_t = list(reversed(input_data))

    for i in range(1, n):
        for j in range(i):
            if input_data_t[i].value > input_data_t[j].value:
                s2 += 1
        Sk2.append(s2)
        Exp_value2.append((i + 1) * (i + 2) / 4.0)
        Var_value2.append((i + 1) * i * (2 * (i + 1) + 5) / 72.0)
        UBk.append((Sk2[i] - Exp_value2[i]) / math.sqrt(Var_value2[i]))
        UBk2.append(-UBk[i])

    UBkT = list(reversed(UBk2))
    diff = [x - y for x, y in zip(UFk, UBkT)]
    K = []

    for k in range(1, n):
        if diff[k - 1] * diff[k] < 0:
            K.append(input_data[k])

    # 绘图代码可以在这里添加，如果需要的话

    return K

# 生成随机的 TimeSeriesData 数据
def generate_random_data():
    """生成随机的 TimeSeriesData 数据"""
    start_date = datetime(2024, 3, 13, 0, 0, 0)
    data = []

    for i in range(50):
        value = random.randint(10, 30)
        current_date = start_date + timedelta(hours=i)
        data_point = TimeSeriesData(value=value, datetime=current_date)
        data.append(data_point)

    return data

# 创建TimeSeriesData实例
data_points = generate_random_data()

change_points = kendall_change_point_detection(data_points)

# 打印突变点
for point in change_points:
    print(point)