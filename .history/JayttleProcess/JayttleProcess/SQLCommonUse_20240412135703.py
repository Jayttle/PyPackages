import pymysql
import functools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Union

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 记录函数调用时间
        call_time = datetime.now()
        start_time = time.time()  # 记录函数开始执行的时间

        print(f"Function '{func.__name__}' called at {call_time}")

        # 记录传入的参数
        args_str = ', '.join(map(repr, args))
        kwargs_str = ', '.join(f"{key}={value!r}" for key, value in kwargs.items())
        all_args = ', '.join(filter(None, [args_str, kwargs_str]))
        print(f"Arguments: {all_args}")

        # 调用函数并记录返回值
        result = func(*args, **kwargs)
        print(f"Returned: {result}")
        end_time = time.time()  # 记录函数执行完毕的时间
        print(f"executed in {(end_time - start_time):.4f}s")  # 打印执行时间
        print()
        return result

    return wrapper


def cache_results(func):
    cache = {}  # 创建一个字典来存储之前的调用结果
    def wrapper(*args, **kwargs):
        # 将参数转换为可哈希的形式，以便用作字典的键
        cache_key = (args, tuple(sorted(kwargs.items())))
        if cache_key in cache:  # 如果缓存中有这个键，直接返回对应的值
            print("Returning cached result for", cache_key, "cache[cache_key]: ", cache[cache_key])
            return cache[cache_key]
        else:  # 否则，调用函数并存储结果到缓存中
            result = func(*args, **kwargs)
            cache[cache_key] = result
            return result
    return wrapper


@cache_results
def expensive_function(x, y):
    # 模拟一个耗时操作
    time.sleep(4)  # 假装这里有复杂计算
    return x + y


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始执行的时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录函数执行完毕的时间
        print(f"Function {func.__name__!r} executed in {(end_time - start_time):.4f}s")  # 打印执行时间
        return result
    return wrapper


@log_function_call
def some_function(x):
    # 模拟一个耗时操作
    time.sleep(x)
    return x


def catch_exceptions(default_value=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"An exception occurred in {func.__name__}: {e}")
                return default_value
        return wrapper
    return decorator


@catch_exceptions(default_value="Error occurred")
def risky_function(x):
    # 这里是一些可能抛出异常的代码
    return 1 / x


SQL_CONFIG = {
    "host": "localhost",
    "user": "Jayttle",
    "password": "@JayttleRoot",
    "database": "jayttle"
}


def execute_sql(sql_statement: str) -> Union[str, list[tuple]]:
    # 建立数据库连接
    conn = pymysql.connect(**SQL_CONFIG)
    cursor = conn.cursor()
    
    try:
        # 执行输入的 SQL 语句
        cursor.execute(sql_statement)
        
        # 如果是查询语句，则返回查询结果
        if sql_statement.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            return results
        else:
            # 提交更改
            conn.commit()
            return "SQL statement executed successfully!"

    except Exception as e:
        # 发生错误时回滚
        conn.rollback()
        return "Error executing SQL statement: " + str(e)

    finally:
        # 关闭游标和数据库连接
        cursor.close()
        conn.close()


def execute_sql_params(sql_statement: str, values: tuple) -> None:
    """执行带参数的 SQL 语句"""
    # 建立数据库连接
    conn = pymysql.connect(**SQL_CONFIG)
    cursor = conn.cursor()
    
    try:
        # 执行输入的 SQL 语句
        cursor.execute(sql_statement, values)
        
        # 如果是查询语句，则打印查询结果
        if sql_statement.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            for row in results:
                print(row)
        else:
            # 提交更改
            conn.commit()
            print("SQL statement executed successfully!")

    except Exception as e:
        # 发生错误时回滚
        conn.rollback()
        print("Error executing SQL statement:", e)

    finally:
        # 关闭游标和数据库连接
        cursor.close()
        conn.close()


def create_database(database_name: str) -> None:
    """新建名为database_name的database"""
    conn = None
    cursor = None
    try:
        # 更新数据库配置信息的数据库名称
        SQL_CONFIG["database"] = ""

        # 连接MySQL数据库
        conn = pymysql.connect(**SQL_CONFIG)

        # 创建一个游标对象
        cursor = conn.cursor()

        # 新建数据库的SQL语句
        create_database_query = f"CREATE DATABASE IF NOT EXISTS {database_name}"

        # 执行SQL语句
        cursor.execute(create_database_query)
        print("New database created successfully.")

    except Exception as e:
        print("Error creating new database:", e)

    finally:
        # 在关闭之前检查变量是否已被赋值
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def list_databases() -> None:
    """列出database名"""
    conn = None
    cursor = None
    try:
        # 连接MySQL数据库
        conn = pymysql.connect(**SQL_CONFIG)

        # 创建一个游标对象
        cursor = conn.cursor()

        # 查询所有数据库的SQL语句
        show_databases_query = "SHOW DATABASES"

        # 执行SQL语句
        cursor.execute(show_databases_query)

        # 获取查询结果
        databases = cursor.fetchall()

        # 打印数据库列表
        print("Databases:")
        for database in databases:
            print(database[0])

    except Exception as e:
        print("Error listing databases:", e)

    finally:
        # 关闭游标和数据库连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def change_database() -> None:
    """改变配置中的database"""
    conn = None
    cursor = None
    try:
        # 连接MySQL数据库
        conn = pymysql.connect(**SQL_CONFIG)

        # 创建一个游标对象
        cursor = conn.cursor()

        # 查询所有数据库的SQL语句
        show_databases_query = "SHOW DATABASES"

        # 执行SQL语句
        cursor.execute(show_databases_query)

        # 获取查询结果
        databases = cursor.fetchall()

        # 打印数据库列表
        print("Databases:")
        for database in databases:
            print(database[0])

        # 提示用户输入指定的数据库名称
        target_database = input("Enter the name of the database you want to update: ")

        # 更新数据库配置信息的数据库名称
        SQL_CONFIG["database"] = target_database

        print(f"Database updated to: {target_database}")

    except Exception as e:
        print("Error listing or updating databases:", e)

    finally:
        # 关闭游标和数据库连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def print_sql_config() -> None:
    """调用函数查询数据库列表"""
    print("SQL_CONFIG = {")
    print("\t\"host\": \"localhost\",")
    print("\t\"user\": \"Jayttle\",")
    print("\t\"password\": \"@JayttleRoot\",")
    print("\t\"database\": \"jayttle\"")
    print("}")


def create_table(listName: str) -> None:
    """创建表的 SQL 语句 """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS {0}} (
        Time DATETIME NOT NULL,
        StationID INT NOT NULL,
        Temperature FLOAT,
        Humidness FLOAT,
        Pressure FLOAT,
        WindSpeed FLOAT,
        WindDirection VARCHAR(20),
        PRIMARY KEY (Time, StationID)
    )
    """.format(listName)
    execute_sql(create_table_query)


def create_table_id() -> None:
    """创建表的 SQL 语句 """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS met_id (
        StationID INT NOT NULL,
        Name VARCHAR(255) NOT NULL,
        PRIMARY KEY (StationID)
    )
    """
    execute_sql(create_table_query)


def delete_data(station_id: int) -> None:
    """删除数据"""
    delete_query = """
    DELETE FROM met_id
    WHERE StationID = %s
    """
    execute_sql_params(delete_query, (station_id,))


def select_data() -> None:
    """查询数据"""
    select_query = """
    SELECT * FROM met_id
    """
    execute_sql(select_query)


def update_data(station_id: int, new_name: str) -> None:
    """更新数据"""
    update_query = """
    UPDATE met_id
    SET Name = %s
    WHERE StationID = %s
    """
    execute_sql_params(update_query, (new_name, station_id))


@log_function_call
def get_min_max_time(listName: str) -> tuple:
    # 查询 Time 列的最小值和最大值
    query = "SELECT MIN(Time) AS min_time, MAX(Time) AS max_time FROM {0};".format(listName)
    conn = pymysql.connect(**SQL_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchone()
        return result
    except Exception as e:
        print("Error executing SQL statement:", e)
        return None
    finally:
        cursor.close()
        conn.close()


@log_function_call
def query_time_difference(listName: str, StartTime: datetime, EndTime: datetime) -> tuple:
    # 构建带有参数的查询语句
    sql_statement = """
    WITH TimeDiffCTE AS (
        SELECT
            time,
            LAG(time) OVER (ORDER BY time) AS PreviousTime,
            TIMESTAMPDIFF(SECOND, LAG(time) OVER (ORDER BY time), time) AS TimeDifference
        FROM
            {0}
        WHERE
            time >= '{1}' AND time <= '{2}'
    )
    SELECT
        PreviousTime,
        time AS CurrentTime,
        TimeDifference
    FROM
        TimeDiffCTE
    WHERE
        (TimeDifference > 100 OR PreviousTime IS NULL)
        AND PreviousTime IS NOT NULL 
    ORDER BY
        time ASC;
    """.format(listName, StartTime, EndTime)

    # 执行 SQL 查询
    conn = pymysql.connect(**SQL_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_statement)
        results = cursor.fetchall()  # 获取查询结果
        return results  # 返回结果
    except Exception as e:
        print("Error executing SQL statement:", e)
        return None
    finally:
        cursor.close()
        conn.close()


@log_function_call
def extract_time_data(listName: str) -> list[tuple[datetime]]:
    # 查询 Time 属性的数据
    query = "SELECT Time FROM {0}".format(listName)
    
    # 执行查询语句
    results = execute_sql(query)
    
    return results


@log_function_call
def preprocess_time_data(time_data: list[tuple[datetime]]) -> list[datetime]:
    processed_time_data = []
    current_year = datetime.now().year  # 获取当前年份

    for row in time_data:
        if row[0] is not None:  # 确保数据不是空值
            # 修改年份为当前年份，保持月、日、时、分、秒不变
            # 注意：这假设你想将所有日期调整为当前年份
            new_time_value = row[0].replace(year=current_year)
            processed_time_data.append(new_time_value)
        else:
            # 处理空值的情况（根据需要实现）
            print("Found None value, skipping...")

    return processed_time_data


@log_function_call
def aggregate_data_by_time(time_data: list[datetime], frequency: str) -> dict[datetime, list[str]]:
    aggregated_data = {}

    for time_point in time_data:
        # 根据给定的频率调整时间点
        if frequency == 'daily':
            aggregated_time = time_point.replace(hour=0, minute=0, second=0, microsecond=0)  # 将时间调整为当天的午夜
        elif frequency == 'weekly':
            # 将时间调整为所在周的周一的午夜
            aggregated_time = time_point - timedelta(days=time_point.weekday())
            aggregated_time = aggregated_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif frequency == 'monthly':
            # 将时间调整为所在月的第一天的午夜
            aggregated_time = time_point.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError("Invalid frequency. Please choose 'daily', 'weekly', or 'monthly'.")

        # 将数据添加到聚合字典中
        if aggregated_time not in aggregated_data:
            aggregated_data[aggregated_time] = []
        # 在列表中记录具有数据的表（这里假设每个时间点的数据是从不同的表中收集的）
        # 如果数据来源于同一张表，则可以调整为添加表名而不是列表
        aggregated_data[aggregated_time].append("Table X")  # 示例表名，你可以根据实际情况调整

    return aggregated_data


def visualize_heatmap(aggregated_data: dict[datetime, list[str]]) -> None:
    # 获取属性名列表
    attr_names = list(set(table for tables in aggregated_data.values() for table in tables))
    attr_names.sort()

    # 创建时间序列
    time_sequence = sorted(aggregated_data.keys())

    # 创建数据矩阵
    data_matrix = np.zeros((len(time_sequence), len(attr_names)))

    for i, time_point in enumerate(time_sequence):
        tables_with_data = aggregated_data[time_point]
        for table in tables_with_data:
            j = attr_names.index(table)
            data_matrix[i, j] = 1  # 数据存在时为1，否则为0

    # 绘制热图
    plt.figure(figsize=(12, 8))
    plt.imshow(data_matrix, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.xticks(np.arange(len(attr_names)), attr_names, rotation=45)
    plt.yticks(np.arange(len(time_sequence)), [time_point.strftime("%Y-%m-%d") for time_point in time_sequence])
    plt.xlabel('属性名', fontproperties='SimHei')  # 使用中文标签
    plt.ylabel('时间', fontproperties='SimHei')  # 使用中文标签
    plt.title('数据存在情况热图', fontproperties='SimHei')  # 使用中文标题
    plt.tight_layout()
    plt.show()


@log_function_call
def count_records_in_table(table_name: str) -> int:
    """查看数据库中的表有多少个数据"""
    # 建立数据库连接
    conn = pymysql.connect(**SQL_CONFIG)
    cursor = conn.cursor()

    try:
        # 构建 SQL 查询语句，统计表中的数据行数
        sql_statement = f"SELECT COUNT(*) FROM {table_name}"
        
        # 执行 SQL 查询
        cursor.execute(sql_statement)
        
        # 获取查询结果
        result = cursor.fetchone()  # fetchone() 用于获取单行结果
        if result:
            record_count = result[0]  # 查询结果是一个包含一个元素的 tuple，获取第一个元素即数据行数
            return record_count  # 返回数据行数

        else:
            return 0  # 如果结果为空，则返回 0 条数据

    except Exception as e:
        # 发生错误时回滚
        conn.rollback()
        raise e  # 将异常抛出，由调用者处理

    finally:
        # 关闭游标和数据库连接
        cursor.close()
        conn.close()


@log_function_call
def count_time_differences(tableName: list, startTime: datetime, stopTime: datetime) -> dict[float, int]:
    # 构造查询 SQL 语句
    sql_statement = f"SELECT Time FROM {tableName} WHERE Time >= %s AND Time <= %s"

    # 建立数据库连接
    conn = pymysql.connect(**SQL_CONFIG)
    cursor = conn.cursor()

    try:
        # 执行 SQL 查询
        cursor.execute(sql_statement, (startTime, stopTime))

        # 获取查询结果
        results = cursor.fetchall()

        # 计算相邻时间差并统计
        time_list = [row[0] for row in results]
        time_list.sort()  # 将时间列表排序

        # 计算相邻时间差
        time_diffs = [(time_list[i + 1] - time_list[i]).total_seconds() for i in range(len(time_list) - 1)]

        # 统计不同时间差的个数
        diff_count = {}
        for diff in time_diffs:
            diff_count[diff] = diff_count.get(diff, 0) + 1

        return diff_count

    except Exception as e:
        # 发生错误时回滚
        conn.rollback()
        print("Error executing SQL statement:", str(e))
        return None

    finally:
        # 关闭游标和数据库连接
        cursor.close()
        conn.close()


def calculate_missing_percentage(
    start_time: datetime, 
    end_time: datetime, 
    missing_intervals: list[tuple[datetime, datetime, int]]
) -> list[float]:
    # Calculate total duration in hours
    total_duration_hours = (end_time - start_time).total_seconds() / 3600

    # Initialize weekly missing time list
    weeks_count = (end_time - start_time).days // 7 + 1
    weekly_missing_hours = [0] * weeks_count

    # Accumulate missing hours per week
    for interval_start, interval_end, missing_seconds in missing_intervals:
        interval_duration_hours = missing_seconds / 3600
        for week_index in range(weeks_count):
            week_start = start_time + timedelta(days=week_index * 7)
            week_end = week_start + timedelta(days=6)
            if interval_start <= week_end and interval_end >= week_start:
                overlap_start = max(interval_start, week_start)
                overlap_end = min(interval_end, week_end)
                overlap_duration_hours = (overlap_end - overlap_start).total_seconds() / 3600
                weekly_missing_hours[week_index] += overlap_duration_hours

    # Calculate missing percentage per week
    weekly_missing_percentage = [(hours / total_duration_hours * 100) for hours in weekly_missing_hours]

    return weekly_missing_percentage

def visualize_missing_data(
    start_time: datetime, 
    end_time: datetime, 
    missing_intervals: list[tuple[datetime, datetime, int]]
) -> None:
    # Calculate weekly missing percentage
    weekly_missing_percentage = calculate_missing_percentage(start_time, end_time, missing_intervals)

    # Create plot
    weeks_count = len(weekly_missing_percentage)
    week_labels = [f'Week {i+1}' for i in range(weeks_count)]
    
    plt.figure(figsize=(12, 6))
    plt.bar(week_labels, weekly_missing_percentage, color='skyblue')
    plt.xlabel('Week')
    plt.ylabel('Missing Time Percentage (%)')
    plt.title('Missing Time Percentage per Week')
    plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def read_tuple_data_from_txt(file_path: str) -> list[tuple[datetime, datetime, int]]:
    """读取数据"""
    tuple_data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into its components
            parts = line.strip().split(',')
            # Convert string representations to datetime objects
            start_time = datetime.strptime(parts[0], "(%Y, %m, %d, %H, %M, %S, %f)")
            end_time = datetime.strptime(parts[1], "(%Y, %m, %d, %H, %M, %S, %f)")
            duration = int(parts[2])
            # Append the tuple to the list
            tuple_data.append((start_time, end_time, duration))
    return tuple_data


def calculate_weekly_missing_percentage(
    start_time: datetime, 
    end_time: datetime, 
    missing_intervals: list[tuple[datetime, datetime, int]]
) -> list[float]:
    """
    计算每周缺失百分比。

    参数：
    - start_time: datetime,起始时间
    - end_time: datetime,结束时间
    - missing_intervals: list[tuple[datetime, datetime, int]]，缺失时间段列表

    返回：
    - list[float]，每周的缺失百分比列表
    """

    # 计算总周数
    total_weeks = (end_time - start_time).days // 7 + 1

    # 初始化列表以存储每周的缺失百分比
    weekly_missing_percentage = []

    # 遍历每一周
    for week_index in range(total_weeks):
        # 定义周的起始和结束时间
        week_start = start_time + timedelta(weeks=week_index)
        week_end = min(start_time + timedelta(weeks=week_index + 1) - timedelta(microseconds=1), end_time)

        # 计算一周的总秒数
        seconds_in_week = (week_end - week_start).total_seconds()

        # 计算一周内的缺失秒数
        missing_seconds_in_week = 0
        for interval_start, interval_end, missing_seconds in missing_intervals:
            overlap_start = max(interval_start, week_start)
            overlap_end = min(interval_end, week_end)
            overlap_duration_seconds = max(0, (overlap_end - overlap_start).total_seconds())
            missing_seconds_in_week += overlap_duration_seconds

        # 计算一周的缺失百分比
        missing_percentage = (missing_seconds_in_week / seconds_in_week) * 100
        weekly_missing_percentage.append(missing_percentage)

    return weekly_missing_percentage

# 最小时间和最大时间
start_time = datetime(2023, 4, 13, 16, 4, 40, 985000)
end_time = datetime(2024, 4, 11, 13, 5, 16, 701000)

# 读取文件并加载数据到 missing_intervals
file_path = "D:/python_proj2/SQL_Met.txt"
missing_intervals = []

with open(file_path, "r") as file:
    for line in file:
        # 去除换行符并拆分字符串
        line = line.strip()
        parts = line.split(", ")

        # 解析日期时间和缺失秒数
        start_time = datetime.strptime(parts[0], "datetime(%Y, %m, %d, %H, %M, %S, %f)")
        end_time = datetime.strptime(parts[1], "datetime(%Y, %m, %d, %H, %M, %S, %f)")
        missing_seconds = int(parts[2])

        # 添加到 missing_intervals
        missing_intervals.append((start_time, end_time, missing_seconds))

# 打印结果以确认数据加载正确
print(missing_intervals)

# 计算每一周内的缺失百分比
weekly_missing_percentage = calculate_weekly_missing_percentage(start_time, end_time, missing_intervals)
print(weekly_missing_percentage)