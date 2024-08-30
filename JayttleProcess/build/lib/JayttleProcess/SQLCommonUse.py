import pymysql
import functools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Union
from tabulate import tabulate

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 记录函数调用时间
        call_time = datetime.now()
        start_time = time.time()  # 记录函数开始执行的时间

        print(f"Function '{func.__name__}' called at {call_time}")

        # 记录传入的参数类型
        args_type_str = ', '.join(map(lambda arg: f"{type(arg).__name__}", args))
        kwargs_type_str = ', '.join(f"{key}={type(value).__name__}" for key, value in kwargs.items())
        all_args_type = ', '.join(filter(None, [args_type_str, kwargs_type_str]))
        print(f"Arguments type: {all_args_type}")

        # 调用函数并记录返回值
        result = func(*args, **kwargs)
        print(f"Returned data type: {type(result).__name__}")  # 打印返回值的数据类型
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


@log_function_call
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


@log_function_call
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


@log_function_call
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


@log_function_call
def calculate_daily_data_availability(min_time: datetime, max_time: datetime, missing_intervals: list[tuple[datetime, datetime, int]]) -> dict[datetime, float]:
    """
    计算每日数据存有率字典
    
    参数：
        min_time (datetime): 数据的最小时间
        max_time (datetime): 数据的最大时间
        missing_intervals (List[Tuple[datetime, datetime, int]]): 包含缺失时间段的列表
        
    返回：
        dict[datetime, float]: 包含日期和每日数据存有率的字典
    """
    # 初始化每日数据存有率字典
    daily_data_availability = {}

    # 生成日期范围
    date_range = pd.date_range(start=min_time.date(), end=max_time.date(), freq='D')

    for date in date_range:
        # 当天的起始时间和结束时间
        start_of_day = datetime.combine(date, datetime.min.time())
        end_of_day = datetime.combine(date, datetime.max.time())

        # 初始化当天缺失数据的秒数
        missing_seconds = 0

        # 计算当天缺失数据的秒数
        for interval_start, interval_end, interval_missing_seconds in missing_intervals:
            # 如果缺失时间段与当天有交集
            if interval_start <= end_of_day and interval_end >= start_of_day:
                # 计算交集部分的缺失秒数
                intersection_start = max(interval_start, start_of_day)
                intersection_end = min(interval_end, end_of_day)
                intersection_duration = (intersection_end - intersection_start).total_seconds()
                missing_seconds += intersection_duration

        # 计算当天的数据存有率
        total_seconds_in_day = (end_of_day - start_of_day).total_seconds()
        data_availability_percentage = 100 * (1 - (missing_seconds / total_seconds_in_day))

        # 将结果添加到字典，并精确到小数点后四位
        daily_data_availability[start_of_day] = round(data_availability_percentage, 4)

    return daily_data_availability


@log_function_call
def export_data_availability_to_file(daily_data_availability: dict[datetime, dict[str, float]]) -> None:
    """
    将每日数据存有率字典输出到文件
    
    参数：
        daily_data_availability (dict[datetime, dict[str, float]]): 包含日期和每日数据存有率的字典
        
    返回：
        None
    """
    # 将字典转换为 DataFrame
    df = pd.DataFrame.from_dict(daily_data_availability, orient='index')
    
    # 将 NaN 替换为 0
    df = df.fillna(0)

    # 将 0 的数据存有率标识为 'N/A'
    df = df.apply(lambda x: x.map(lambda val: 'N/A' if val == 0 else val))

    # 重置索引，使日期成为列
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)

    # 排序日期
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 生成表格
    table = tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False)

    # 将表格输出到文件
    with open('output_table.txt', 'w', encoding='utf-8') as file:
        file.write(table)

    print("表格已经成功输出到文件 output_table.txt 中。")


def export_data_availability_to_excel(daily_data_availability: dict[datetime, dict[str, float]]) -> None:
    """
    将每日数据存有率字典输出到 Excel 文件
    
    参数：
        daily_data_availability (Dict[datetime, Dict[str, float]]): 包含日期和每日数据存有率的字典
        
    返回：
        None
    """
    # 将字典转换为 DataFrame
    df = pd.DataFrame.from_dict(daily_data_availability, orient='index')
    
    # 将 NaN 替换为 0
    df = df.fillna(0)

    # 将 0 的数据存有率标识为 'N/A'
    df = df.applymap(lambda val: 'N/A' if val == 0 else val)

    # 重置索引，使日期成为列
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)

    # 排序日期
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 生成 Excel 文件
    excel_filename = 'output_table.xlsx'
    df.to_excel(excel_filename, index=False)

    print(f"表格已经成功输出到 Excel 文件 {excel_filename} 中。")


@log_function_call
def load_missing_intervals(file_path: str) -> list[tuple[datetime, datetime, int]]:
    """
    从文件加载数据到 missing_intervals 列表中
    
    参数：
        file_path (str): 文件路径
        
    返回：
        List[Tuple[datetime, datetime, int]]: 包含缺失时间段的列表
    """
    missing_intervals = []

    with open(file_path, "r") as file:
        for line in file:
            # 替换字符串中的 "datetime.datetime" 为 "datetime"
            line = line.replace("datetime.datetime", "datetime")

            # 使用 eval() 函数将字符串转换为元组
            parts = eval(line)

            # 解析日期时间和缺失秒数
            start_time = parts[0]
            end_time = parts[1]
            missing_seconds = parts[2]

            # 添加到 missing_intervals
            missing_intervals.append((start_time, end_time, missing_seconds))

    return missing_intervals


@log_function_call
def main() -> None:
    file_path_met = "D:/python_proj2/SQL_Met.txt"
    file_path_accelerometer = "D:/python_proj2/SQL_accelerometer.txt"
    file_path_tiltmeter = "D:/python_proj2/SQL_tiltmeter.txt"
    file_path_arr = "D:/python_proj2/SQL_arr.txt"
    file_path_ggkx = "D:/python_proj2/SQL_arr.txt"


    missing_intervals = load_missing_intervals(file_path_met)
    missing_intervals_accelerometer = load_missing_intervals(file_path_accelerometer)
    missing_intervals_tiltmeter = load_missing_intervals(file_path_tiltmeter)
    missing_intervals_arr = load_missing_intervals(file_path_arr)
    missing_intervals_ggkx = load_missing_intervals(file_path_ggkx)

    # 最小时间和最大时间
    min_time = datetime(2023, 4, 13, 16, 4, 40, 985000)
    max_time = datetime(2024, 4, 11, 13, 5, 16, 701000)


    accelerometer_min_time = datetime(2023, 7, 17, 16, 40, 39, 2000)
    accelerometer_max_time = datetime(2023, 8, 16, 19, 49, 8, 583000)

    tiltmeter_min_time = datetime(2023, 4, 17, 23, 49, 5, 782000)
    tiltmeter_max_time = datetime(2024, 4, 12, 21, 1, 12, 430000)

    arr_min_time = datetime(2023, 3, 23, 15, 15, 43)
    arr_max_time = datetime(2024, 4, 13, 18, 36, 40)

    ggkx_min_time = datetime(2023, 3, 23, 15, 15, 43)
    ggkx_max_time = datetime(2024, 4, 13, 18, 39, 43)

    # 示例用法
    daily_availability_met = calculate_daily_data_availability(min_time, max_time, missing_intervals)

    # 计算SQL_accelerometer和SQL_tiltmeter的每日数据存有率
    daily_availability_accelerometer = calculate_daily_data_availability(accelerometer_min_time, accelerometer_max_time, missing_intervals_accelerometer)
    daily_availability_tiltmeter = calculate_daily_data_availability(tiltmeter_min_time, tiltmeter_max_time, missing_intervals_tiltmeter)
    daily_availability_arr = calculate_daily_data_availability(arr_min_time, arr_max_time, missing_intervals_arr)
    daily_availability_ggkx = calculate_daily_data_availability(ggkx_min_time, ggkx_max_time, missing_intervals_ggkx)


    # 找出最小和最大日期
    all_dates: list[datetime] = []
    all_dates.extend(daily_availability_met.keys())
    all_dates.extend(daily_availability_accelerometer.keys())
    all_dates.extend(daily_availability_tiltmeter.keys())
    all_dates.extend(daily_availability_arr.keys())
    all_dates.extend(daily_availability_ggkx.keys())

    min_date = min(all_dates)
    max_date = max(all_dates)

    # 创建新字典并填充
    combined_dict:dict[datetime, dict[str, float]] = {}  # 类型注解
    for date in (min_date + timedelta(days=i) for i in range((max_date - min_date).days + 1)):
        combined_dict[date] = {
            'daily_arr': daily_availability_arr.get(date, 0),
            'daily_ggkx': daily_availability_ggkx.get(date, 0),
            'daily_met': daily_availability_met.get(date, 0),    
            'daily_tiltmeter': daily_availability_tiltmeter.get(date, 0),
            'daily_accelerometer': daily_availability_accelerometer.get(date, 0),
        }

    # export_data_availability_to_file(combined_dict)
    export_data_availability_to_excel(combined_dict)


def parse_tabulated_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = [line.rstrip() for line in file if line.strip()]

    # Determine the indices for header and data sections
    header_index = next((i for i, line in enumerate(lines) if '╞' in line), None)
    data_start_index = header_index + 1 if header_index is not None else None

    if header_index is None or data_start_index is None:
        return []  # No valid data section found

    # Parse headers
    headers = lines[header_index].split('│')[1:-1]
    headers = [h.strip() for h in headers]

    # Parse data
    data = []
    for line in lines[data_start_index:]:
        if '├' in line or '┤' in line:
            continue  # Skip frame lines
        parts = line.split('│')[1:-1]
        parts = [part.strip() for part in parts]
        if len(parts) != len(headers):
            continue  # Skip lines that do not match header count
        row_dict = dict(zip(headers, parts))
        data.append(row_dict)

    return data


@log_function_call
def query_high_frequency_periods(listName: str, StartTime: datetime, EndTime: datetime) -> list:
    # Corrected the date formatting to be compatible with Python's datetime formatting
    sql_statement = f"""
    WITH TimeGroup AS (
        SELECT
            DATE_FORMAT(time, '%Y-%m-%d %H:%i:%S') AS RoundedTime,
            COUNT(*) AS CountPerSecond
        FROM
            {listName}
        WHERE
            time >= '{StartTime.strftime('%Y-%m-%d %H:%M:%S')}' AND
            time <= '{EndTime.strftime('%Y-%m-%d %H:%M:%S')}'
        GROUP BY
            RoundedTime
    )
    SELECT
        RoundedTime,
        CountPerSecond
    FROM
        TimeGroup
    WHERE
        CountPerSecond >= 5
    ORDER BY
        RoundedTime ASC;
    """

    # 执行 SQL 查询
    conn = pymysql.connect(**SQL_CONFIG)  # 确保 SQL_CONFIG 已定义
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



def execute_sql_and_save_to_txt(sql_statement: str, file_path: str):
    # 建立数据库连接
    conn = pymysql.connect(**SQL_CONFIG)
    cursor = conn.cursor()
    
    try:
        # 执行输入的 SQL 语句
        cursor.execute(sql_statement)
        
        # 如果是查询语句，则将查询结果写入到 txt 文件中
        if sql_statement.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            with open(file_path, 'w') as f:
                for row in results:
                    f.write(','.join(map(str, row)) + '\n')
            return "Query executed successfully. Results saved to " + file_path
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


def Proj1_high_frequency_data():
    # 执行 SQL 查询并将结果保存到指定的 txt 文件中
    sql_statement = """
    SELECT * FROM arr
    WHERE time >= '2024-07-01 00:00:00' AND time <= '2024-07-02 00:00:00';
    """
    file_path = "SQLquery_results.txt"
    result = execute_sql_and_save_to_txt(sql_statement, file_path)
    
