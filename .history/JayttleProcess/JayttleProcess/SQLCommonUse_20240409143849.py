import pymysql
import functools
import time
from datetime import datetime, timedelta

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

# 使用装饰器
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

# 使用装饰器
@log_function_call
def some_function(x):
    # 模拟一个耗时操作
    time.sleep(x)
    return x

# 测试装饰器
result = some_function(2)

# # 测试装饰器
# print(expensive_function(4, 5))
# print(expensive_function(4, 5))  # 第二次调用将直接使用缓存结果

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

# 使用装饰器
@catch_exceptions(default_value="Error occurred")
def risky_function(x):
    # 这里是一些可能抛出异常的代码
    return 1 / x


# 全局变量存储数据库连接信息
SQL_CONFIG = {
    "host": "localhost",
    "user": "Jayttle",
    "password": "@JayttleRoot",
    "database": "jayttle"
}
# # 全局变量存储数据库连接信息
# SQL_CONFIG = {
#     "host": "47.98.201.213",
#     "user": "root",
#     "password": "TJ1qazXSW@",
#     "database": "tianmeng_cableway"
# }
def execute_sql(sql_statement: str) -> None:
    # 建立数据库连接
    conn = pymysql.connect(**SQL_CONFIG)
    cursor = conn.cursor()
    
    try:
        # 执行输入的 SQL 语句
        cursor.execute(sql_statement)
        
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


def create_table() -> None:
    """创建表的 SQL 语句 """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS met (
        Time DATETIME NOT NULL,
        StationID INT NOT NULL,
        Temperature FLOAT,
        Humidness FLOAT,
        Pressure FLOAT,
        WindSpeed FLOAT,
        WindDirection VARCHAR(20),
        PRIMARY KEY (Time, StationID)
    )
    """
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


@log_function_call
def insert_data(*args, **kwargs):
    if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
        station_id, name = args
        insert_query = """
        INSERT INTO met 
        (Time, StationID, Temperature, Humidness, Pressure, WindSpeed, WindDirection)
        VALUES 
        (%s, %s, 25.5, 50.0, 1013.25, 10.0, 'N')
        """
        values = (args[0], args[1])  # 使用 %s 占位符，并传入参数值的元组
        execute_sql_params(insert_query, values)
    else:
        default_insert_data_query = """
        INSERT INTO met (Time, StationID, Temperature, Humidness, Pressure, WindSpeed, WindDirection)
        VALUES ('2024-04-08 12:00:00', 7, 25.5, 50.0, 1013.25, 10.0, 'N')
        """
        execute_sql(default_insert_data_query)


def insert_data_id() -> None:
    """插入数据的 SQL 语句 """
    insert_data_query = """
    INSERT INTO met_id (StationID, Name)
    VALUES
    (7, 'M07')
    """
    execute_sql(insert_data_query)  


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


