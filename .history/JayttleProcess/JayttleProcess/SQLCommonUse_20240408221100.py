import pymysql

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


def insert_data() -> None:
    """插入数据的 SQL 语句 """
    insert_data_query = """
    INSERT INTO met (Time, StationID, Temperature, Humidness, Pressure, WindSpeed, WindDirection)
    VALUES
    ('2024-04-08 12:00:00', 1, 25.5, 50.0, 1013.25, 10.0, 'N')
    """
    execute_sql(insert_data_query)    


def insert_data_id() -> None:
    """插入数据的 SQL 语句 """
    insert_data_query = """
    INSERT INTO met_id (StationID, Name)
    VALUES
    ('7', 'M07')
    """
    execute_sql(insert_data_query)   