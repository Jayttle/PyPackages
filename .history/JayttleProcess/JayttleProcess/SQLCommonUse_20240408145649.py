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


# 调用函数查询数据库列表
def print_sql_config() -> None:
    """调用函数查询数据库列表"""
    print("SQL_CONFIG = {")
    print("\t\"host\": \"localhost\",")
    print("\t\"user\": \"Jayttle\",")
    print("\t\"password\": \"@JayttleRoot\",")
    print("\t\"database\": \"jayttle\"")
    print("}")


change_database()
print_sql_config()
def insert_data(name, age):
    try:
        # 连接数据库
        conn = pymysql.connect(
            host=SQL_CONFIG["host"],
            user=SQL_CONFIG["user"],
            password=SQL_CONFIG["password"],
            database=SQL_CONFIG["database"]
        )

        # 创建游标对象
        cursor = conn.cursor()

        # 插入数据
        sql = "INSERT INTO table_name (name, age) VALUES (%s, %s)"
        val = (name, age)
        cursor.execute(sql, val)

        # 提交事务
        conn.commit()

        print("Data inserted successfully")

    except Exception as e:
        print("Error:", e)

    finally:
        # 关闭游标和连接
        cursor.close()
        conn.close()

def delete_data(id):
    try:
        # 连接数据库
        conn = pymysql.connect(
            host=SQL_CONFIG["host"],
            user=SQL_CONFIG["user"],
            password=SQL_CONFIG["password"],
            database=SQL_CONFIG["database"]
        )

        # 创建游标对象
        cursor = conn.cursor()

        # 删除数据
        sql = "DELETE FROM table_name WHERE id = %s"
        val = (id,)
        cursor.execute(sql, val)

        # 提交事务
        conn.commit()

        print("Data deleted successfully")

    except Exception as e:
        print("Error:", e)

    finally:
        # 关闭游标和连接
        cursor.close()
        conn.close()

def select_data():
    try:
        # 连接数据库
        conn = pymysql.connect(
            host=SQL_CONFIG["host"],
            user=SQL_CONFIG["user"],
            password=SQL_CONFIG["password"],
            database=SQL_CONFIG["database"]
        )

        # 创建游标对象
        cursor = conn.cursor()

        # 查询数据
        sql = "SELECT * FROM table_name"
        cursor.execute(sql)

        # 获取查询结果
        result = cursor.fetchall()

        # 打印查询结果
        for row in result:
            print(row)

    except Exception as e:
        print("Error:", e)

    finally:
        # 关闭游标和连接
        cursor.close()
        conn.close()

def update_data(id, new_age):
    try:
        # 连接数据库
        conn = pymysql.connect(
            host=SQL_CONFIG["host"],
            user=SQL_CONFIG["user"],
            password=SQL_CONFIG["password"],
            database=SQL_CONFIG["database"]
        )

        # 创建游标对象
        cursor = conn.cursor()

        # 更新数据
        sql = "UPDATE table_name SET age = %s WHERE id = %s"
        val = (new_age, id)
        cursor.execute(sql, val)

        # 提交事务
        conn.commit()

        print("Data updated successfully")

    except Exception as e:
        print("Error:", e)

    finally:
        # 关闭游标和连接
        cursor.close()
        conn.close()


