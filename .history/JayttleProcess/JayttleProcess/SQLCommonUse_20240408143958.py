import pymysql

# 全局变量存储数据库连接信息
SQL_CONFIG = {
    "host": "localhost",
    "user": "Jayttle",
    "password": "123456",
    "database": " "
}
# # 全局变量存储数据库连接信息
# SQL_CONFIG = {
#     "host": "47.98.201.213",
#     "user": "root",
#     "password": "TJ1qazXSW@",
#     "database": "tianmeng_cableway"
# }

def create_database(database_name: str) -> None:
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
        # 关闭游标和数据库连接
        cursor.close()
        conn.close()


# 调用函数创建数据库
create_database()


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

