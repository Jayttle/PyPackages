import mysql.connector

# 全局变量存储数据库连接信息
SQL_CONFIG = {
    "host": "47.98.201.213",
    "user": "root",
    "password": "TJ1qazXSW@",
    "database": "tianmeng_cableway"
}

def query_database():
    # 连接数据库
    conn = mysql.connector.connect(
        host=SQL_CONFIG["host"],
        user=SQL_CONFIG["user"],
        password=SQL_CONFIG["password"],
        database=SQL_CONFIG["database"]
    )

    # 创建游标对象
    cursor = conn.cursor()

    # 执行SQL查询语句
    cursor.execute("SELECT * FROM table_name")

    # 获取查询结果
    result = cursor.fetchall()

    # 处理结果
    for row in result:
        print(row)

    # 关闭游标和连接
    cursor.close()
    conn.close()

# 调用函数执行数据库查询
query_database()