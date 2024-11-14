import mysql.connector

# 连接数据库
conn = mysql.connector.connect(
    host="localhost",
    user="username",
    password="password",
    database="database_name"
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