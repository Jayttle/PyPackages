import smtplib
import os
import io
import json
import shutil
import numpy as np
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta
import poplib
from typing import Union, Optional
from email import policy
from email.parser import BytesParser
from email.header import decode_header, make_header
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union, Set, Dict

class EmailUseType:
    def __init__(self, config_file_path: str = None) -> None:
        self.sender: Optional[str] = None
        self.user: Optional[str] = None
        self.passwd: Optional[str] = None
        self.receiver: Optional[str] = None
        if config_file_path is not None:
            self._load_emailInfo(config_file_path) 

    def _load_emailInfo(self, file_path: str):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            # 提取 QQ_email_config 部分
            qq_email_config = data.get('QQ_email_config', {})

            # 提取 username、passwd 和 receiver
            self.sender = qq_email_config.get('username', self.sender)
            self.user = qq_email_config.get('username', self.user)
            self.passwd = qq_email_config.get('passwd', self.passwd)
            self.receiver = qq_email_config.get('receiver', self.receiver)
            
            if not all([self.sender, self.user, self.passwd, self.receiver]):
                print("警告:email 配置中的某些信息缺失。")
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")
        except json.JSONDecodeError:
            print("文件内容不是有效的 JSON 格式。")
        except Exception as e:
            print(f"发生错误：{e}")

    def _input_emailInfo(self, sender: str, passwd: str, receiver: str):
        self.sender = self.user = sender
        self.passwd = passwd
        self.receiver = receiver

    def _check_emailInfo(self) -> None:
        print('----- Email Info -----')
        print(f"Sender: {self.sender}")
        print(f"User: {self.user}")
        print(f"Password: {self.passwd}")
        print(f"Receiver: {self.receiver}")


    def _send_QQ_email_plain(self, email_title: str = 'Null', email_content: str = ''):
        "发送纯文本的qq邮件"
        sender = user = self.sender    # 发送方的邮箱账号
        passwd = self.passwd            # 授权码
        receiver = self.receiver        # 接收方的邮箱账号，不一定是QQ邮箱

        # 纯文本内容 
        msg = MIMEText(email_content, 'plain', 'utf-8')
        # From 的内容是有要求的，前面的abc为自己定义的 nickname，如果是ASCII格式，则可以直接写
        msg['From'] = user
        msg['To'] = receiver
        msg['Subject'] = email_title         # 点开详情后的标题


        try:
            # 建立 SMTP 、SSL 的连接，连接发送方的邮箱服务器
            smtp = smtplib.SMTP_SSL('smtp.qq.com', 465)
    
            # 登录发送方的邮箱账号
            smtp.login(user, passwd)
    
            # 发送邮件 发送方，接收方，发送的内容
            smtp.sendmail(sender, receiver, msg.as_string())
    
            print('邮件发送成功')
    
            smtp.quit()
        except Exception as e:
            print(e)
            print('发送邮件失败')

    def _send_QQ_email_mul(self, email_title: str = 'Null', email_content: str = '', annex_path: str = ''):
        sender = user = self.sender    # 发送方的邮箱账号
        passwd = self.passwd            # 授权码
        receiver = self.receiver        # 接收方的邮箱账号，不一定是QQ邮箱
    
        content = MIMEMultipart()           # 创建一个包含多个部分的内容
        content['From'] = user
        content['To'] = receiver
        content['Subject'] = email_title
    
        # 添加文本内容
        text = MIMEText(email_content, 'plain', 'utf-8')
        content.attach(text)

        # 如果 annex_path 是文件夹，则压缩文件夹
        if os.path.isdir(annex_path):
            zip_path = f'{annex_path}.zip'
            shutil.make_archive(annex_path, 'zip', annex_path)
            annex_path = zip_path

        # 添加附件
        with open(annex_path, 'rb') as f:
            attachment = MIMEApplication(f.read())    # 读取为附件
            attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(annex_path))
            content.attach(attachment)
    
        try:
            smtp = smtplib.SMTP_SSL('smtp.qq.com', 465)
            smtp.login(user, passwd)
            smtp.sendmail(sender, receiver, content.as_string())
            print('邮件发送成功')
        except Exception as e:
            print(e)
            print('发送邮件失败')

    def _sanitize_folder_name(self, name):
        return ''.join(c for c in name if c.isalnum() or c in ('', '_', '-')).rstrip()

    def _get_payload(self, email):
        if email.is_multipart():
            for part in email.iter_parts():
                content_type = part.get_content_type()
                content_disposition = part.get("Content-Disposition", "")
                
                if content_type == 'text/plain':
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        return payload.decode(charset), 'text', []
                    except (UnicodeDecodeError, TypeError):
                        return payload.decode('utf-8', errors='ignore'), 'text', []
                elif content_type == 'text/html':
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        return payload.decode(charset), 'html', []
                    except (UnicodeDecodeError, TypeError):
                        return payload.decode('utf-8', errors='ignore'), 'html', []
                elif 'attachment' in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        extension = filename.split('.')[-1]
                        return part.get_payload(decode=True), 'image', [extension]
        else:
            content_type = email.get_content_type()
            if content_type == 'text/plain':
                payload = email.get_payload(decode=True)
                charset = email.get_content_charset() or 'utf-8'
                try:
                    return payload.decode(charset), 'text', []
                except (UnicodeDecodeError, TypeError):
                    return payload.decode('utf-8', errors='ignore'), 'text', []
            elif content_type == 'text/html':
                payload = email.get_payload(decode=True)
                charset = email.get_content_charset() or 'utf-8'
                try:
                    return payload.decode(charset), 'html', []
                except (UnicodeDecodeError, TypeError):
                    return payload.decode('utf-8', errors='ignore'), 'html', []
        
        return '', 'unknown', []
    
    def _get_email_and_save(self):
        sender = user = self.sender    # 发送方的邮箱账号
        passwd = self.passwd            # 授权码
        # 连接到POP3服务器
        pop_server = poplib.POP3_SSL('pop.qq.com', 995)
        pop_server.user(sender)
        pop_server.pass_(passwd)
        
        # 获取邮箱中的邮件信息
        num_emails = len(pop_server.list()[1])
        
        # 计算7天前的日期
        seven_days_ago = datetime.now() - timedelta(days=7)
            
        # 确保 seven_days_ago 是 offset-naive
        seven_days_ago = seven_days_ago.replace(tzinfo=None)

        # 遍历每封邮件
        for i in range(num_emails):
            # 获取邮件内容
            response, lines, octets = pop_server.retr(i + 1)
            email_content = b'\r\n'.join(lines)
        
            # 解析邮件内容
            email_parser = BytesParser(policy=policy.default)
            email = email_parser.parsebytes(email_content)
        
            # 解析邮件头部信息
            email_from = email.get('From').strip()
            email_from = str(make_header(decode_header(email_from)))

            try:        
                date_str = email.get('Date')
                # 解析邮件日期
                email_date = parsedate_to_datetime(date_str)
                # 确保 email_date 是 offset-naive
                email_date = email_date.replace(tzinfo=None)
            except (TypeError, ValueError):
                continue  # 如果邮件没有日期，跳过

            # 只处理最近7天内的邮件
            if email_date < seven_days_ago:
                continue

            if email_from:
                subject = email.get('Subject').strip()
                decoded_subject = str(make_header(decode_header(subject))) if subject else 'No Subject'

                email_body, body_type, *extras = self.get_payload(email)
                
                safe_folder_name = self.sanitize_folder_name(email_from)
                safe_subject = self.sanitize_folder_name(decoded_subject)

                print("------------------")
                print("From:", email_from)
                print("Subject:", decoded_subject)
                print("Body Type:", body_type)

                if body_type == 'text':
                    print("Body:", email_body)
                elif body_type == 'html':
                    directory = safe_folder_name
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    with open(f'{directory}/{safe_subject}.html', 'w') as f:
                        f.write(email_body)
                elif body_type == 'image':
                    directory = safe_folder_name
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    image_data = email_body
                    image_extension = extras[0] if extras else 'png'
                    with open(f'{directory}/{safe_subject}.{image_extension[0]}', 'wb') as f:
                        f.write(image_data)
            else:
                continue  # 跳过缺失发件人的邮件
        # 关闭连接
        pop_server.quit()

     
class SQLUseType:
    def __init__(self, SQL_config_path: str = None) -> None:
        self.SQL_CONFIG: dict = {}
        if SQL_config_path is not None:
            self.load_SQLConfig(SQL_config_path)
        
    def load_SQLConfig(self, SQL_config_path: str) -> None:
        try:
            with open(SQL_config_path, 'r') as file:
                data = json.load(file)
                
            SQL_config = data.get('SQL_tianmeng_config', {})
            self.SQL_CONFIG['host'] = SQL_config.get('host')
            self.SQL_CONFIG['user']  = SQL_config.get('user')
            self.SQL_CONFIG['password']  = SQL_config.get('password')
            self.SQL_CONFIG['database'] = SQL_config.get('database')

            if not all([self.SQL_CONFIG.get('host'), self.SQL_CONFIG.get('user'), self.SQL_CONFIG.get('password'), self.SQL_CONFIG.get('database')]):
                print("警告:SQL 配置中的某些信息缺失。")
        
        except FileNotFoundError:
            print(f"文件 {SQL_config_path} 未找到。")
        except json.JSONDecodeError:
            print("文件内容不是有效的 JSON 格式。")
        except Exception as e:
            print(f"发生错误：{e}")

    def input_emailInfo(self, host: str, user: str, password: str, database: str):
        self.SQL_CONFIG['host'] = host
        self.SQL_CONFIG['user']  = user
        self.SQL_CONFIG['password']  = password
        self.SQL_CONFIG['database'] = database


    def check_SQLInfo(self) -> None:
        print('-----check------')
        print(f"Host: {self.SQL_CONFIG.get('host')}")
        print(f"User: {self.SQL_CONFIG.get('user')}")
        print(f"Password: {self.SQL_CONFIG.get('password')}")
        print(f"Database: {self.SQL_CONFIG.get('database')}")


    def execute_sql(self, sql_statement: str) -> Union[str, list[tuple]]:
        # 建立数据库连接
        conn = pymysql.connect(**self.SQL_CONFIG)
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

    def execute_sql_and_save_to_txt(self, sql_statement: str, file_path: str):
        # 建立数据库连接
        conn = pymysql.connect(**self.SQL_CONFIG)
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

    
    def get_min_max_time(self, listName: str) -> tuple:
        # 查询 Time 列的最小值和最大值
        query = "SELECT MIN(Time) AS min_time, MAX(Time) AS max_time FROM {0};".format(listName)
        conn = pymysql.connect(**self.SQL_CONFIG)
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

    def query_time_difference(self, listName: str, StartTime: datetime, EndTime: datetime) -> tuple:
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
        conn = pymysql.connect(**self.SQL_CONFIG)
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
    
    def count_records_in_table(self, table_name: str) -> int:
        """查看数据库中的表有多少个数据"""
        # 建立数据库连接
        conn = pymysql.connect(**self.SQL_CONFIG)
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

    def count_time_differences(self, tableName: list, startTime: datetime, stopTime: datetime) -> dict[float, int]:
        # 构造查询 SQL 语句
        sql_statement = f"SELECT Time FROM {tableName} WHERE Time >= %s AND Time <= %s"

        # 建立数据库连接
        conn = pymysql.connect(**self.SQL_CONFIG)
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

class Avr:
    def __init__(self, time: str, station_id: int, fix_mode: int, yaw: float, tilt: float, range_val: float, pdop: float, sat_num: int):
        try:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
        except ValueError:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
        self.station_id = station_id
        self.fix_mode = fix_mode
        self.yaw = yaw
        self.tilt = tilt
        self.range_val = range_val
        self.pdop = pdop
        self.sat_num = sat_num
    
    @classmethod
    def _from_string(cls, line: str) -> 'Avr':
        parts = line.strip().split(',')
        time = parts[0]
        station_id = int(parts[1])
        fix_mode = int(parts[2])
        yaw = float(parts[3])
        tilt = float(parts[4])
        range_val = float(parts[5])
        pdop = float(parts[6])
        sat_num = int(parts[7])
        return cls(time, station_id, fix_mode, yaw, tilt, range_val, pdop, sat_num)

    @classmethod
    def _from_file(cls, file_path: str) -> pd.DataFrame:
        avr_data: list[Avr] = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                if line.strip():  # Ensure the line is not empty
                    avr_instance = cls._from_string(line)
                    avr_data.append(avr_instance)
        # Convert to DataFrame
        data = [{
            'Time': a.time,
            'StationID': a.station_id,
            'Fix Mode': a.fix_mode,
            'Yaw': a.yaw,
            'Tilt': a.tilt,
            'Range': a.range_val,
            'PDOP': a.pdop,
            'Sat Num': a.sat_num
        } for a in avr_data]
        return pd.DataFrame(data) if avr_data else pd.DataFrame()

    @classmethod
    def _split_avr_columns(cls, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 选择需要的列并创建新的 DataFrame
        df_yaw = df[['Time', 'Yaw']].copy()
        df_tilt = df[['Time', 'Tilt']].copy()
        df_range = df[['Time', 'Range']].copy()
        
        # 返回字典，其中包含三个 DataFrame
        return {
            'Yaw': df_yaw,
            'Tilt': df_tilt,
            'Range': df_range
        }
    
    
    @classmethod
    def _split_station_data(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Split the DataFrame based on StationID values 3 and 8
        df_station_3 = df[df['StationID'] == 3]
        df_station_8 = df[df['StationID'] == 8]
        return df_station_3, df_station_8

    
class Ggkx:
    def __init__(self, time: str, station_id: int, receiver_id: int, lat: float, lon: float, geo_height: float,
                 fix_mode: int, sate_num: int, pdop: float, sigma_e: float, sigma_n: float, sigma_u: float,
                 prop_age: int):
        try:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
        except ValueError:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
        self.station_id = station_id
        self.receiver_id = receiver_id
        self.lat = lat
        self.lon = lon
        self.geo_height = geo_height
        self.fix_mode = fix_mode
        self.sate_num = sate_num
        self.pdop = pdop
        self.sigma_e = sigma_e
        self.sigma_n = sigma_n
        self.sigma_u = sigma_u
        self.prop_age = prop_age

    @classmethod
    def _from_file(cls, file_path: str) -> pd.DataFrame:
        ggkx_data_list: List['Ggkx'] = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split(',')
                if len(data) == 13:
                    # Convert appropriate data types
                    time = data[0]
                    station_id = int(data[1])
                    receiver_id = int(data[2])
                    lat = float(data[3])
                    lon = float(data[4])
                    geo_height = float(data[5])
                    fix_mode = int(data[6])
                    sate_num = int(data[7])
                    pdop = float(data[8])
                    sigma_e = float(data[9])
                    sigma_n = float(data[10])
                    sigma_u = float(data[11])
                    prop_age = int(data[12])
                    
                    if fix_mode == 3:
                        ggkx_data_list.append(cls(time, station_id, receiver_id, lat, lon, geo_height,
                                                  fix_mode, sate_num, pdop, sigma_e, sigma_n, sigma_u, prop_age))
        # Convert to DataFrame
        data = [{
            'Time': g.time,
            'StationID': g.station_id,
            'Receiver ID': g.receiver_id,
            'Lat': g.lat,
            'Lon': g.lon,
            'Geo Height': g.geo_height,
            'Fix Mode': g.fix_mode,
            'Sate Num': g.sate_num,
            'PDOP': g.pdop,
            'Sigma E': g.sigma_e,
            'Sigma N': g.sigma_n,
            'Sigma U': g.sigma_u,
            'Prop Age': g.prop_age
        } for g in ggkx_data_list]
        return pd.DataFrame(data) if ggkx_data_list else pd.DataFrame()

    @classmethod
    def _split_ggkx_columns(cls, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 选择需要的列并创建新的 DataFrame
        df_lat = df[['Time', 'Lat']].copy()
        df_lon = df[['Time', 'Lon']].copy()
        df_geoheight = df[['Time', 'GeoHeight']].copy()
        
        # 返回字典，其中包含三个 DataFrame
        return {
            'Lat': df_lat,
            'Lon': df_lon,
            'GeoHeight': df_geoheight
        }
    
class Met:
    def __init__(self, time: str, station_id: int, temperature: float, humidness: float, pressure: float, windSpeed: float, windDirection: float) -> None:
        try:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
        except ValueError:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
        self.station_id = station_id
        self.temperature = temperature
        self.humidness = humidness
        self.pressure = pressure
        self.windSpeed = windSpeed
        self.windDirection = windDirection

    @classmethod
    def _from_file(cls, file_path: str) -> pd.DataFrame:
        met_data_list: List['Met'] = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split(',')
                if len(data) == 7:
                    # Convert appropriate data types
                    time = data[0]
                    station_id = int(data[1])
                    temperature = float(data[2])
                    humidness = float(data[3])
                    pressure = float(data[4])
                    windSpeed = float(data[5])
                    windDirection = float(data[6])
                    
                    met_data_list.append(cls(time, station_id, temperature, humidness, pressure, windSpeed, windDirection))
        # Convert to DataFrame
        data = [{
            'Time': t.time,
            'StationID': t.station_id,
            'Temperature': t.temperature,
            'Humidness': t.humidness,
            'Pressure': t.pressure,
            'WindSpeed': t.windSpeed,
            'WindDirection': t.windDirection
        } for t in met_data_list]
        return pd.DataFrame(data) if met_data_list else pd.DataFrame()
    
    @classmethod
    def _split_met_columns(cls, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 选择需要的列并创建新的 DataFrame
        df_Temperature = df[['Time', 'Temperature']].copy()
        df_Humidness = df[['Time', 'Humidness']].copy()
        df_Pressure = df[['Time', 'Pressure']].copy()
        df_WindSpeed = df[['Time', 'WindSpeed']].copy()
        df_WindDirection = df[['Time', 'WindDirection']].copy()
        
        # 返回字典，其中包含五个 DataFrame
        return {
            'Temperature': df_Temperature,
            'Humidness': df_Humidness,
            'Pressure': df_Pressure,
            'WindSpeed': df_WindSpeed,
            'WindDirection': df_WindDirection
        }
    

class TiltmeterData:
    def __init__(self, time: str, station_id: int, pitch: float, roll: float):
        try:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
        except ValueError:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
        self.station_id = station_id
        self.pitch = pitch
        self.roll = roll
    
    @classmethod
    def _from_file(cls, file_path: str) -> pd.DataFrame:
        tiltmeter_data_list: List['TiltmeterData'] = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split(',')
                if len(data) == 4:
                    # Convert appropriate data types
                    time = data[0]
                    station_id = int(data[1])
                    pitch = float(data[2])
                    roll = float(data[3])
                    
                    tiltmeter_data_list.append(cls(time, station_id, pitch, roll))
        # Convert to DataFrame
        data = [{
            'Time': t.time,
            'StationID': t.station_id,
            'Pitch': t.pitch,
            'Roll': t.roll
        } for t in tiltmeter_data_list]
        return pd.DataFrame(data) if tiltmeter_data_list else pd.DataFrame()

    @classmethod
    def _split_tiltmeter_columns(cls, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 选择需要的列并创建新的 DataFrame
        df_pitch = df[['Time', 'Pitch']].copy()
        df_roll = df[['Time', 'Roll']].copy()
        
        # 返回字典，其中包含三个 DataFrame
        return {
            'Pitch': df_pitch,
            'Roll': df_roll,
        }

class pdDataFrameProcess:
    @classmethod
    def _pdData_clean(cls, data: pd.DataFrame) -> pd.DataFrame:
        # 计算每列的缺失值数量
        missing_values = data.isnull().sum()
        print("Missing values per column:\n", missing_values)

        # 计算并删除重复行
        duplicate_count = data.duplicated().sum()
        print(f"Duplicate rows count: {duplicate_count}")
        data = data.drop_duplicates()
        
        return data

    @classmethod
    def _clean_time_data(cls, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        # 处理时间数据
        if time_column not in data.columns:
            raise ValueError(f"Column '{time_column}' not found in DataFrame.")

        # 将时间列转换为 datetime 类型，处理格式错误
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')

        # 打印转换后的时间列
        print(f"After conversion, unique time values:\n{data[time_column].unique()}")

        # 检查转换后的缺失值
        missing_time_values = data[time_column].isnull().sum()
        print(f"Missing time values count: {missing_time_values}")

        # 删除时间列中无效的时间数据
        data = data.dropna(subset=[time_column])
        
        return data


    @classmethod
    def _save_pdDataFrame(self, df: pd.DataFrame, save_file_path: str) -> None:
        df.to_csv(save_file_path, sep='\t', index=False)

        
    @classmethod
    def _create_summary_dataframe(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        summary = []
        for name, df in datasets.items():
            if df.empty:
                # 处理空的数据框
                row_count = 0
                col_count = 0
                missing_values = 0
                duplicate_count = 0
            else:
                # 计算缺失值总数
                missing_values = df.isnull().sum().sum()
                # 计算重复行数
                duplicate_count = df.duplicated().sum()
                # 统计行数和列数
                row_count = df.shape[0]
                col_count = df.shape[1]

            summary_entry = {
                'Dataset': name,
                'Rows': row_count,
                'Columns': col_count,
                'Missing Values': missing_values,
                'Duplicate Rows': duplicate_count,
            }
            summary.append(summary_entry)
        
        return pd.DataFrame(summary)

    # 将 DataFrame 的打印输出捕获到一个字符串中
    @classmethod
    def _capture_dataframe_output(cls, df: pd.DataFrame) -> str:
        output = io.StringIO()
        df.to_string(buf=output)
        return output.getvalue()
    
class TimeDataFrameMethod:
    @classmethod
    def _save_pdDataFrame(self, df: pd.DataFrame, save_file_path: str) -> None:
        df.to_csv(save_file_path, sep='\t', index=False)

    @classmethod
    def _plot_timeDF(cls, df: pd.DataFrame, colum_key: str, title: str, xlabel: str, ylabel: str, save_path: str) -> None:
        # 时间序列分析
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time')

        plt.figure(figsize=(9, 6))
        ax = plt.gca()

        # 设置日期格式化器和日期刻度定位器
        date_fmt = mdates.DateFormatter("%H:%M")  # 显示小时:分钟
        date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

        # 隐藏右边框和上边框
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # 绘制折线图，根据时间间隔连接线段
        prev_time = None
        prev_value = None

        for time, value in zip(df['Time'], df[colum_key]):
            if prev_time is not None:
                time_diff = time - prev_time
                if time_diff < timedelta(seconds=120):  # 如果时间间隔小于120秒，则连接线段
                    plt.plot([prev_time, time], [prev_value, value], linestyle='-', color='b')
                else:  # 否则不连接线段
                    plt.scatter(time, value, color='b')  # 标记不连接的点

            prev_time = time
            prev_value = value

        # 设置x轴日期格式和刻度定位
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(date_locator)

        # 设置刻度朝向内部，并调整刻度与坐标轴的距离
        ax.tick_params(axis='x', direction='in', pad=10)
        ax.tick_params(axis='y', direction='in', pad=10)

        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=12)
        plt.grid(True)

        # 调整底部边界向上移动一点
        plt.subplots_adjust(bottom=0.15)
        
        # 保存图表到本地文件
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # @classmethod
    # def _plot_timeDF(cls, df: pd.DataFrame, colum_key: str, title: str, xlabel: str, ylabel: str, save_path: str) -> None:
    #     # 时间序列分析
    #     df['Time'] = pd.to_datetime(df['Time'])
    #     df = df.sort_values('Time')

    #     plt.figure(figsize=(9, 6))
    #     ax = plt.gca()

    #     # 设置日期格式化器和日期刻度定位器
    #     date_fmt = mdates.DateFormatter("%H:%M")  # 显示小时:分钟
    #     date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

    #     # 隐藏右边框和上边框
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)

    #     # 绘制折线图
    #     plt.plot(df['Time'], df[colum_key], marker='', linestyle='-', color='b')

    #     # 设置x轴日期格式和刻度定位
    #     ax.xaxis.set_major_formatter(date_fmt)
    #     ax.xaxis.set_major_locator(date_locator)

    #     # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    #     ax.tick_params(axis='x', direction='in', pad=10)
    #     ax.tick_params(axis='y', direction='in', pad=10)

    #     plt.xlabel(xlabel, fontsize=10)
    #     plt.ylabel(ylabel, fontsize=10)
    #     plt.title(title, fontsize=12)
    #     plt.grid(True)

    #     # 调整底部边界向上移动一点
    #     plt.subplots_adjust(bottom=0.15)
        
    #     # 保存图表到本地文件
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()

# class FileConfig:
#     def __init__(self):
#         self.avr_yesterday_file_path = r"/root/dataSave/avr_yesterday_data.txt"                    
#         self.ggkx_yesterday_file_path = r"/root/dataSave/ggkx_yesterday_data.txt"
#         self.met_yesterday_file_path = r"/root/dataSave/met_yesterday_data.txt"
#         self.tiltmeter_yesterday_file_path = r"/root/dataSave/tiltmeter_yesterday_data.txt"        

class FileConfig:
    def __init__(self):
        self.avr_yesterday_file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\avr_yesterday_data.txt"                     
        self.ggkx_yesterday_file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\ggkx_yesterday_data.txt"
        self.met_yesterday_file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\met_yesterday_data.txt"
        self.tiltmeter_yesterday_file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\tiltmeter_yesterday_data.txt"        

if __name__ =='__main__':
    print('-------------run------------')
    # sql_use = SQLUseType()
    # sql_use.input_emailInfo(host='47.98.201.213', user='root', password='TJ1qazXSW@', database='tianmeng_cableway')
    # sql_use.check_SQLInfo()

    # today = datetime.now().strftime('%Y-%m-%d')
    # yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

    # # 确保目标文件夹存在
    # output_folder = 'dataSave'
    # os.makedirs(output_folder, exist_ok=True)   
    # to_check_list = ['avr', 'ggkx', 'met', 'tiltmeter']
    # for item in to_check_list:
    #     # 构建查询语句获取昨天的数据
    #     if item == 'avr' or item == 'ggkx':
    #         query = rf"SELECT * FROM {item} WHERE Time >= '{yesterday}' AND Time < '{today}' AND FixMode = 3"
    #     else:
    #         query = rf"SELECT * FROM {item} WHERE Time >= '{yesterday}' AND Time < '{today}'"
    #     # 执行查询并保存到文件
    #     file_path = os.path.join(output_folder, rf'{item}_yesterday_data.txt')
    #     result = sql_use.execute_sql_and_save_to_txt(query, file_path)
    # print('------end------')
    dataprocess_exe = FileConfig()
    avr_df: pd.DataFrame = Avr._from_file(dataprocess_exe.avr_yesterday_file_path)
    ggkx_df: pd.DataFrame = Ggkx._from_file(dataprocess_exe.ggkx_yesterday_file_path)
    met_df: pd.DataFrame = Met._from_file(dataprocess_exe.met_yesterday_file_path)
    tiltmeter_df: pd.DataFrame = TiltmeterData._from_file(dataprocess_exe.tiltmeter_yesterday_file_path)

    # 创建汇总数据框
    datasets: dict[str, pd.DataFrame] = {
        'AVR': avr_df,
        'GGKX': ggkx_df,
        'MET': met_df,
        'Tiltmeter': tiltmeter_df
    }

    summary_df = pdDataFrameProcess._create_summary_dataframe(datasets)
    summary_str = pdDataFrameProcess._capture_dataframe_output(summary_df)
    avr_df_3, avr_df_8 = Avr._split_station_data(avr_df)

    # 使用 pdDataFrameProcess 类中的方法
    avr3_split_dict: dict[str, pd.DataFrame] = Avr._split_avr_columns(avr_df_3)
    avr8_split_dict: dict[str, pd.DataFrame] = Avr._split_avr_columns(avr_df_8)
    met_split_dict: dict[str, pd.DataFrame] = Met._split_met_columns(met_df)
    tiltmeter_split_dict: dict[str, pd.DataFrame] = TiltmeterData._split_tiltmeter_columns(tiltmeter_df)
    del avr_df_3, avr_df_8, met_df, ggkx_df, tiltmeter_df

    
    # 确保目标文件夹存在
    output_folder = 'output_images'
    os.makedirs(output_folder, exist_ok=True)   
    # 使用文件夹路径来保存图片
    for key, df in avr3_split_dict.items():
        save_path = os.path.join(output_folder, f'avr_03_{key}.png')
        TimeDataFrameMethod._plot_timeDF(df, colum_key=key, title=f'{key} Over Time', xlabel='Time', ylabel=key, save_path=save_path)
        
    for key, df in avr8_split_dict.items():
        save_path = os.path.join(output_folder, f'avr_08_{key}.png')
        TimeDataFrameMethod._plot_timeDF(df, colum_key=key, title=f'{key} Over Time', xlabel='Time', ylabel=key, save_path=save_path)
        
    for key, df in met_split_dict.items():
        save_path = os.path.join(output_folder, f'met_{key}.png')
        TimeDataFrameMethod._plot_timeDF(df, colum_key=key, title=f'{key} Over Time', xlabel='Time', ylabel=key, save_path=save_path)
        
    for key, df in tiltmeter_split_dict.items():
        save_path = os.path.join(output_folder, f'tiltmeter_{key}.png')
        TimeDataFrameMethod._plot_timeDF(df, colum_key=key, title=f'{key} Over Time', xlabel='Time', ylabel=key, save_path=save_path)

    # print('------send email------')
    # email_use = EmailUseType()
    # email_use._input_emailInfo(sender='3231518238@qq.com', passwd='ehrhfcvuhatucide', receiver='294448068@qq.com')
    # email_use._check_emailInfo()
    # email_use._send_QQ_email_mul(email_title=f"{datetime.now()}定时邮件", email_content=f'{summary_str}', annex_path=output_folder)
