import math
import random
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN
from sklearn.linear_model import Ridge ,Lasso, LassoCV,ElasticNet,LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error,r2_score,mean_squared_error
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import signal,fft
from scipy.stats import t,shapiro,pearsonr,f_oneway
from scipy.signal import hilbert,find_peaks
from PyEMD import EMD,EEMD, CEEMDAN
from typing import List, Optional
import pywt
import warnings
import time
"""
python:
O(n)算法: 输入n: 1,000,000 耗时: 15.312433242797852 ms
O(n^2)算法输入n: 10,000 耗时: 1610.5492115020752 ms
O(nlogn)算法输入n: 10,000 耗时: 5.4988861083984375 ms
"""
# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
# 禁用特定警告
warnings.filterwarnings('ignore', category=UserWarning, append=True)
# endregion
class TimeSeriesData:
    def __init__(self, value, date_time_str):
        self.value = value
        self.datetime = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")

    def __str__(self):
        return f"Value: {self.value}, Datetime: {self.datetime}"

# region 无关功能
def show_prcoess_time(start_time: float, addTxt: Optional[str] = None) -> None:
    end_time: float = time.time()  # 记录程序结束时间
    runtime: float = end_time - start_time  # 计算程序运行时间
    if addTxt is not None:
        print(f"{addTxt}运行时间：{runtime:.2f} 秒")
    else:
        print(f"程序运行时间：{runtime:.2f} 秒")
# endregion
# region 基础功能
def load_csv_data(data_file = "D:\python_proj\Data_dynamics\data_month_1.txt"):
    num_data_to_read = 1000000
    time_series_data = []
    count = 0

    with open(data_file, 'r') as file:
        for line in file:
            line_data = line.strip().split('\t')
            value = float(line_data[1])
            datetime = line_data[4]
            time_series_data.append(TimeSeriesData(value, datetime))
            
            count += 1
            if count == num_data_to_read:
                break
    return time_series_data
def create_time_series_data(values, datetimes):
    """给values和datetimes return一个time_series_data"""
    time_series_data = []
    for value, datetime in zip(values, datetimes):
        data = TimeSeriesData(value, datetime)
        time_series_data.append(data)
    return time_series_data

def plot_TimeSeriesData(TimeSeriesData: List[TimeSeriesData], isShow: bool = False, SaveFilePath: Optional[str] = None) -> None:
    """绘制TimeSeriesData对象的时间序列图"""
    values = [data.value for data in TimeSeriesData]
    datetimes = [data.datetime for data in TimeSeriesData]

    # 修改标签和标题的文本为中文
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.title('时间序列数据')

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d")  # 仅显示月-日
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    
    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置

    plt.plot(datetimes, values)

    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()

def plot_TimeSeriesData_Compare(TimeSeriesData1: List[TimeSeriesData], TimeSeriesData2: List[TimeSeriesData], SaveFilePath: Optional[str] = None) -> None:
    """绘制两个TimeSeriesData对象的时间序列图"""
    values1 = [data.value for data in TimeSeriesData1]
    datetimes1 = [data.datetime for data in TimeSeriesData1]

    values2 = [data.value for data in TimeSeriesData2]

    # 修改标签和标题的文本为中文
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.title('时间序列数据')

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d")  # 仅显示月-日
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    
    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置

    plt.plot(datetimes1, values1, label='TimeSeriesData1')
    plt.plot(datetimes1, values2, label='TimeSeriesData2')

    plt.legend()  # 显示图例

    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()
def generate_random_data() -> List[TimeSeriesData]:
    """生成随机的 TimeSeriesData 数据"""
    start_date: datetime = datetime(2024, 3, 13, 0, 0, 0)
    data: List[TimeSeriesData] = []

    for i in range(150):
        value: int = random.randint(10, 30)
        current_date: datetime = start_date + timedelta(hours=i)
        data_point: TimeSeriesData = TimeSeriesData(value=value, datetime=current_date)
        data.append(data_point)

    return data
# endregion
# region 基础统计分析
def calculate_mean(TimeSeriesData: List[TimeSeriesData]) -> float:
    """计算TimeSeriesData对象的平均值"""
    total = sum([data.value for data in TimeSeriesData])
    mean = total / len(TimeSeriesData)
    return mean

def calculate_median(TimeSeriesData: List[TimeSeriesData]) -> float:
    """计算TimeSeriesData对象的中位数"""
    sorted_data = sorted(TimeSeriesData, key=lambda data: data.value)
    n = len(sorted_data)
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1].value + sorted_data[n // 2].value) / 2
    else:
        median = sorted_data[n // 2].value
    return median

def calculate_variance(TimeSeriesData: List[TimeSeriesData]) -> float:
    """计算TimeSeriesData对象的方差"""
    mean = calculate_mean(TimeSeriesData)
    squared_diff = [(data.value - mean) ** 2 for data in TimeSeriesData]
    variance = sum(squared_diff) / len(TimeSeriesData)
    return variance

def calculate_standard_deviation(TimeSeriesData: List[TimeSeriesData]) -> float:
    """计算TimeSeriesData对象的标准差"""
    variance = calculate_variance(TimeSeriesData)
    standard_deviation = math.sqrt(variance)
    return standard_deviation

def calculate_change_rate(time_series_data: List[TimeSeriesData]) -> List[float]:
    """计算TimeSeriesData对象的变化率"""
    change_rates = []
    for i in range(1, len(time_series_data)):
        current_value = time_series_data[i].value
        previous_value = time_series_data[i-1].value
        change_rate = (current_value - previous_value) / previous_value
        change_rates.append(change_rate)
    return change_rates

def get_x_values(data: List) -> List[int]:
    """获取时间步作为x值"""
    # 获取时间步作为x值
    x = []
    for i in range(len(data)):
        x.append(i)
    return x

def get_y_values(data: List) -> List[float]:
    """获取数值作为y值"""
    # 获取数值作为y值
    y = []
    for i in range(len(data)):
        y.append(data[i].value)
    return y



def calculate_trend_line_residuals(data: List, trend_line: np.ndarray) -> np.ndarray:
    """计算残差"""
    y = get_y_values(data)
    residuals = np.array(y) - trend_line
    return residuals

def plot_residuals(residuals: np.ndarray, SaveFilePath: Optional[str] = None) -> None:
    """绘制残差图"""
    plt.figure(figsize=(25.6, 14.4))
    plt.scatter(range(len(residuals)), residuals, color='blue', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('残差图')
    plt.xlabel('数据点')
    plt.ylabel('残差')
    plt.grid(True)
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 R-squared (R方)"""
    mean_y_true = np.mean(y_true)
    total_sum_squares = np.sum((y_true - mean_y_true) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    return r_squared

def calculate_durbin_watson(residuals: np.ndarray) -> float:
    """手动计算德宾沃森统计量"""
    diff_residuals = np.diff(residuals)  # 计算残差的差分
    durbin_watson_statistic = np.sum(diff_residuals ** 2) / np.sum(residuals ** 2)
    return durbin_watson_statistic
# endregion
# region 拟合多项式趋势线
def fit_polynomial_trend(data: List, degree: int = 3) -> np.ndarray:
    """拟合多项式趋势线"""
    # 拟合多项式趋势线
    x = get_x_values(data)
    y = get_y_values(data)
    coefficients = np.polyfit(x, y, degree)
    trend_line = np.polyval(coefficients, x)
    return trend_line

def calculate_trend_line_r_squared(data: List, trend_line: np.ndarray) -> float:
    """计算趋势线拟合度"""
    y = get_y_values(data)
    y_mean = np.mean(y)
    
    # 计算总平方和 TSS
    tss = np.sum((y - y_mean) ** 2)
    
    # 计算残差平方和 ESS
    ess = np.sum((y - trend_line) ** 2)
    
    # 计算 R 方值
    r_squared = 1 - (ess / tss)
    
    return r_squared

def plot_trend_line_residuals(data: List, residuals: np.ndarray, SaveFilePath: Optional[str] = None) -> None:
    """绘制残差图"""
    x = get_x_values(data)
    plt.scatter(x, residuals)
    plt.xlabel('数据')
    plt.ylabel('残差')
    plt.title('残差图')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

def print_trend_line_residuals_Statistics(residuals: np.ndarray) -> None:
    """计算残差的统计指标"""
    # 计算残差的统计指标
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    residual_max = np.max(residuals)
    residual_min = np.min(residuals)

    # 执行正态性检验
    _, p_value = shapiro(residuals)
    is_normal = p_value > 0.05

    print("残差统计信息:")
    print("均值:", residual_mean)
    print("标准差:", residual_std)
    print("最大值:", residual_max)
    print("最小值:", residual_min)
    print("是否正态分布:", is_normal)

def plot_change_rate(change_rates: np.ndarray, SaveFilePath: Optional[str] = None) -> None:
    """对change_rate进行绘图"""
    # 绘制图表
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.plot(change_rates)
    plt.xlabel('索引')
    plt.ylabel('变化率')
    plt.title('变化率可视化')
    plt.grid(True)
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)  
    else:
        plt.show()
    plt.close()
    
def find_Ridge_best_degree(data):
    """找到 Ridge 回归模型最佳的阶数"""
    x = get_x_values(data)
    y = get_y_values(data)
    x = np.array(x)
    y = np.array(y)

    best_degree = None
    best_r_squared = -1  # 初始化为负数，确保能够更新
    best_mae = float('inf')  # 初始化为正无穷
    best_mape = float('inf')  # 初始化为正无穷
    best_bic = float('inf')  # 初始化为正无穷

    for degree in range(1, 21):  # 假设循环从1到21的范围
        # 创建 Polynomial 特征转换器
        polynomial_features = PolynomialFeatures(degree=degree)
        X_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

        # 创建 Ridge 回归模型
        ridge = Ridge()
        ridge.fit(X_poly, y)

        # 计算预测值
        y_pred = ridge.predict(X_poly)

        # 计算 R 方值
        r_squared = r2_score(y, y_pred)

        # 计算矫正决定系数
        n = len(y)
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - degree - 1)

        # 计算误差平方和 SSE
        sse = np.sum((y - y_pred) ** 2)

         # 计算均方误差 MSE
        mse = mean_squared_error(y, y_pred)

        # 计算平均绝对误差 MAE
        mae = mean_absolute_error(y, y_pred)

        # 计算平均绝对百分比误差 MAPE
        mape = np.mean(np.abs((y - y_pred) / y)) * 100

        # 计算模型参数数量
        num_params = X_poly.shape[1]
        
        # 计算 AIC 值
        aic = 2 * num_params - 2 * np.log(np.sum((y - y_pred) ** 2))

        # 计算 BIC 值
        bic = n * np.log(sse / n) + num_params * np.log(n)

        print(f"Degree {degree}: R方值 = {r_squared}, 矫正R方值 = {adjusted_r_squared}, SSE = {sse},  MSE = {mse},MAE = {mae}, MAPE = {mape}, AIC = {aic}, BIC = {bic}")

        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_degree = degree

        if mae < best_mae:
            best_mae = mae

        if mape < best_mape:
            best_mape = mape

        if bic < best_bic:
            best_bic = bic

    print(f"最佳Degree: {best_degree}, 对应的最佳R方值: {best_r_squared}, 最佳MAE: {best_mae}, 最佳MAPE: {best_mape}, 最佳BIC: {best_bic}")
    return best_degree, best_r_squared

def find_Lasso_best_degree(data):
    """找到 Lasso 回归模型最佳的阶数"""
    x = get_x_values(data)
    y = get_y_values(data)
    x=np.array(x)
    y=np.array(y)

    best_degree = None
    best_r_squared = -1  # 初始化为负数，确保能够更新

    for degree in range(1, 21):  # 假设循环从1到21的范围
        # 创建 Polynomial 特征转换器
        polynomial_features = PolynomialFeatures(degree=degree)
        X_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

        # 创建 Lasso 回归模型
        lasso = Lasso()
        lasso.fit(X_poly, y)

        # 计算预测值
        y_pred = lasso.predict(X_poly)

        # 计算 R 方值
        r_squared = r2_score(y, y_pred)

        print(f"Degree {degree}: R方值 = {r_squared}")

        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_degree = degree

    print(f"最佳Degree: {best_degree}, 对应的最佳R方值: {best_r_squared}")
    return best_degree, best_r_squared

def find_Ridge_best_degree_MSE(data):
    """找到 Ridge 回归模型最佳的阶数"""
    x = get_x_values(data)
    y = get_y_values(data)
    x=np.array(x)
    y=np.array(y)

    best_degree = None
    best_mse = float('inf')  # 初始化为正无穷大

    for degree in range(1, 21):  # 假设循环从1到21的范围
        # 创建 Polynomial 特征转换器
        polynomial_features = PolynomialFeatures(degree=degree)
        X_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

        # 创建 Ridge 回归模型
        ridge = Ridge()
        ridge.fit(X_poly, y)

        # 计算预测值
        y_pred = ridge.predict(X_poly)

        # 计算 MSE 值
        mse = mean_squared_error(y, y_pred)

        print(f"Degree {degree}: MSE = {mse}")

        if mse < best_mse:
            best_mse = mse
            best_degree = degree

    print(f"最佳Degree: {best_degree}, 对应的最佳MSE值: {best_mse}")
    return best_degree, best_mse

def calculate_AIC(data, y_pred, degree):
    """计算AIC值"""
    n = len(data)
    k = degree + 1  # 参数数量为阶数加上截距项
    mse = mean_squared_error(get_y_values(data), y_pred)
    likelihood = (n * math.log(2 * math.pi * mse) + n) / 2  # 根据高斯分布的似然函数计算
    AIC = 2 * k - 2 * math.log(likelihood)
    return AIC

def find_best_Lasso_degree_using_AIC(data):
    """利用AIC找到最佳的Lasso回归模型阶数"""
    x = get_x_values(data)
    y = get_y_values(data)
    x = np.array(x)
    y = np.array(y)

    best_degree = None
    best_AIC = float('inf')  # 初始化为正无穷大

    for degree in range(1, 21):  # 假设循环从1到21的范围
        # 创建 Polynomial 特征转换器
        polynomial_features = PolynomialFeatures(degree=degree)
        X_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

        # 创建 Lasso 回归模型
        lasso = Lasso()
        lasso.fit(X_poly, y)

        # 计算预测值
        y_pred = lasso.predict(X_poly)

        # 计算 AIC 值
        AIC = calculate_AIC(data, y_pred, degree)

        print(f"Degree {degree}: AIC = {AIC}")

        if AIC < best_AIC:
            best_AIC = AIC
            best_degree = degree

    print(f"最佳Degree: {best_degree}, 对应的最佳AIC值: {best_AIC}")
    return best_degree, best_AIC

def calculate_BIC(data, y_pred, degree):
    """计算BIC值"""
    n = len(data)
    k = degree + 1  # 参数数量为阶数加上截距项
    mse = mean_squared_error(get_y_values(data), y_pred)
    likelihood = (n * math.log(2 * math.pi * mse) + n) / 2  # 根据高斯分布的似然函数计算
    BIC = k * math.log(n) - 2 * math.log(likelihood)
    return BIC


def find_best_Lasso_degree_using_BIC(data):
    """利用BIC找到最佳的Lasso回归模型阶数"""
    x = get_x_values(data)
    y = get_y_values(data)
    x = np.array(x)
    y = np.array(y)

    best_degree = None
    best_BIC = float('inf')  # 初始化为正无穷大

    for degree in range(1, 21):  # 假设循环从1到21的范围
        # 创建 Polynomial 特征转换器
        polynomial_features = PolynomialFeatures(degree=degree)
        X_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

        # 创建 Lasso 回归模型
        lasso = Lasso()
        lasso.fit(X_poly, y)

        # 计算预测值
        y_pred = lasso.predict(X_poly)

        # 计算 BIC 值
        BIC = calculate_BIC(data, y_pred, degree)

        print(f"Degree {degree}: BIC = {BIC}")

        if BIC < best_BIC:
            best_BIC = BIC
            best_degree = degree

    print(f"最佳Degree: {best_degree}, 对应的最佳BIC值: {best_BIC}")
    return best_degree, best_BIC

def fit_Ridge_polynomial_trend(data, degree=11, alpha=1.0):
    """使用岭回归拟合多项式趋势线"""
    x = np.array(get_x_values(data))
    y = np.array([point.value for point in data])

    # 创建 Polynomial 特征转换器
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

    # 创建岭回归模型
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_poly, y)

    # 预测拟合结果
    y_pred = ridge.predict(X_poly)

    return y_pred

def fit_Lasso_polynomial_trend(data, degree=11, alpha=1.0):
    """使用Lasso回归拟合多项式趋势线"""
    x = np.array(get_x_values(data))
    y = np.array([point.value for point in data])

    # 创建 Polynomial 特征转换器
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

    # 创建Lasso回归模型
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_poly, y)

    # 预测拟合结果
    y_pred = lasso.predict(X_poly)

    return y_pred

def fit_ElasticNet_polynomial_trend(data, degree=11, alpha=1.0, l1_ratio=0.5):
    """使用弹性网络回归拟合多项式趋势线"""
    x = np.array(get_x_values(data))
    y = np.array([point.value for point in data])

    # 创建 Polynomial 特征转换器
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

    # 创建弹性网络回归模型
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elastic_net.fit(X_poly, y)

    # 预测拟合结果
    y_pred = elastic_net.predict(X_poly)

    return y_pred

def evaluate_predictions(ridge_pred, lasso_pred, elastic_pred, polynomial_pred, data,degree):
    """评估 Ridge、Lasso、ElasticNet 和多项式拟合的预测结果拟合效果"""
    y_true = get_y_values(data)
    # 计算 R方值
    ridge_r2 = r2_score(y_true, ridge_pred)
    lasso_r2 = r2_score(y_true, lasso_pred)
    elastic_r2 = r2_score(y_true, elastic_pred)
    polynomial_r2 = r2_score(y_true, polynomial_pred)

    # 计算均方误差（MSE）
    ridge_mse = mean_squared_error(y_true, ridge_pred)
    lasso_mse = mean_squared_error(y_true, lasso_pred)
    elastic_mse = mean_squared_error(y_true, elastic_pred)
    polynomial_mse = mean_squared_error(y_true, polynomial_pred)

    # 计算模型参数数量
    num_parameters_ridge = degree 
    num_parameters_lasso = degree 
    num_parameters_elastic = degree 
    num_parameters_polynomial = degree  # 多项式拟合的参数数量为多项式次数 + 1

    # 计算 AIC 和 BIC
    aic_ridge = calculate_AIC(data, ridge_pred, num_parameters_ridge)
    bic_ridge = calculate_BIC(data, ridge_pred, num_parameters_ridge)
    aic_lasso = calculate_AIC(data, lasso_pred, num_parameters_lasso)
    bic_lasso = calculate_BIC(data, lasso_pred, num_parameters_lasso)
    aic_elastic = calculate_AIC(data, elastic_pred, num_parameters_elastic)
    bic_elastic = calculate_BIC(data, elastic_pred, num_parameters_elastic)
    aic_polynomial = calculate_AIC(data, polynomial_pred, num_parameters_polynomial)
    bic_polynomial = calculate_BIC(data, polynomial_pred, num_parameters_polynomial)

    # 打印结果
    print("Ridge回归模型的评估结果:")
    print("R方值: ", ridge_r2)
    print("均方误差 (MSE): ", ridge_mse)
    print("AIC: ", aic_ridge)
    print("BIC: ", bic_ridge)

    print("\nLasso回归模型的评估结果:")
    print("R方值: ", lasso_r2)
    print("均方误差 (MSE): ", lasso_mse)
    print("AIC: ", aic_lasso)
    print("BIC: ", bic_lasso)

    print("\nElasticNet回归模型的评估结果:")
    print("R方值: ", elastic_r2)
    print("均方误差 (MSE): ", elastic_mse)
    print("AIC: ", aic_elastic)
    print("BIC: ", bic_elastic)

    print("\n多项式回归模型的评估结果:")
    print("R方值: ", polynomial_r2)
    print("均方误差 (MSE): ", polynomial_mse)
    print("AIC: ", aic_polynomial)
    print("BIC: ", bic_polynomial)

    # 比较 R方值
    max_r2 = max(ridge_r2, lasso_r2, elastic_r2, polynomial_r2)
    if max_r2 == ridge_r2:
        print("\nRidge回归模型的拟合效果最好")
    elif max_r2 == lasso_r2:
        print("\nLasso回归模型的拟合效果最好")
    elif max_r2 == elastic_r2:
        print("\nElasticNet回归模型的拟合效果最好")
    else:
        print("\n多项式回归模型的拟合效果最好")

    # 比较 AIC
    min_aic = min(aic_ridge, aic_lasso, aic_elastic, aic_polynomial)
    if min_aic == aic_ridge:
        print("Ridge回归模型的拟合效果最好 (最小AIC)")
    elif min_aic == aic_lasso:
        print("Lasso回归模型的拟合效果最好 (最小AIC)")
    elif min_aic == aic_elastic:
        print("ElasticNet回归模型的拟合效果最好 (最小AIC)")
    else:
        print("多项式回归模型的拟合效果最好 (最小AIC)")

    # 比较 BIC
    min_bic = min(bic_ridge, bic_lasso, bic_elastic, bic_polynomial)
    if min_bic == bic_ridge:
        print("Ridge回归模型的拟合效果最好 (最小BIC)")
    elif min_bic == bic_lasso:
        print("Lasso回归模型的拟合效果最好 (最小BIC)")
    elif min_bic == bic_elastic:
        print("ElasticNet回归模型的拟合效果最好 (最小BIC)")
    else:
        print("多项式回归模型的拟合效果最好 (最小BIC)")

def plot_polynomial_trend(data, y_pred, SaveFilePath=None):
    """绘制多项式趋势线"""
    # 提取日期时间和数值信息
    datetimes = [point.datetime for point in data]
    values = [point.value for point in data]

    # 修改标签和标题的文本为中文
    plt.figure(figsize=(25.6, 14.4))  # 设置图形大小
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.title('多项式趋势线')

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d")  # 仅显示月-日
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    
    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置

    # 绘制折线图
    plt.plot(datetimes, values, label='原始数据', color='gray')
    plt.plot(datetimes, y_pred, label='多项式趋势线', color='r')

    plt.legend()  # 添加图例
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

def visualize_predictions(TimeSeriesData, ridge_pred, lasso_pred, elastic_pred, polynomial_pred, SaveFilePath=None):
    x = np.array([point.datetime for point in TimeSeriesData])
    y_true = np.array([point.value for point in TimeSeriesData])

    plt.figure(figsize=(25.6, 14.4))
    plt.plot(x, y_true, color='blue', label='真实数据', alpha=0.5)
    plt.plot(x, ridge_pred, color='red', label='岭回归', linewidth=2)
    plt.plot(x, lasso_pred, color='green', label='Lasso回归', linewidth=2)
    plt.plot(x, elastic_pred, color='orange', label='弹性网络回归', linewidth=2)
    plt.plot(x, polynomial_pred, color='purple', label='多项式拟合', linewidth=2)

    plt.title('多项式趋势回归')
    plt.xlabel('时间')
    plt.ylabel('数值')
    plt.legend(loc='best')
    plt.grid(True)
    
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath) 
    else:
        plt.show()
        
    plt.close()

def compare_trend_lines(ridge_pred, lasso_pred, elastic_pred, polynomial_pred):
    # 计算欧几里得距离
    euclidean_distances = {
        ('ridge', 'lasso'): euclidean(ridge_pred, lasso_pred),
        ('ridge', 'elastic'): euclidean(ridge_pred, elastic_pred),
        ('ridge', 'polynomial'): euclidean(ridge_pred, polynomial_pred),
        ('lasso', 'elastic'): euclidean(lasso_pred, elastic_pred),
        ('lasso', 'polynomial'): euclidean(lasso_pred, polynomial_pred),
        ('elastic', 'polynomial'): euclidean(elastic_pred, polynomial_pred)
    }

    # 计算皮尔逊相关系数
    pearson_correlations = {
        ('ridge', 'lasso'): pearsonr(ridge_pred, lasso_pred)[0],
        ('ridge', 'elastic'): pearsonr(ridge_pred, elastic_pred)[0],
        ('ridge', 'polynomial'): pearsonr(ridge_pred, polynomial_pred)[0],
        ('lasso', 'elastic'): pearsonr(lasso_pred, elastic_pred)[0],
        ('lasso', 'polynomial'): pearsonr(lasso_pred, polynomial_pred)[0],
        ('elastic', 'polynomial'): pearsonr(elastic_pred, polynomial_pred)[0]
    }

    # 计算ANOVA
    f_statistic, p_value = f_oneway(ridge_pred, lasso_pred, elastic_pred, polynomial_pred)

    return euclidean_distances, pearson_correlations, f_statistic, p_value

def find_Ridge_best_degree_without_print(data:List[TimeSeriesData]) -> int:
    """找到 Ridge 回归模型最佳的阶数"""
    x = get_x_values(data)
    y = get_y_values(data)
    x = np.array(x)
    y = np.array(y)

    best_degree = None
    best_bic = float('inf')  # 初始化为正无穷

    for degree in range(1, 21):  # 假设循环从1到21的范围
        # 创建 Polynomial 特征转换器
        polynomial_features = PolynomialFeatures(degree=degree)
        X_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

        # 创建 Ridge 回归模型
        ridge = Ridge()
        ridge.fit(X_poly, y)

        # 计算预测值
        y_pred = ridge.predict(X_poly)

        # 计算矫正决定系数
        n = len(y)

        # 计算误差平方和 SSE
        sse = np.sum((y - y_pred) ** 2)

        # 计算模型参数数量
        num_params = X_poly.shape[1]
        
        # 计算 AIC 值

        # 计算 BIC 值
        bic = n * np.log(sse / n) + num_params * np.log(n)

        
        # 更新最佳degree和对应的评价指标
        if bic < best_bic:
            best_degree = degree
            best_bic = bic

    return best_degree


def calculate_acceleration(data: List[TimeSeriesData]) -> List[float]:
    """
    计算加速度序列
    时间复杂度为 ( O(n) )
    """
    accelerations = []
    for i in range(1, len(data) - 1):
        dt1 = (data[i].datetime - data[i-1].datetime).total_seconds()
        dt2 = (data[i+1].datetime - data[i].datetime).total_seconds()
        dv1 = data[i].value - data[i-1].value
        dv2 = data[i+1].value - data[i].value
        acceleration = (dv2/dt2 - dv1/dt1) / ((dt1 + dt2) / 2)
        accelerations.append(acceleration)
    return accelerations

def plot_acceleration(data: List[TimeSeriesData], accelerations: List[float]) -> None:
    timestamps = [data[i].datetime for i in range(1, len(data) - 1)]
    
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, accelerations, color='blue', marker='.', linestyle='-')  # 将 marker 参数设置为更小的点
    plt.title('加速度随时间变化图')
    plt.xlabel('时间戳')
    plt.ylabel('加速度')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def fit_polynomial_trend_line(data: List[TimeSeriesData], window: int, degree: int) -> np.ndarray:
    # 将数据准备为特征矩阵
    x = np.array(range(window)).reshape(-1, 1)
    y = np.array([data[i].value for i in range(window)])
    
    # 使用多项式特征转换器
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    
    # 应用 Ridge 回归模型
    ridge = Ridge()
    ridge.fit(x_poly, y)
    
    # 预测趋势线
    trend_line = ridge.predict(x_poly)
    
    return trend_line

def adaptive_polynomial_fitting(data: List[TimeSeriesData], initial_threshold: float) -> np.ndarray:
    """
    自适应拟合趋势线
    外层循环：外层循环 while n < N - 1: 的执行次数取决于数据的长度 N。假设数据长度为 N,那么外层循环的执行次数为 N 次，时间复杂度为 O(N)。
    内层循环：内层循环 while n + r < N - 1:在最坏情况下，内层循环的执行次数可能会达到 O(N)。
    拟合趋势线和计算相关系数：假设这两个步骤的时间复杂度为 O(M)，其中 M 是数据的长度和拟合多项式的阶数中较大的一个。
    总的时间复杂度可以近似为 O(N^2 * M)，其中 N 是数据的长度,M 是拟合多项式的阶数。
    """
    N = len(data)
    trend_lines = []  # 存储所有拟合的趋势线
    accelerations = calculate_all_accelerations(data)  # 预先计算所有加速度值

    n = 0
    while n < N - 1:
        r = 1
        while n + r < N - 1:
            segment_accelerations = accelerations[n:n+r+1]
            Δa = max(segment_accelerations) - min(segment_accelerations)
            if Δa < initial_threshold:
                break
            r += 1
        
        # 拟合多项式趋势线并计算相关系数
        best_degree = find_Ridge_best_degree(data)
        trend_line = fit_polynomial_trend_line(data, r, best_degree)
        trend_lines.append(trend_line)  # 将当前窗口的拟合趋势线添加到列表中
        
        correlation = np.corrcoef([data[i].value for i in range(n, n+r)], trend_line)[0, 1]
        
        # 检查相关系数是否满足条件
        if correlation < 0.8:
            initial_threshold *= 0.9  # 调整阈值
            continue

        # 更新索引并重置r
        n += r
        r = 1

    # 将所有拟合的趋势线连接起来，得到完整的趋势线
    complete_trend_line = np.concatenate(trend_lines)
    
    return complete_trend_line
# endregion
# region 频域分析
def fourier_transform(TimeSeriesData):
    """对TimeSeriesData对象进行傅里叶变换"""
    values = [data.value for data in TimeSeriesData]
    transformed_values = np.fft.fft(values)
    return transformed_values

def extract_frequency_features(transformed_values, sample_rate):
    """
    从傅里叶变换结果中提取频域特征
    输入参数:
    - transformed_values: 傅里叶变换结果
    - sample_rate: 采样率(每秒钟的样本数)
    返回值:
    - frequency_features: 提取的频域特征
    """
    # 获取频率轴
    freqs = np.fft.fftfreq(len(transformed_values), d=1/sample_rate)

    # 计算频域特征
    # 清除零频率分量
    non_zero_indices = np.where(freqs != 0)[0]
    freqs = freqs[non_zero_indices]
    magnitude_spectrum = np.abs(transformed_values)[non_zero_indices]
    phase_spectrum = np.angle(transformed_values)[non_zero_indices]

    # 构造频域特征字典
    frequency_features = {
        'frequency': freqs,
        'magnitude_spectrum': magnitude_spectrum,
        'phase_spectrum': phase_spectrum
    }

    return frequency_features

def plot_frequency_features(frequency_features, SaveFilePath=None):
    """
    可视化频域特征
    输入参数:
    - frequency_features: 频域特征字典
    """
    freqs = frequency_features['frequency']
    magnitude_spectrum = frequency_features['magnitude_spectrum']
    phase_spectrum = frequency_features['phase_spectrum']
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    # 可视化幅度谱
    plt.plot(freqs, magnitude_spectrum)
    plt.title('幅度谱')
    plt.xlabel('频率')
    plt.ylabel('幅度')

    # 可视化相位谱
    plt.plot(freqs, phase_spectrum)
    plt.title('相位谱')
    plt.xlabel('频率')
    plt.ylabel('相位')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

def analyze_frequency_components(frequency_features, threshold=None):
    """
    分析频率成分并查找峰值频率
    输入参数:
    - frequency_features: 频域特征字典
    - threshold: 峰值检测的阈值,可选参数,默认为None
    返回值:
    - peak_frequencies: 峰值频率
    - peak_values: 峰值对应的幅度值

    通过设置threshold参数来指定峰值检测的阈值。较大的阈值可以过滤掉小幅度的峰值。
    通过调整threshold参数的值,你可以控制检测到的峰值数量和灵敏度。进一步分析和处理这些峰值频率,可以获得关于主要频率成分的更多信息,例如频率的分布、频率的演化趋势等。
    
    threshold = None  # 可选参数,设定峰值检测的阈值,默认为None
    peak_frequencies, peak_values = analyze_frequency_components(frequency_features, threshold)

    """

    magnitude_spectrum = frequency_features['magnitude_spectrum']

    # 使用峰值检测算法查找幅度谱中的峰值
    peaks, _ = find_peaks(magnitude_spectrum, height=threshold)

    # 获取峰值对应的频率和幅度值
    peak_frequencies = frequency_features['frequency'][peaks]
    peak_values = magnitude_spectrum[peaks]

    return peak_frequencies, peak_values

def plot_frequency_components(peak_frequencies, peak_values, SaveFilePath=None):
    """
    可视化峰值频率和幅度值
    输入参数:
    - peak_frequencies: 峰值频率
    - peak_values: 峰值对应的幅度值

    peak_frequencies, peak_values = analyze_frequency_components(frequency_features, threshold)
    # 可视化峰值频率和幅度值
    plot_frequency_components(peak_frequencies, peak_values)
    """
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.plot(peak_frequencies, peak_values, 'ro')
    plt.title('频率成分')
    plt.xlabel('频率')
    plt.ylabel('幅度')
    plt.grid(True)
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()
    
def lowpass_filter(transformed_values, cutoff_freq):
    """
    使用低通滤波器去除高频噪声
    输入参数:
    - transformed_values: 傅里叶变换后的数据
    - cutoff_freq: 截止频率
    返回值:
    - filtered_data: 去除高频噪声后的数据
    """
    # 获取傅里叶频谱的频率轴
    freqs = np.fft.fftfreq(len(transformed_values))

    # 将高于截止频率的频谱部分设为0(去除高频噪声)
    transformed_values[np.abs(freqs) > cutoff_freq] = 0

    # 对处理后的数据进行逆傅里叶变换
    filtered_data = np.fft.ifft(transformed_values)

    return filtered_data

def highpass_filter(transformed_values, cutoff_freq):
    """
    使用高通滤波器去除低频趋势
    输入参数:
    - transformed_values: 傅里叶变换后的数据
    - cutoff_freq: 截止频率
    返回值:
    - filtered_data: 去除低频趋势后的数据
    """
    # 获取傅里叶频谱的频率轴
    freqs = np.fft.fftfreq(len(transformed_values))

    # 将低于截止频率的频谱部分设为0(去除低频趋势)
    transformed_values[np.abs(freqs) < cutoff_freq] = 0

    # 对处理后的数据进行逆傅里叶变换
    filtered_data = np.fft.ifft(transformed_values)

    return filtered_data

def bandpass_filter(transformed_values, freq_low, freq_high):
    """
    使用带通滤波器只保留特定频段的信号
    输入参数:
    - transformed_values: 傅里叶变换后的数据
    - freq_low: 保留频段的下限
    - freq_high: 保留频段的上限
    返回值:
    - filtered_data: 保留特定频段后的数据
    """
    # 获取傅里叶频谱的频率轴
    freqs = np.fft.fftfreq(len(transformed_values))

    # 将频率轴范围外的频率部分设为0
    transformed_values[(np.abs(freqs) < freq_low) | (np.abs(freqs) > freq_high)] = 0

    # 对处理后的数据进行逆傅里叶变换
    filtered_data = np.fft.ifft(transformed_values)

    return filtered_data

def remove_noise(transformed_values, threshold):
    """
    在频域中识别并去除噪声成分
    输入参数:
    - transformed_values: 傅里叶变换后的数据
    - threshold: 噪声判断阈值
    返回值:
    - filtered_data: 去除噪声成分后的数据
    """
    # 获取傅里叶频谱的振幅
    amplitudes = np.abs(transformed_values)

    # 将低于阈值的振幅部分设为0(去除噪声成分)
    transformed_values[amplitudes < threshold] = 0

    # 对处理后的数据进行逆傅里叶变换
    filtered_data = np.fft.ifft(transformed_values)

    return filtered_data

def plot_filtered_data(filtered_data,SaveFilePath=None):
    """
    可视化去除低频趋势后的数据
    输入参数:
    - filtered_data: 去除低频趋势后的数据
    """
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.plot(filtered_data)
    plt.title('滤波数据')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

def plot_harmonic_frequency_distribution(peak_frequencies, SaveFilePath=None):
    """
    绘制谐波频率的直方图
    输入参数:
    - peak_frequencies: 谐波频率
    """
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.hist(peak_frequencies, bins='auto')
    plt.title('谐波频率分布')
    plt.xlabel('频率')
    plt.ylabel('频次')
    plt.grid(True)
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

def plot_harmonic_frequency_bar_chart(peak_frequencies, peak_values, SaveFilePath=None):
    """
    绘制谐波频率的条形图
    输入参数:
    - peak_frequencies: 谐波频率
    - peak_values: 谐波频率对应的幅度值
    """
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.bar(peak_frequencies, peak_values)
    plt.title('谐波频率分布')
    plt.xlabel('频率')
    plt.ylabel('幅度')
    plt.grid(True)
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()

def analyze_fourier_transform_results(data, SaveFilePath=None):
    """
    函数 analyze_fourier_transform_results 用于计算和可视化 TimeSeriesData 数据的振幅谱,并识别数据中的主要周期分量。
    
    :param data: TimeSeriesData 类型的列表,包含 value 和 datetime 属性。
    """
    # 提取 value 值和构建时间序列
    values = [entry.value for entry in data]
    datetime = [entry.datetime for entry in data]

    # 计算采样率
    sampling_rate = (data[-1].datetime - data[0].datetime).total_seconds() / len(data)

    # 进行傅里叶变换
    transformed_values = np.fft.fft(values)

    # 构建频率轴
    N = len(data)  # 数据的长度
    frequencies = np.fft.fftfreq(N, d=sampling_rate)

    # 获取右半边的频率和对应的振幅值
    frequencies_right = frequencies[:N//2+1]
    transformed_values_right = transformed_values[:N//2+1]
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    # 可视化振幅谱
    plt.plot(frequencies_right, np.abs(transformed_values_right))
    plt.xlabel('频率')
    plt.ylabel('振幅')
    plt.title('振幅谱')
    plt.grid(True)
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

    """
    通过绘制频谱的幅度谱图,可以观察不同频率成分的能量分布情况。从图中你可以获取以下信息:
    峰值表示在该频率上存在主要的周期性成分。
    频谱中的宽峰表示存在多个相关频率的周期性成分。
    幅度谱中较低的值表示在该频率上不存在明显的周期性成分。
    """

def chebyshev_filter(TimeSeriesData, cutoff_freq=0.8, order=4, filter_type='lowpass'):
    """
    使用切比雪夫滤波器对时序数据进行滤波
    输入参数:
    - data: 时序数据的数组或列表
    - cutoff_freq: 截止频率
    - order: 滤波器的阶数(低阶数表示较为平滑的滤波)
    - filter_type: 滤波器的类型,可选参数为'lowpass', 'highpass', 'bandpass'和'bandstop'(默认为'lowpass')
    返回值:
    - filtered_data: 经过滤波处理后的数据
    - b,a: IIR滤波器的分子(b)和分母(a)多项式系数向量。output='ba'
    高通滤波
    b, a = signal.butter(8, 0.02, 'highpass')
    filtedData = signal.filtfilt(b, a, data)#data为要过滤的信号
    低通滤波
    b, a = signal.butter(8, 0.02, 'lowpass') 
    filtedData = signal.filtfilt(b, a, data)       #data为要过滤的信号
    带通滤波
    b, a = signal.butter(8, [0.02,0.8], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)   #data为要过滤的信号
    """

    # 将输入转换为numpy数组
    data = np.array([data.value for data in TimeSeriesData])

    # 计算滤波器的参数
    b, a = signal.cheby1(order, 0.5, cutoff_freq, btype=filter_type, analog=False, output='ba')

    # 应用滤波器
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data
# endregion
# region 移动平均

def moving_average(TimeSeriesData, window_size):
    """计算移动平均值"""
    values = [data.value for data in TimeSeriesData]
    datetimes = [data.datetime for data in TimeSeriesData]
    n = len(values)
    moving_avg = []

    for i in range(n - window_size + 1):
        window_values = values[i : i + window_size]
        avg = sum(window_values) / window_size
        moving_avg.append(avg)

    return moving_avg
def plot_moving_average(TimeSeriesData, window_size, SaveFilePath=None):
    """绘制移动平均线"""
    avg_values = moving_average(TimeSeriesData, window_size)
    datetimes = [data.datetime for data in TimeSeriesData]
    moving_date=datetimes[window_size - 1:]

    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.plot(moving_date, avg_values, label="移动平均")
    plt.xlabel('日期')  # 指定中文标签
    plt.ylabel('数值') # 指定中文标签
    plt.title('移动平均线')  # 指定中文标签

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d")  # 仅显示月-日
    date_locator = mdates.DayLocator()  # 每天显示一个刻度

    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()

# endregion
# region statsmodels的运用
def plot_time_series_decomposition(time_series, SaveFilePath=None):
    """进行季节性分解"""
    # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    index = pd.to_datetime(datetimes)
    index = pd.DatetimeIndex(index, freq='2D')
    ts = pd.Series(values, index=index)

  
    # 进行季节性分解
    decomposition = sm.tsa.seasonal_decompose(ts, model='additive')

    # 提取分解后的各部分
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # 绘制分解后的组成部分
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ts.plot(ax=axes[0])
    axes[0].set_ylabel('原始数据')
    trend.plot(ax=axes[1])
    axes[1].set_ylabel('趋势')
    seasonal.plot(ax=axes[2])
    axes[2].set_ylabel('季节性')
    residual.plot(ax=axes[3])
    axes[3].set_ylabel('残差')
    
    plt.xlabel('日期')
    plt.tight_layout()
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()

def stl_decomposition(time_series, SaveFilePath=None):
    """进行季节性分解时序回归模型(STL)"""
    # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    index = pd.to_datetime(datetimes)
    index = pd.DatetimeIndex(index, freq='2D')
    ts = pd.Series(values, index=index)

    # 进行季节性分解时序回归模型(STL)
    result = sm.tsa.STL(ts, seasonal=13).fit()  # 以13为季节周期,可以根据需要进行调整

    # 提取分解后的各部分
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # 绘制分解后的组成部分
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ts.plot(ax=axes[0])
    axes[0].set_ylabel('原始数据')
    trend.plot(ax=axes[1])
    axes[1].set_ylabel('趋势')
    seasonal.plot(ax=axes[2])
    axes[2].set_ylabel('季节性')
    residual.plot(ax=axes[3])
    axes[3].set_ylabel('残差')
    
    plt.xlabel('日期')
    plt.tight_layout()
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()


# 创建ARMA模型
def create_arma_model(time_series, order):
     # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    ts = pd.Series(values, index=datetimes)
    model = sm.tsa.ARMA(ts, order=order).fit()
    return model

# 创建ARIMA模型
def create_arima_model(time_series, order):
    # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    ts = pd.Series(values, index=datetimes)
    model = sm.tsa.ARIMA(ts, order=order).fit()
    return model

def predict_analyze_evaluate(time_series, order=(2, 1,1),SaveFilePath=None):
    # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    ts = pd.Series(values, index=datetimes)
    
    # 创建ARIMA模型
    arima_model = sm.tsa.ARIMA(ts, order=order).fit()
    
    arima_predictions = arima_model.predict(start=len(ts), end=len(ts)+2)
    
    print("ARIMA模型预测结果:", arima_predictions)
    
    # 残差分析
    arima_residuals = arima_model.resid
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.subplot(1, 2, 1)
    plt.plot(arima_residuals)
    plt.xlabel('时间')
    plt.ylabel('残差')
    plt.title('ARIMA模型残差图')

    plt.tight_layout()
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()
    
    # 模型评估
    arima_aic = arima_model.aic

    print("ARIMA模型AIC:", arima_aic)
# endregion
# region 突变检测
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

    # 绘图代码可以在这里添加,如果需要的话

    return K

def pettitt_change_point_detection(data):
    """
    使用Pettitt突变检测方法检测时间序列数据中的突变点。

    :param data: TimeSeriesData 类型的列表,包含 value 和 datetime 属性。
    :return: 突变点的位置和统计量。
    """
    # 提取 value 值
    values = [entry.value for entry in data]

    # 计算累积和
    cumulative_sum = np.cumsum(values)

    # 突变点的位置和统计量
    change_point = 0
    max_test_statistic = 0

    n = len(values)

    for i in range(n):
        current_statistic = abs(cumulative_sum[i] - cumulative_sum[n-i-1])
        if current_statistic > max_test_statistic:
            max_test_statistic = current_statistic
            change_point = i

    return change_point, max_test_statistic

def cusum(data, threshold=1):
    """
    计算CUSUM
    # 设置阈值,根据具体情况调整
    """
    # 计算CUSUM
    cusum_values = [0]  # 起始值为0
   
    for i in range(1, len(data)):
        diff = data[i].value - data[i-1].value
        cusum_values.append(max(0, cusum_values[i-1] + diff - threshold))
    
    return cusum_values

def plot_cusum(data,cusum_values, SaveFilePath=None):
    """绘制CUSUM"""
    # 绘制CUSUM
    x = np.arange(len(data))
    y = get_y_values(data)
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.plot(x, y, label='时序数据')
    plt.plot(x, cusum_values, label='CUSUM')
    plt.xlabel('时间步')
    plt.ylabel('数值')
    plt.legend()
    plt.title('时序数据的CUSUM')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()

def detect_cusum_threshold(cusum_values, threshold):
    """
    对cusum_values设置阈值,并将超过阈值的值作为变化点
    """
    change_points = []
    for i, value in enumerate(cusum_values):
        if value > threshold:
            change_points.append(i)
    
    return change_points

def detect_cusum_diff_threshold(cusum_values, threshold):
    """
    计算相邻CUSUM值的差异diff，如果差异超过了设定的阈值threshold
    """
    change_points = []
  
    for i in range(1, len(cusum_values)):
        diff = cusum_values[i] - cusum_values[i-1]
        if diff > threshold:
            change_points.append(i)
    
    return change_points
def detect_cusum_window(cusum_values, threshold, window_size):
    """
    滑动窗口来检测连续的CUSUM值
    使用启发式规则:一些经验法则建议窗口大小选择为数据点总数的一定比例,例如窗口大小为数据总点数的1/10或1/20。
    统计方法:可以根据数据的统计特征来选择阈值。例如,基于数据的标准差、平均值等来设置阈值,使得超过阈值的CUSUM值被认为是结构性变化点。
    可视化和交互:可以通过可视化CUSUM图形,并与领域专家或数据分析人员进行交互来优化阈值的选择。观察图形中的结构性变化,根据专家意见或实际需求来调整阈值。
    """
    change_points = []

    for i in range(window_size, len(cusum_values)):
        sub_cusum = cusum_values[i-window_size:i+1]

        if all(value >= threshold for value in sub_cusum):
            change_points.append(i)
    
    return change_points

def calculate_rolling_std(data, window_size):
    """滚动标准差来检测波动性的变动"""
    values = [item.value for item in data]  # 提取时序数据中的值
    rolling_std = np.std(values[:window_size])  # 初始窗口的标准差
    std_values = [rolling_std]

    for i in range(window_size, len(values)):
        window_values = values[i-window_size+1:i+1]
        rolling_std = np.std(window_values)
        std_values.append(rolling_std)
    
    return std_values
def plot_std_values(std_values, SaveFilePath=None):
    x = range(len(std_values))  # x轴为数据点的索引
    y = std_values  # y轴为滚动标准差的值
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.plot(x, y)
    plt.xlabel('数据点')  # x轴标签
    plt.ylabel('标准差')  # y轴标签
    plt.title('滚动标准差')  # 图表标题
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()

def apply_grubbs_test(data, alpha=0.05):
    """实现格拉布斯检验函数"""
    values = [item.value for item in data]
    n = len(values)
    outliers = []

    while True:
        mean = np.mean(values)
        std = np.std(values)
        t_critical = t.ppf(1 - alpha / (2*n), n - 2)
        max_residual = np.max(np.abs(values - mean))
        max_residual_idx = np.argmax(np.abs(values - mean))
        test_statistic = max_residual / std

        if test_statistic < t_critical:
            break

        outliers.append(data.pop(max_residual_idx))
        values = [item.value for item in data]
        n -= 1

    return outliers

def calculate_z_scores(data):
    """计算Z分数"""
    values = [item.value for item in data]
    mean = np.mean(values)
    std = np.std(values)
    z_scores = []
    for item in data:
        z_score = (item.value - mean) / std
        z_scores.append(z_score)
    return z_scores
def detect_outliers(z_scores, threshold=3):
    """标记异常值：
    z_scores = calculate_z_scores(data)
    outliers = detect_outliers(z_scores, threshold=3)
    """
    outliers = []
    for i, z_score in enumerate(z_scores):
        if z_score < -threshold or z_score > threshold:
            outliers.append(i)
    return outliers

def apply_dbscan_clustering(data, epsilon, min_samples):
    """
    实现DBSCAN聚类函数
    epsilon = 0.5  # DBSCAN的邻域半径
    min_samples = 5  # 聚类的最小样本数
    clusters = apply_dbscan_clustering(data, epsilon, min_samples)
    """
    values = np.array([item.value for item in data]).reshape(-1, 1)
    
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(values)
    labels = dbscan.labels_
    
    clusters = {}
    for i, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(data[i])
        else:
            clusters[label] = [data[i]]
    return clusters

def sliding_t_test(time_series_data, window_size=10, significance_level=0.05):
    results = []
    
    for i in range(len(time_series_data) - window_size):
        window_values = [data_point.value for data_point in time_series_data[i:i+window_size]]
        mean_before = np.mean(window_values[:window_size // 2])
        mean_after = np.mean(window_values[window_size // 2:])
        std_dev = np.std(window_values)
        t_statistic = (mean_after - mean_before) / (std_dev / np.sqrt(window_size // 2))
        p_value = 2 * (1 - t.cdf(abs(t_statistic), df=window_size - 1))
        results.append((time_series_data[i + window_size // 2].datetime, t_statistic, p_value))
    
    return results
# endregion
# region scikit-learn使用
def calculate_similarity(ts1, ts2, similarity_metric='euclidean'):
    """
    计算两个时间序列之间的相似性或差异性

    Args:
        ts1 (list or numpy array): 第一个时间序列
        ts2 (list or numpy array): 第二个时间序列
        similarity_metric (str, optional): 相似性度量方法,默认为'euclidean'(欧氏距离)。可选值包括'euclidean'(欧氏距离),
                                            'pearson'(皮尔逊相关系数)。

    Returns:
        float: 两个时间序列之间的相似性或差异性值
    """
    if similarity_metric == 'euclidean':
        # 计算欧氏距离
        similarity = euclidean(ts1, ts2)
    elif similarity_metric == 'pearson':
        # 计算皮尔逊相关系数
        similarity = np.corrcoef(ts1, ts2)[0, 1]
    else:
        raise ValueError("不支持的相似性度量方法")

    return similarity

def time_series_clustering(ts_data, num_clusters):
    # 取出时间序列数据的值
    values = [data.value for data in ts_data]
    
    # 转换为numpy数组
    X = np.array(values).reshape(-1, 1)
    
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    
    # 获取每个时间序列的聚类标签
    labels = kmeans.labels_
    
    # 返回聚类结果
    return labels
# endregion
# region 归一化处理
def min_max_normalization(data):
    """
    使用最小-最大归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据,TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    min_val = min(values)
    max_val = max(values)
    normalized_values = [(val - min_val) / (max_val - min_val) for val in values]
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
    return normalized_data

def standardization(data):
    """
    使用标准化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据,TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    mean_val = np.mean(values)
    std_dev = np.std(values)
    normalized_values = [(val - mean_val) / std_dev for val in values]
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
    return normalized_data

def decimal_scaling_normalization(data):
    """
    使用小数定标归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据,TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    max_abs = max(abs(min(values)), abs(max(values)))

    normalized_data = [
        TimeSeriesData(value / max_abs, entry.datetime)
        for value, entry in zip(values, data)
    ]
    return normalized_data

def log_normalization(data):
    """
    使用对数归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据,TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    min_val = min(values)
    normalized_values = np.log(values) - np.log(min_val)
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
  
    return normalized_data

def l1_normalization(data):
    """
    使用L1范数归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据,TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    l1_norm = np.sum(np.abs(values))

    normalized_values = [value / l1_norm for value in values]
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
  
    return normalized_data

def l2_normalization(data):
    """
    使用L2范数归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据,TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    l2_norm = np.linalg.norm(values)

    normalized_values = [value / l2_norm for value in values]
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
  
    return normalized_data
# endregion
# region 模态分解
def hilbert_transform(time_series):
    """对时间序列进行希尔伯特变换
    amplitude_envelope, instantaneous_phase = hilbert_transform(data)
    process_and_plot_hilbert_transform(amplitude_envelope, instantaneous_phase)
    """
    values = [data.value for data in time_series]
    analytic_signal = hilbert(values)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    return amplitude_envelope, instantaneous_phase

def process_and_plot_hilbert_transform(amplitude_envelope, instantaneous_phase, SaveFilePath=None):
    """处理和可视化希尔伯特变换的结果"""
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # 可视化振幅包络
    ax1.plot(amplitude_envelope)
    ax1.set_title("振幅包络")
    ax1.set_xlabel("时间")
    ax1.set_ylabel("振幅")

    # 可视化瞬时相位
    ax2.plot(instantaneous_phase)
    ax2.set_title("瞬时相位")
    ax2.set_xlabel("时间")
    ax2.set_ylabel("相位")

    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()

def empirical_mode_decomposition(time_series_data):
    """对时序数据进行经验模态分解
    imfs = empirical_mode_decomposition(data)
    """
    values = np.array([data.value for data in time_series_data])
    
    # 创建EMD对象,并进行分解
    emd = EMD()
    imfs = emd.emd(values)
    
    return imfs

def plot_imfs(imfs, SaveFilePath=None):
    """绘制IMFs"""
    num_imfs = len(imfs)
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    # 创建子图布局
    fig, axes = plt.subplots(num_imfs, 1, figsize=(8, 2*num_imfs), sharex=True)

    # 绘制每个IMF的图形
    for i, imf in enumerate(imfs):
        axes[i].plot(imf)
        axes[i].set_ylabel(f"IMF {i+1}")

    # 设置横坐标标签
    axes[-1].set_xlabel("Time")

    # 调整子图之间的间距
    plt.tight_layout()
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    # 显示图形
    else: 
        plt.show()
    plt.close()

def plot_imfs_rm(imfs, rm, SaveFilePath=None):
    """绘制IMFs和残差项Rm"""
    num_imfs = len(imfs)
    num_rows = num_imfs // 2 + num_imfs % 2
    
    # 创建子图布局
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 3*num_rows), sharex=True)

    # 绘制每个IMF的图形
    for i, imf in enumerate(imfs):
        row_idx = i // 2
        col_idx = i % 2
        axes[row_idx, col_idx].plot(imf)
        axes[row_idx, col_idx].set_ylabel(f"IMF {i+1}")

    # 绘制残差项的图形
    rm_row_idx = num_imfs // 2
    rm_col_idx = 0 if num_imfs % 2 == 0 else 1
    axes[rm_row_idx, rm_col_idx].plot(rm)
    axes[rm_row_idx, rm_col_idx].set_ylabel("Rm")

    # 清除未使用的子图
    for i in range(rm_row_idx+1, num_rows):
        for j in range(2):
            axes[i, j].axis("off")

    # 设置横坐标标签
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")

    # 调整子图之间的间距
    plt.tight_layout()
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    # 显示图形
    else: 
        plt.show()
    plt.close()


def eemd(time_series_data, num_trials=100):
    """实现EEMD"""
    values = np.array([data.value for data in time_series_data])
    eemd = EEMD()
    eemd.trials = num_trials
    imfs = eemd.eemd(values)
    
    rm = values - np.sum(imfs, axis=0)
    return imfs,rm

def ceemd(time_series_data, num_trials=100):
    """实现CEEMD"""
    values = np.array([data.value for data in time_series_data])
    ceemdan = CEEMDAN(trials=num_trials)
    imfs = ceemdan.ceemdan(values)

    rm = values - np.sum(imfs, axis=0)
    return imfs,rm

def calculate_variance(imfs):
    """计算每个IMF的方差"""
    variances = []
    for imf in imfs:
        variance = np.var(imf)
        variances.append(variance)
    return variances

def calculate_energy_ratio(imfs):
    """计算每个IMF的能量比"""
    total_energy = np.sum(np.square(imfs))
    energy_ratios = np.square(imfs) / total_energy
    return energy_ratios

def calculate_snr(variances):
    """
    负值的信噪比(如 IMF 1-7)表示信号的能量比噪声的能量要小。负值越小,信号与噪声的能量差异越大,表明信号更清晰。例如,IMF 5 的信噪比为 -8.07 dB,表示信号能量远小于噪声能量,可能是纯粹的噪声成分。
    正值的信噪比(如 IMF 8-9)表示信号的能量比噪声的能量要大。正值越大,信号与噪声的能量差异越大,表明信号更明显。例如,IMF 8 的信噪比为 5.56 dB,表示信号能量远大于噪声能量,可能是较为明显的信号成分。
    信噪比值的大小可以用来判断信号与噪声的相对强度。较小的信噪比值通常表示噪声较为明显,而较大的信噪比则表示信号较为明显。
    根据信噪比值的差异,可以初步判断每个 IMF 中是否包含噪声。例如,IMF 5 的信噪比最小,可能是主要由噪声组成的成分。而 IMF 8 和 IMF 9 的信噪比相对较大,可能是包含较为明显信号的成分。
    """
    snr_values = []
    for variance in variances:
        noise_variance = np.mean(variances)  # 假设噪声方差为所有 IMF 方差的均值
        snr = 10 * math.log10(variance / noise_variance)
        snr_values.append(snr)
    return snr_values

def plot_imf_properties(variances, SaveFilePath=None):
    """
    用柱状图显示每个IMF的方差和能量比
    """
    num_imfs = len(variances)
    imf_indices = np.arange(num_imfs)
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.bar(imf_indices, variances)
    plt.ylabel("方差")
    plt.xticks(imf_indices, [str(i+1) + "号IMF" for i in range(num_imfs)])
    plt.title("IMF方差")

    # 调整子图之间的间距
    plt.tight_layout()
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    # 显示图形
    else: 
        plt.show()
    plt.close()

def wavelet_packet_analysis(signal, wavelet_name='db4', level=5):
    """
    小波包分析
    # 小波包分析的相关参数
    wavelet_name = 'db4'  # 选择小波基函数
    level = 5  # 分解的层数
    """
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet_name, mode='symmetric')
    wp_level = wp.get_level(level)
    
    spectral_energy = []
    for node in wp_level:
        spectral_energy.append(sum(node.data ** 2))
    
    return spectral_energy

def calculate_peak_frequency(spectral_energy,sample_rate = 1):
    """找到频谱能量列表中的最大值及其对应的索引"""
    # 找到频谱能量列表中的最大值及其对应的索引
    max_energy = max(spectral_energy)
    peak_index = spectral_energy.index(max_energy)

    # 假设频谱范围是 [0 Hz, 100 Hz],将索引转换为对应的频率
    max_frequency = peak_index * (sample_rate / len(spectral_energy))

    return max_frequency

def calculate_peak_energy(spectral_energy):
    """找到频谱能量列表中的最大值"""
    # 找到频谱能量列表中的最大值
    peak_energy = max(spectral_energy)

    return peak_energy

def plot_spectral_energy(spectral_energy, SaveFilePath=None):
    """
    可视化频谱能量
    for i, imf in enumerate(imfs):
        spectral_energy = wavelet_packet_analysis(imf)
        plot_spectral_energy(spectral_energy)
    """
    frequencies = range(len(spectral_energy))  # 假设频率范围是0到N-1
    
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.plot(frequencies, spectral_energy)
    plt.xlabel('频率')
    plt.ylabel('频谱能量')
    plt.title('频谱能量分析')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

def calculate_energy_bandwidth(spectral_energy, threshold):
    """
    计算频谱能量的带宽:
    calculate_energy_bandwidth 函数接受频谱能量列表和能量阈值作为参数,并计算出超过阈值所需的频率带宽。该函数通过累计能量来确定带宽,并在能量超过阈值时停止计算。返回的带宽是超过阈值的频率范围所包含的数据点数量。
    threshold = 0.9  # 设定能量阈值,可以根据需求进行调整
    bandwidth = calculate_energy_bandwidth(spectral_energy, threshold)
    print(f"频谱能量的带宽:{bandwidth}")
    """
    total_energy = sum(spectral_energy)
    target_energy = threshold * total_energy

    cumulative_energy = 0
    bandwidth = 0

    for energy in spectral_energy:
        cumulative_energy += energy
        if cumulative_energy >= target_energy:
            bandwidth += 1
        else:
            break

    return bandwidth

def calculate_energy_ratio(spectral_energy):
    """
    计算频谱能量的能量比值
    计算每个能量值与总能量之间的比值
    energy_ratio = calculate_energy_ratio(spectral_energy)
    """
    total_energy = sum(spectral_energy)
    energy_ratio = []

    for energy in spectral_energy:
        ratio = energy / total_energy
        energy_ratio.append(ratio)

    return energy_ratio
def calculate_spectrum_centroid(spectral_energy):
    """谱心是频谱能量的重心位置"""
    frequencies = range(len(spectral_energy))  # 假设频率范围是0到N-1
    energy_sum = 0
    centroid_sum = 0

    for i in range(len(spectral_energy)):
        energy_sum += spectral_energy[i]
        centroid_sum += frequencies[i] * spectral_energy[i]

    if energy_sum != 0:
        centroid = centroid_sum / energy_sum
    else:
        centroid = None

    return centroid

def calculate_spectrum_width(spectral_energy, threshold=0.5):
    """
    计算频谱的宽度通常涉及到定义何种情况下能量减少到原来的一定百分比
    常见的定义是计算能量减少到原来能量的一半所对应的频率范围,即谱宽度为能量衰减到50%的频率范围
    frequencies = range(len(spectral_energy))  # 假设频率范围是0到N-1
    threshold = 0.5  # 定义能量衰减到原来的一半作为阈值
    width = calculate_spectrum_width(spectral_energy, frequencies, threshold)
    print(f"频谱的宽度:{width}")
    """
    frequencies = range(len(spectral_energy))  # 假设频率范围是0到N-1
    total_energy = sum(spectral_energy)
    target_energy = threshold * total_energy

    left_index = 0
    right_index = len(spectral_energy) - 1

    cumulative_energy = 0
    while cumulative_energy < target_energy and left_index < right_index:
        left_energy = spectral_energy[left_index]
        right_energy = spectral_energy[right_index]

        if cumulative_energy + left_energy <= target_energy:
            cumulative_energy += left_energy
            left_index += 1

        if cumulative_energy + right_energy <= target_energy:
            cumulative_energy += right_energy
            right_index -= 1

    width = frequencies[right_index] - frequencies[left_index]

    return width

def calculate_primary_frequency_range(spectral_energy,  top_n=1):
    """
    计算主要频段
    start_freq, end_freq = calculate_primary_frequency_range(spectral_energy, frequencies, top_n)
    print(f"主要频段:{start_freq}Hz - {end_freq}Hz")
    """
    frequencies = range(len(spectral_energy))  # 假设频率范围是0到N-1
    sorted_indices = sorted(range(len(spectral_energy)), key=lambda i: spectral_energy[i], reverse=True)
    top_indices = sorted_indices[:top_n]
    start_freq = frequencies[top_indices[0]]
    end_freq = frequencies[top_indices[-1]]

    return (start_freq, end_freq)

def butterworth_filter(imfs, rm, order, cutoff_freq):
    """butter worth滤波"""
    filtered_imfs = []
    filtered_rm = None
    
    # 对每个IMF分量进行滤波
    for imf in imfs:
        b, a = signal.butter(order, cutoff_freq, output='ba')
        filtered_imf = signal.lfilter(b, a, imf)
        filtered_imfs.append(filtered_imf)
    
    # 对rm进行滤波
    b, a = signal.butter(order, cutoff_freq, output='ba')
    filtered_rm = signal.lfilter(b, a, rm)
    
    return filtered_imfs, filtered_rm

def reconstruct_signal(filtered_imfs, filtered_rm):
    """还原时域数据"""
    signal_reconstructed = np.sum(filtered_imfs, axis=0) + filtered_rm
    return signal_reconstructed

def plot_reconstruct_signal(signal_reconstructed,SaveFilePath=None):
    # 绘制信号重构结果的时间序列图
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.plot(signal_reconstructed)
    plt.xlabel('时间')
    plt.ylabel('信号值')
    plt.title('信号重构结果')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

# endregion
# region 自回归模型及预测值评估
def ar_model_forecast(TimeSeriesData, lags=1, future_steps=10, SaveFilePath=None):
    """
    自回归模型(Autoregressive Model, AR)
    lags = 1
    future_steps = 10
    future_forecast, mse = ar_model_forecast(time_series_data, lags, future_steps)
    print("未来预测的均方根误差:", mse)
    在AR模型中,lags参数表示使用多少个滞后观测值作为特征来预测当前观测值。它控制了模型的阶数,也就是自回归的程度
    future_steps参数表示未来要预测的步数,即要预测多少个未来观测值。通过使用拟合的AR模型和历史观测值,可以对未来指定步数的观测值进行预测
    """
    # 提取时序数据的观测值
    X = np.array([data.value for data in TimeSeriesData])

    # 拟合AR模型
    model = AutoReg(X, lags=lags)
    model_fit = model.fit()

    # 进行未来预测
    future_forecast = model_fit.forecast(steps=future_steps)

    # 计算均方根误差
    mse = root_mean_squared_error(X[-future_steps:], future_forecast)
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    # 绘制原始数据和预测结果
    plt.plot(X, label='原始数据')
    plt.plot(np.arange(len(X), len(X) + future_steps), future_forecast, label='预测结果')
    plt.xlabel('时间')
    plt.ylabel('观测值')
    plt.legend()
    plt.title('AR模型预测结果')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

    return future_forecast, mse

def ma_model_forecast(TimeSeriesData, q=1, future_steps=10, SaveFilePath=None):
    """
    移动平均模型(Moving Average Model, MA)
    future_forecast, mse = ma_model_forecast(time_series_data, q, future_steps)
    我们使用 ARIMA 类从 statsmodels 库中拟合一个 MA 模型,并使用 order=(0, 0, q) 指定使用 q 阶移动平均模型
    我们使用 ARIMA 类从 statsmodels 库中拟合一个 MA 模型,并使用 order=(0, 0, q) 指定使用 q 阶移动平均模型
    """
    # 提取时序数据的观测值
    X = np.array([data.value for data in TimeSeriesData])
    
    # 拟合MA模型
    model = ARIMA(X, order=(0, 0, q))
    model_fit = model.fit()
    
    # 进行未来预测
    future_forecast = model_fit.forecast(steps=future_steps)
    
    mse = root_mean_squared_error(np.array(X[-future_steps:]), np.array(future_forecast))
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    # 绘制原始数据和预测结果
    plt.plot(X, label='原始数据')
    plt.plot(np.arange(len(X), len(X) + future_steps), future_forecast, label='预测结果')
    plt.xlabel('时间')
    plt.ylabel('观测值')
    plt.legend()
    plt.title('MA模型预测结果')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()
    
    return future_forecast, mse

def arma_model_forecast(TimeSeriesData, p, q, future_steps, SaveFilePath=None):
    """
    自回归移动平均模型(Autoregressive Moving Average Model,ARM)
    并指定合适的p、q以及未来预测的步数future_steps,然后该函数将返回未来预测的结果以及均方根误差。
    """
    # 提取时序数据的观测值
    X = [data.value for data in TimeSeriesData]
    
    # 拟合ARMA模型
    model = ARIMA(X, order=(p, 0, q))
    model_fit = model.fit()
    
    # 进行未来预测
    future_forecast = model_fit.forecast(steps=future_steps)[0]
    
    # 计算均方根误差
    mse = root_mean_squared_error(X[-future_steps:], future_forecast, squared=False)
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    # 绘制原始数据和预测结果
    plt.plot(X, label='原始数据')
    plt.plot(np.arange(len(X), len(X) + future_steps), future_forecast, label='预测结果')
    plt.xlabel('时间')
    plt.ylabel('观测值')
    plt.legend()
    plt.title('ARMA模型预测结果')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else: 
        plt.show()
    plt.close()
    
    return future_forecast, mse

def plot_ACF(TimeSeriesData, SaveFilePath=None):
    """绘制自相关函数(ACF)图表"""
    # 提取时序数据的观测值
    X = [data.value for data in TimeSeriesData]

    # 计算自相关函数(ACF)
    acf = sm.tsa.acf(X, nlags=len(TimeSeriesData))
    # 绘制自相关函数(ACF)图表
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.stem(acf)
    plt.xlabel('滞后阶数')
    plt.ylabel('相关系数')
    plt.title('自相关函数(ACF)')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

def plot_PACF(TimeSeriesData,lags=48, SaveFilePath=None):
    """绘制偏自相关函数(PACF)图表"""
    # 提取时序数据的观测值
    X = [data.value for data in TimeSeriesData]

    # 计算偏自相关函数(PACF)
    pacf = sm.tsa.pacf(X, nlags=lags)
    # 绘制偏自相关函数(PACF)图表
    plt.figure(figsize=(25.6, 14.4))  # 单位是英寸
    plt.stem(pacf)
    plt.xlabel('滞后阶数')
    plt.ylabel('相关系数')
    plt.title('偏自相关函数(PACF)')
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    else:
        plt.show()
    plt.close()

def evaluate_arma_model(data):
    # 提取时序数据的观测值
    X = [point.value for point in data]

    # 选择一系列可能的模型阶数
    p_values = range(1, 5)  # 自回归阶数
    q_values = range(1, 5)  # 移动平均阶数

    # 用网格搜索方式寻找最佳的ARMA模型
    best_aic = np.inf
    best_params = None

    for p in p_values:
        for q in q_values:
            try:
                model = sm.tsa.ARMA(X, order=(p, q))
                results = model.fit()
                aic = results.aic
                bic = results.bic
                if aic < best_aic:
                    best_aic = aic
                    best_bic = bic
                    best_params = (p, q)
            except:
                continue

    if best_params is not None:
        print("最佳模型的参数:p={}, q={}".format(best_params[0], best_params[1]))
        print("最佳模型的AIC值:{}".format(best_aic))
        print("最佳模型的BIC值:{}".format(best_bic))
    else:
        print("未找到最佳模型")

def fit_arima_model(data, p=1, d=1, q=1, num_steps=10,output_file="arima_results.txt"):
    """
    差分自回归移动平均模型(ARIMA)
    该函数接受一个时序数据作为输入(例如TimeSeriesData实例的列表),并设置ARIMA模型的阶数(p、d、q)以及预测步数(num_steps)。
    它使用此时序数据来执行ARIMA模型的拟合,并打印出模型的统计摘要和预测的未来数据点。
    """

    X = [point.value for point in data]

    # 创建ARIMA模型对象
    model = sm.tsa.ARIMA(X, order=(p, d, q))

    # 拟合ARIMA模型
    results = model.fit()

    # 将结果写入txt文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(results.summary().as_text())

        # 预测未来的数据点
        forecast = results.forecast(steps=num_steps)
        file.write("\n\n预测结果:\n")
        file.write(str(forecast))

    return forecast

def evaluate_forecast(TimeSeriesData, predicted_values):
    """
    评估预测结果的准确度
    输入参数:
    - actual_values: 实际观测值的数组或列表
    - predicted_values: 预测值的数组或列表
    返回值:
    - rmse: 均方根误差(RMSE)
    - mae: 平均绝对误差(MAE)

    # 评估预测结果
    rmse, mae, correlation, r2 = evaluate_forecast(data, predicted_values)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Correlation:", correlation)
    print("R^2:", r2)
    """
    # 将输入转换为numpy数组
    predicted_values = np.array(predicted_values)
    # 提取实际观测值
    actual_values = np.array([data.value for data in TimeSeriesData])
    actual_values = [point.value for point in TimeSeriesData[-len(predicted_values):]]


    # 均方根误差(RMSE)
    rmse = np.sqrt(root_mean_squared_error(actual_values, predicted_values))

    # 平均绝对误差(MAE)
    mae = mean_absolute_error(actual_values, predicted_values)

    # 相关系数
    correlation = np.corrcoef(actual_values, predicted_values)[0, 1]

    # 决定系数
    r2 = r2_score(actual_values, predicted_values)

    return rmse, mae, correlation, r2
# endregion
# region 数据去噪
def smooth_noise(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        end_index = min(len(data), i + 1)

        window_values = [data[j] for j in range(start_index, end_index)]
        smoothed_value = sum(window_values) / len(window_values)
        smoothed_data.append(smoothed_value)

    return smoothed_data

def equal_depth_binning(data, num_bins):
    sorted_data = sorted(data, key=lambda x: x.value)
    bin_size = len(data) // num_bins

    bins = []
    start_index = 0

    for i in range(num_bins - 1):
        end_index = start_index + bin_size
        bin_data = sorted_data[start_index:end_index]
        bins.append(bin_data)
        start_index = end_index

    # Add remaining data to the last bin
    last_bin = sorted_data[start_index:]
    bins.append(last_bin)

    return bins

def mean_smoothing(bins):
    smoothed_bins = []
    for bin_data in bins:
        bin_values = [data.value for data in bin_data]
        mean_value = sum(bin_values) / len(bin_values)
        smoothed_bin = [TimeSeriesData(mean_value, data.datetime) for data in bin_data]
        smoothed_bins.append(smoothed_bin)

    return smoothed_bins

# 定义可视化函数
def plot_smoothed_data_and_bins(data, smoothed_data, bins, smoothed_bins):
    # 绘制平滑噪声的结果
    plt.figure(figsize=(10, 5))
    plt.plot([data_point.datetime for data_point in data], smoothed_data, label="平滑数据")
    plt.xlabel("时间")
    plt.ylabel("数值")
    plt.title("平滑噪声")
    plt.legend()
    plt.show()

    # 绘制等深分箱和均值平滑的结果
    plt.figure(figsize=(10, 5))
    for i, bin_data in enumerate(smoothed_bins):
        bin_values = [data_point.value for data_point in bin_data]
        bin_datetimes = [data_point.datetime for data_point in bin_data]
        plt.plot(bin_datetimes, bin_values, label=f"箱子 {i+1}")

    plt.xlabel("时间")
    plt.ylabel("数值")
    plt.title("等深分箱和均值平滑")
    plt.legend()
    plt.show()

# 平滑噪声—等深分箱—均值平滑
def aequilatus_box_mean(data, bins=3):
    length = len(data)
    labels = []
    
    for i in range(bins):
        labels.append('a' + str(i+1)) # 添加标签
    
    data_df = pd.DataFrame({'value': [data_point.value for data_point in data]})
    new_data = pd.qcut(data_df['value'], bins, labels=labels) # 等深分箱
    data_df['label'] = new_data

    for label in labels:
        label_index_min = data_df[data_df.label==label].index.min() # 分箱后索引最小值
        label_index_max = data_df[data_df.label==label].index.max() # 分箱后索引最大值
        mean_value = np.mean([data_point.value for data_point in data[label_index_min:label_index_max+1]]) # 计算各箱均值
        for i in range(label_index_min, label_index_max+1):
            data[i].value = mean_value # 修改各数据点的数值为均值

    return data

def wavelet_denoising(data):
    """小波转换去噪"""
    # 提取数据中的数值部分
    signal_values = [data_point.value for data_point in data]

    # 选择小波基函数和分解级别
    wavelet = 'db4'
    level = pywt.dwt_max_level(len(signal_values), wavelet)

    # 进行小波分解
    coeffs = pywt.wavedec(signal_values, wavelet, level=level)

    # 估计噪声水平并计算阈值
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_values)))

    # 对小波系数进行阈值处理
    coeffs = [pywt.threshold(c, threshold) for c in coeffs]

    # 进行小波重构
    denoised_signal = pywt.waverec(coeffs, wavelet)

    return denoised_signal


def plot_denoised_signal(signal_data, denoised_signal):
    # 从 TimeSeriesData 对象中提取数值部分
    signal = [data_point.value for data_point in signal_data]

    plt.figure(figsize=(10, 5))
    plt.plot(signal, label='原始信号')
    plt.plot(denoised_signal, label='去噪后信号')
    plt.xlabel('时间')
    plt.ylabel('幅值')
    plt.title('时序信号去噪')
    plt.legend()
    plt.show()
# endregion
# # 创建 TimeSeriesData 实例
start_time = time.time()  # 记录程序开始时间
data = load_csv_data("D:\python_proj2\daily_data_1.txt")
# 计算 value 的平均值
mean_value = np.mean([point.value for point in data])
# 减去平均值
for point in data:
    point.value -= mean_value
    point.value = point.value * 1000
accelerations=calculate_acceleration(data)
print(len(data))
print(len(accelerations))
# plot_acceleration(data, accelerations)
complete_trend_line = adaptive_polynomial_fitting(data, 5)
plot_polynomial_trend(data, compare_trend_lines)
show_prcoess_time(start_time, "adaptive_polynomial_fitting")
# plot_TimeSeriesData(isShow=True,TimeSeriesData=data)

# degree=11
# alpha=1.0
# y_pred=fit_Ridge_polynomial_trend(data,11)
# residuals=calculate_trend_line_residuals(data,y_pred)
# plot_residuals(residuals)
# durbin_watson_statistic=calculate_durbin_watson(residuals)
# print("Durbin-Watson statistic (德宾沃森统计量):", durbin_watson_statistic)

# # residuals 是残差数据
# lb_test_statistic, p_value = perform_ljung_box_test(residuals)
# # 输出检验统计量和 p 值
# print("Ljung-Box检验统计量:", lb_test_statistic)
# print("P值:", p_value)

# ridge_pred=fit_Ridge_polynomial_trend(data,degree,alpha)
# lasso_pred=fit_Lasso_polynomial_trend(data,degree)
# elastic_pred=fit_ElasticNet_polynomial_trend(data,degree)
# polynomial_pred=fit_polynomial_trend(data,degree)
# euclidean_distances, pearson_correlations, f_statistic, p_value = compare_trend_lines(ridge_pred, lasso_pred, elastic_pred, polynomial_pred)
# print("欧几里得距离：", euclidean_distances)
# print("皮尔逊相关系数：", pearson_correlations)
# print("ANOVA F统计量：", f_statistic)
# print("ANOVA p值：", p_value)
# # 执行滑动T检验
# window_size = 10
# significance_level = 0.05
# results = sliding_t_test(data, window_size, significance_level)

# # 打印检验结果
# for result in results:
#     print("时间:", result[0],"T统计量:", result[1],"P值:", result[2],"显著性结果:", "存在显著的突变" if result[2] < significance_level else "无显著的突变")
