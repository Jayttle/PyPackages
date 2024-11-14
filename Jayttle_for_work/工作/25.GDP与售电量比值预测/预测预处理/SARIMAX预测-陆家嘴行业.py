import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import warnings
warnings.filterwarnings("ignore")


def read_and_prepare_data(file_path: str, columns: list):
    df = pd.read_excel(file_path)
    # 读取指定的列
    df = pd.read_excel(file_path, usecols=columns)
    # 确保日期列是 datetime 类型
    if '日期' in df.columns:
        # 提取日期部分并转换为 datetime
        df['日期'] = pd.to_datetime(df['日期'].str[2:], format='%Y%m')
    else:
        raise KeyError("DataFrame 中找不到 '日期' 列")

    # 设置日期列为索引
    df.set_index('日期', inplace=True)
    df.columns = ['总和']  # 修改第二列的列名
    df['总和'] = df['总和'] / 10000 / 10000
    # 划分测试集与验证集
    df_train = df.iloc[:-4]
    df_test = df.iloc[-4:]

    print("训练集样本数：", len(df_train))
    print("测试集样本数：", len(df_test))
    return df_train, df_test

def evaluate_sarima(params, df_train, df_test):
    model = SARIMAX(df_train['总和'], 
                    order=params['order'], 
                    seasonal_order=params['seasonal_order'])
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=len(df_test))
    
    df_test['预测'] = forecast.values
    df_test['误差'] = (df_test['预测'] - df_test['总和']) / df_test['总和']
    
    mse = mean_squared_error(df_test['总和'], df_test['预测'])
    aic = model_fit.aic  # 获取AIC值
    return mse, aic  # 返回均方误差和AIC值

def grid_search_sarima(df_train, df_test):
    best_params = None
    best_mse = np.inf
    best_aic = np.inf  # 初始化最佳AIC

    param_grid = {
        'order': [(p, d, q) for p in range(3) for d in range(3) for q in range(3)],
        'seasonal_order': [(P, D, Q, S) for P in range(3) for D in range(3) for Q in range(3) for S in [12]]
    }
    
    for params in ParameterGrid(param_grid):
        try:
            mse, aic = evaluate_sarima(params, df_train, df_test)
            print(f"测试参数 {params} 的均方误差: {mse}, AIC: {aic}")
            
            if aic < best_aic:  # 使用AIC来选择最佳参数
                best_aic = aic
                best_mse = mse
                best_params = params
        except Exception as e:
            print(f"参数 {params} 出现错误: {e}。跳过这一组参数。")

    if best_params is not None:
        print(f"最佳参数组合: {best_params}")
        print(f"最佳均方误差: {best_mse}")
        print(f"最佳AIC: {best_aic}")
    else:
        print("没有找到合适的参数组合。")

    return best_params
def residual_analysis(model_fit, df_train):
    # 计算残差
    fitted_values = model_fit.fittedvalues.values.flatten()  # 获取数值并确保为一维数组
    residuals = df_train['总和'].values - fitted_values
    residuals = residuals.flatten()  # 确保残差为一维数组
    # 正态性检验
    shapiro_test = stats.shapiro(residuals)
    print('Shapiro-Wilk检验统计量:', shapiro_test.statistic)
    print('p值:', shapiro_test.pvalue)
    if shapiro_test.pvalue > 0.05:
        print("残差近似正态分布")
    else:
        print("残差不符合正态分布")

    # 自相关性检验
    ljung_box_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(ljung_box_test)
    if ljung_box_test['lb_pvalue'].values[0] > 0.05:
        print("残差没有显著自相关")
    else:
        print("残差存在显著自相关")

    # 绘制残差直方图
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True)
    plt.title('残差直方图')
    plt.xlabel('残差')
    plt.ylabel('频率')
    plt.show()

    # 绘制Q-Q图
    sm.qqplot(residuals, line='s')
    plt.title('残差Q-Q图')
    plt.show()

def train_and_predict_sarima(df_train, df_test):
    # 使用网格搜索法找到最佳参数
    # best_params = grid_search_sarima(df_train, df_test)
    
    #     使用最佳参数训练模型
    model = SARIMAX(df_train['总和'], 
                    order=(1, 1, 2), 
                    seasonal_order=(1, 1, 0, 12))
    model_fit = model.fit(disp=False)
    
    # 进行预测
    forecast = model_fit.forecast(steps=len(df_test))
    
    # 将预测结果与实际测试集进行比较
    df_test['预测'] = forecast.values
    df_test['误差'] = (df_test['预测'] - df_test['总和']) / df_test['总和']
    comparison_df = df_test[['总和', '预测', '误差']]
    
    print("\n实际值、预测值和误差对比:")
    print(comparison_df)
    
    # 计算并打印均方误差
    mse = mean_squared_error(df_test['总和'], df_test['预测'])
    print(f"均方误差 (MSE): {mse}")

    
    # 创建结果 DataFrame
    result = pd.DataFrame({
        '总和': comparison_df['总和'],
        '预测': comparison_df['预测'],
        '误差': comparison_df['误差'],
    #     '模型': best_params
    })
    # residual_analysis(model_fit, df_train)
    return result

def predict_sarima(df_train, order, seasonal_order):
    # 使用最佳参数训练模型
    model = SARIMAX(df_train['总和'], 
                    order=order, 
                    seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    # 进行预测
    forecast = model_fit.forecast(steps=6)

    # 创建未来6个月的日期索引
    future_dates = pd.date_range(start=df_train.index[-1] + pd.DateOffset(months=1), periods=len(df_test), freq='M')
    
    # 更新 df_test 的索引
    df_test.index = future_dates

    # 将预测结果添加到 df_test 中
    df_test['预测'] = forecast.values
    print(df_test['预测'])


def calculate_sum_to_save():
    file_path = r"C:\Users\juntaox\Desktop\浦东的分析报告\浦东四大重点区域用户清单-0827(1)(1)(1).xlsx"
    
    # 指定工作表名称
    sheet_names = ['1、小陆家嘴', '3、国际度假区-川沙', '4、东方枢纽-祝桥']
    
    results = {}

    for sheet_name in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        electricity_columns = [col for col in df.columns if col.startswith('电量')]
        
        # 计算每列的和
        sums = df[electricity_columns].sum()
        results[sheet_name] = sums

    # 将结果输出到新的Excel文件
    output_file = r"C:\Users\juntaox\Desktop\电量汇总结果.xlsx"
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=True)

if __name__ == '__main__':
    # 读取数据
    print('---------------------------------------------------------------')
    # 示例调用
    file_path = r"C:\Users\juntaox\Desktop\工作\23.浦东四大重点区域负荷分析\行业预测\陆家嘴行业.xlsx"# 替换为实际的文件路径
    sheet_names = [
#     "二、工业",
    "三、建筑业",
    "四、交通运输、仓储和邮政业",
    "五、信息传输、软件和信息技术服务业",
    "六、批发和零售业",
    "七、住宿和餐饮业",
    "八、金融业",
    "九、房地产业",
    "十、租赁和商务服务业",
    "十一、公共服务及管理组织"
    ]
    results = {}
    
    for name in sheet_names:
        columns_to_read = ['日期', name]  # 替换为你需要的列名
        df_train, df_test = read_and_prepare_data(file_path, columns_to_read)
        results[name] = train_and_predict_sarima(df_train, df_test)

    # 保存结果到同一个 Excel 文件中的不同工作表
    output_file_path = r"C:\Users\juntaox\Desktop\工作\23.浦东四大重点区域负荷分析\行业预测\陆家嘴行业-预测.xlsx"  # 替换为你的输出路径
    with pd.ExcelWriter(output_file_path) as writer:
        for name, result in results.items():
            result.to_excel(writer, sheet_name=name, index=False)

    print(f"结果已保存到 {output_file_path}。")
