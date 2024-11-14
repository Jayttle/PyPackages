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
    
    # 划分测试集与验证集
    df_train = df
    df_test = df[-4:]
    return df_train, df_test


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


def predict_sarima(df_train, df_test, order, seasonal_order):
    # 使用最佳参数训练模型
    model = SARIMAX(df_train['总和'], 
                    order=order, 
                    seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    # 进行预测
    forecast = model_fit.forecast(steps=len(df_test))

    # 创建未来6个月的日期索引
    future_dates = pd.date_range(start=df_train.index[-1] + pd.DateOffset(months=1), periods=len(df_test), freq='M')
    
    # 更新 df_test 的索引
    df_test.index = future_dates

    # 将预测结果添加到 df_test 中
    df_test['预测'] = forecast.values
    df_test['MAPE'] = np.mean(np.abs((df_test['总和'] - df_test['预测']) / df_test['预测'])) * 100
    # 创建结果 DataFrame
    result = pd.DataFrame({
        '预测': df_test['预测'] ,
    })
    # residual_analysis(model_fit, df_train)
    return result

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
    file_path = r"C:\Users\juntaox\Desktop\电量汇总结果.xlsx"  # 替换为实际的文件路径
    # sheet_names = ['1、小陆家嘴', '2、临港新片区', '3、国际度假区-川沙', '4、东方枢纽-祝桥']
    # model_dict ={
    #     '1、小陆家嘴':[(1, 1, 2), (1, 1, 0, 12)],
    #     '2、临港新片区':[(1, 1, 2), (1, 2 , 0, 12)],
    #     '3、国际度假区-川沙':[(0, 1, 2), (1, 1, 1,12)],
    #     '4、东方枢纽-祝桥':[(1, 1, 2), (1, 2 , 0, 12)]
    # }
    sheet_names = ['2、临港新片区', '3、国际度假区-川沙', '4、东方枢纽-祝桥']
    model_dict ={
        '2、临港新片区':[(1, 1, 2), (1, 2 , 0, 12)],
        '3、国际度假区-川沙':[(2, 1, 1), (0, 2, 1, 12)],
        '4、东方枢纽-祝桥':[(1, 0, 3), (0, 2 , 1, 12)]
    }
    results = {}
    
    for name in sheet_names:
        columns_to_read = ['日期', name]  # 替换为你需要的列名
        df_train, df_test = read_and_prepare_data(file_path, columns_to_read)
        results[name] = predict_sarima(df_train, df_test, model_dict[name][0], model_dict[name][1])

    # 保存结果到同一个 Excel 文件中的不同工作表
    output_file_path = r"C:\Users\juntaox\Desktop\未来预测结果-1.xlsx"  # 替换为你的输出路径
    with pd.ExcelWriter(output_file_path) as writer:
        for name, result in results.items():
            result.to_excel(writer, sheet_name=name, index=False)

    print(f"结果已保存到 {output_file_path}。")
