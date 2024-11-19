import json
import pandas as pd

# 读取 JSON 文件并转换为 Excel
def json_to_excel(json_file, excel_file):
    # 读取 JSON 数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建一个空的列表来存储数据
    rows = []
    
    # 遍历每个 key 和它对应的列表
    for category, items in data.items():
        for item in items:
            rows.append([category, item])  # 第一列是 category (键)，第二列是 item (值)
    
    # 创建 DataFrame
    df = pd.DataFrame(rows, columns=['Category', 'Item'])

    return df


def read_fenlei(excel_file):
    df = pd.read_excel(excel_file)

    # 提取 'name1' 和 'name3' 列
    columns_to_extract = ['NAME1', 'NAME2']
    extracted_data = df[columns_to_extract]


    return extracted_data

def run_region_fenlei():
    region_names = ['陆家嘴', '临港片区','东方枢纽-祝桥', '川沙']

    for region in region_names:
        # 调用函数并指定 JSON 文件和输出 Excel 文件路径
        df_origin = json_to_excel(fr"E:\OneDrive\PyPackages\Jayttle_for_work\工作\3.浦东新区四大重点经济圈\{region}行业分类.json", r"E:\OneDrive\PyPackages\Jayttle_for_work\工作\3.浦东新区四大重点经济圈\陆家嘴行业分类.xlsx")
        df_biaozhun = read_fenlei(r"C:\Users\juntaox\Desktop\朗新营销系统行业分类-1.xlsx")

        # 合并 df_origin 和 df_biaozhun，根据 df_origin 中的 'item' 列与 df_biaozhun 中的 'NAME3' 列
        df_merged = pd.merge(df_origin, df_biaozhun, left_on='Item', right_on='NAME2', how='left')

        # 查看合并结果
        print(df_merged.head())

        # 自定义列名重命名字典
        rename_columns = {
            'Category': '行业类别',
            'Item': '行业名称',
            'NAME1': '朗新系统行业类别',
            'NAME2': '朗新系统行业名称'
        }

        # 对 df_merged_with_no_match 的列名进行重命名
        df_merged.rename(columns=rename_columns, inplace=True)

        # 为 df_merged 添加 '产业类别' 列，根据 '行业类别' 进行分类
        def classify_industry(row):
            if '农业' in row['行业类别'] or '一、农业' in row['行业类别']:
                return '第一产业'
            elif '工业' in row['行业类别'] or '二、工业' in row['行业类别']:
                return '第二产业'
            elif '建筑业' in row['行业类别'] or '三、建筑业' in row['行业类别']:
                return '第二产业'
            elif '房地产业' in row['行业类别'] or '九、房地产业' in row['行业类别']:
                return '第二产业'
            else:
                return '第三产业'

        # 使用 apply() 方法为每行应用 classify_industry 函数
        df_merged['产业类别'] = df_merged.apply(classify_industry, axis=1)

        # df_merged Excel 文件中
        df_merged.to_excel(fr"E:\OneDrive\PyPackages\Jayttle_for_work\工作\3.浦东新区四大重点经济圈\{region}行业_检查表.xlsx", index=False)


def run_region_chanye():
    # region_names = ['陆家嘴', '临港片区','东方枢纽-祝桥', '川沙']
    region_names = ['川沙']

    for region in region_names:
        # 调用函数并指定 JSON 文件和输出 Excel 文件路径
        df_origin = json_to_excel(fr"E:\OneDrive\PyPackages\Jayttle_for_work\工作\3.浦东新区四大重点经济圈\{region}行业分类.json", r"E:\OneDrive\PyPackages\Jayttle_for_work\工作\3.浦东新区四大重点经济圈\陆家嘴行业分类.xlsx")
        df_biaozhun = read_fenlei(r"C:\Users\juntaox\Desktop\朗新营销系统行业分类-1.xlsx")

        # 合并 df_origin 和 df_biaozhun，根据 df_origin 中的 'item' 列与 df_biaozhun 中的 'NAME3' 列
        df_merged = pd.merge(df_origin, df_biaozhun, left_on='Item', right_on='NAME2', how='left')


        # 自定义列名重命名字典
        rename_columns = {
            'Category': '行业类别',
            'Item': '行业名称',
            'NAME1': '朗新系统行业类别',
            'NAME2': '朗新系统行业名称'
        }

        # 对 df_merged_with_no_match 的列名进行重命名
        df_merged.rename(columns=rename_columns, inplace=True)


def qingdan_check():
    # region_names = ['陆家嘴', '临港片区','东方枢纽-祝桥', '川沙']
    region_names = ['临港片区']
    qingdan_file = r"C:\Users\juntaox\Desktop\浦东四大重点区域用户清单.xlsx"
    
    qingdan_file = r"C:\Users\juntaox\Desktop\工作\3.浦东新区四大重点经济圈发展分析\临港202001-202408用户电量明细.xlsx"
    # 指定要读取的sheet_name，比如这里假设是 '3、国际度假区-川沙'
    # sheet_name = '4、东方枢纽-祝桥'  # 替换为实际的工作表名称
    
    # 读取指定的工作表
    # df_qingdan = pd.read_excel(qingdan_file, sheet_name=sheet_name)
    df_qingdan = pd.read_excel(qingdan_file)
    
    # 打印读取的表格的前几行（查看数据）
    print(df_qingdan.head())
    
    for region in region_names:
        check_excel = rf'E:\OneDrive\PyPackages\Jayttle_for_work\工作\3.浦东新区四大重点经济圈\{region}行业_检查表.xlsx'
        df_check = pd.read_excel(check_excel)
        
        # 只提取 '行业名称' 和 '产业类别' 列
        df_filtered = df_check[['行业名称', '产业类别']]
        
        # 如果需要将 NaN 值去除，使用 dropna()
        df_filtered = df_filtered.dropna()
        
        # 转换为字典形式，'行业名称' 为键，'产业类别' 为值
        industry_mapping = pd.Series(df_filtered['产业类别'].values, index=df_filtered['行业名称']).to_dict()

        # 假设 df_qingdan 中有一列 '行业分类' 用来查找行业名称
        # 使用 `map()` 方法将 '行业分类' 映射到对应的 '产业类别'
        df_qingdan['对应的产业类别'] = df_qingdan['行业分类'].map(industry_mapping)

        
        # 调整列的顺序，将 '对应的产业类别' 列放到 '行业分类' 列后面
        cols = df_qingdan.columns.tolist()  # 获取所有列名
        idx = cols.index('行业分类')  # 找到 '行业分类' 列的位置
        # 将 '对应的产业类别' 列移到 '行业分类' 后面
        cols.insert(idx + 1, cols.pop(cols.index('对应的产业类别')))
        df_qingdan = df_qingdan[cols]  # 重新按照新的列顺序排列
        
        df_qingdan.to_excel(rf"C:\Users\juntaox\Desktop\{region}清单.xlsx")

if  __name__ == "__main__":
    qingdan_check()

        # # 查看修改后的合并结果
        # print("\n修改后的合并结果（包括没有匹配的行）：")
        # print(df_merged.head())