import pandas as pd


def filter_df_for_user_type(df: pd.DataFrame, user_type_column: str, user_type_value: str) -> pd.DataFrame:
    """
    筛选出 '用户类型' 列中值为指定类型的行，然后计算指定列的数值总和。
    
    :param df: DataFrame 包含数据
    :param user_type_column: '用户类型' 列名
    :param user_type_value: 需要筛选的用户类型值
    :param column_name: 要计算总和的列名
    :return: 计算后的列总和
    """
    if user_type_column not in df.columns:
        raise ValueError(f"数据中不包含 '{user_type_column}' 列")
    # 筛选出 '用户类型' 列中值为指定类型的行
    filtered_df = df[df[user_type_column] == user_type_value]
    return filtered_df

def filter_df_for_user_type_containstr(df: pd.DataFrame, user_type_column: str, substring: str) -> pd.DataFrame:
    """
    筛选出指定列中包含指定子字符串的行。
    
    :param df: DataFrame 包含数据
    :param user_type_column: 要筛选的列名
    :param substring: 要匹配的子字符串
    :return: 筛选后的 DataFrame
    """
    if user_type_column not in df.columns:
        raise ValueError(f"数据中不包含 '{user_type_column}' 列")
    
    # 筛选出指定列中包含指定子字符串的行
    filtered_df = df[df[user_type_column].str.contains(substring, na=False)]
    return filtered_df


def main() -> None:
    # clear_log(log_file_path)
    print("------------------------------run------------------------------")
    
    file_path = r"C:\Users\juntaox\Desktop\工作待办\浦东月报\临港202001-202408用户电量明细.xlsx"
    
    # 读取数据
    df = pd.read_excel(file_path)

    print(df.shape[0])
    print(df.head())

    
    gaoya_df = filter_df_for_user_type(df, '用户类别', '高压')
    print(gaoya_df.shape[0])
    print(gaoya_df.head())
    
if __name__ == '__main__':
    main()
