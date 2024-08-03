import pandas as pd

# # 读取两个CSV文件
# df1 = pd.read_csv('./features_with_labels.csv')
# df2 = pd.read_csv('./teatures_with_labels.csv')
#
# # 纵向合并这两个文件
# merged_df = pd.concat([df1, df2])
#
# # 保存合并后的文件到新的CSV文件
# merged_df.to_csv('./data.csv', index=False)


def merge_csv_files():
    print("merge csv files")
    # 定义两个CSV文件的路径
    file_path1 = './features_with_labels.csv'
    file_path2 = './teatures_with_labels.csv'

    # 读取两个CSV文件
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # 纵向合并这两个文件
    merged_df = pd.concat([df1, df2])

    # 保存合并后的文件到新的CSV文件
    merged_df.to_csv('./data.csv', index=False)
    print("Merged CSV files saved as data.csv")