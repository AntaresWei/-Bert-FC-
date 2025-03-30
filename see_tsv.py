# 查看数据集tsv格式
import pandas as pd

# 读取 .tsv 文件
file_path = "./chnsenticorp/test.tsv"  # 替换为你的文件路径
data = pd.read_csv(file_path, sep='\t')

# 查看数据样例
print(data.head())  # 打印前 5 行数据

# 查看全部标签种类
labels = data['label']  # 假设标签在第二列
unique_labels = labels.unique()
print("\n查看全部标签种类：", unique_labels)

# 查看数据大小
num_rows, _ = data.shape
print(f"数据规模：{num_rows}")