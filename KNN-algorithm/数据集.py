# coding:utf-8
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split

# 数据集获取
iris = load_iris()
print(iris)

# 大数据集获取
# news = fetch_20newsgroups()
# print(news)

# 数据集属性描述
# print('数据集特征是：\n', iris.data)
# print('数据集目标值是：\n', iris['target'])
# print('数据集特征值是：\n', iris.feature_names)
# print('数据集目标值名字是：\n', iris.target_names)
# print('数据集的描述是：\n', iris.DESCR)

# 数据集可视化
# iris_data = pd.DataFrame(data=iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
# iris_data['target'] = iris.target
# print(iris_data)
#

# 图形绘制
# def iris_plot(data, col1, col2):
#     sns.lmplot(x=col1, y=col2, data=data, hue='target', fit_reg=False)
#     plt.title('鸢尾花数据显示')
#     plt.show()


# iris_plot(iris_data, 'Sepal_Width', 'Sepal_Length')
# iris_plot(iris_data, 'Sepal_Length', 'Sepal_Width')


# 数据集的划分
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
print('训练集的特征值：\n', x_train)
print('训练集的目标值：\n', y_train)
print('测试集的特征值：\n', x_test)
print('测试集的特征值：\n', y_test)

print('训练集目标值形状是：\n', y_train.shape)
print('测试集目标值形状是：\n', y_test.shape)

