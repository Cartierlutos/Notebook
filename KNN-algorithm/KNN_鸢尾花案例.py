from sklearn.datasets import load_iris  # 导入数据
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import StandardScaler  # 特征预处理，标准化
from sklearn.neighbors import KNeighborsClassifier  # 机器学习KNN
from sklearn.model_selection import GridSearchCV  # 网格搜索

# 获取数据
iris = load_iris()

# 数据基本处理, 更改随机数种子，正确率会不同
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

# 特征工程， 特征预处理
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 机器学习
# 实例化一个估计器
estimator = KNeighborsClassifier()  # 需要手动调节的参数统称为超参数

# 模型调优——交叉验证，网格搜索
param_grid = {'n_neighbors': [1, 3, 5, 7]}
estimator = GridSearchCV(estimator, param_grid=param_grid, cv=5)

# 模型训练
estimator.fit(x_train, y_train)
# 模型评估
y_pre = estimator.predict(x_test)
print('预测值是：\n', y_pre)
print('预测值和真实值的对比是：\n', y_pre == y_test)  # True表示预测正确， False表示预测错误

# 准确率计算
score = estimator.score(x_test, y_test)
print('准确率是：', score)

# 查看交叉验证，网格搜索的一些属性
print('在交叉验证中，得到的最好结果是：\n', estimator.best_score_)
print('在交叉验证中，得到的最好的模型是：\n', estimator.best_estimator_)
print('在交叉验证中，得到的模型结果是：\n', estimator.cv_results_)