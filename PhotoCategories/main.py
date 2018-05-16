# -*- coding: utf-8 -*-


import os
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import utils
import config

# True表示使用简单默认的Logistic Regression分类器
# 否则使用多个模型的交叉验证
IS_SIMPLE_EXP = True


def train_model(X_train, y_train, X_test, y_test, model_name, model, param_range):
    """

        根据给定的参数训练模型，并返回
        1. 最优模型
        2. 平均训练耗时
        3. 准确率
    """
    print('训练{}...'.format(model_name))
    clf = GridSearchCV(estimator=model,
                       param_grid=param_range,
                       cv=3,
                       scoring='accuracy',
                       refit=True)
    start = time.time()
    clf.fit(X_train, y_train)
    # 计时
    end = time.time()
    duration = end - start
    print('耗时{:.4f}s'.format(duration))

    # 验证模型
    print('训练准确率：{:.3f}'.format(clf.score(X_train, y_train)))

    score = clf.score(X_test, y_test)
    print('测试准确率：{:.3f}'.format(score))
    print('训练模型耗时: {:.4f}s'.format(duration))
    print()

    return clf, score, duration


def main():
    """
        主函数
    """
    # 加载数据
    print('加载训练数据...')
    X_train, y_train = utils.load_fashion_mnist_dataset(config.train_data_file)

    print('加载测试数据...')
    X_test, y_test = utils.load_fashion_mnist_dataset(config.test_data_file)

    # 随机查看9张图像
    utils.plot_random_samples(X_train)

    # 特征提取
    print('训练数据特征提取...')
    feats_train = utils.extract_feats(X_train)
    print('测试数据特征提取...')
    feats_test = utils.extract_feats(X_test)

    # 特征归一化处理
    proc_feats_train, proc_feats_test = utils.do_feature_engineering(feats_train, feats_test)

    # 数据建模及验证
    print('\n===================== 数据建模及验证 =====================')

    if IS_SIMPLE_EXP:
        # 耗时比较短
        print('简单的Logistic Regression分类：')
        lr = LogisticRegression()
        lr.fit(proc_feats_train, y_train)
        print('测试准确率：{:.3f}'.format(lr.score(proc_feats_test, y_test))) # 结果为：0.8289

    else:
        # 耗时比较长
        print('多个模型交叉验证分类比较：')
        model_name_param_dict = {'kNN': (KNeighborsClassifier(),
                                         {'n_neighbors': [5, 25, 55]}),
                                 'LR': (LogisticRegression(),
                                        {'C': [0.01, 1, 100]}),
                                 'SVM': (SVC(kernel='linear'),
                                         {'C': [0.01, 1, 100]}),
                                 'DT': (DecisionTreeClassifier(),
                                        {'max_depth': [50, 100, 150]}),
                                 'AdaBoost': (AdaBoostClassifier(),
                                              {'n_estimators': [100, 150, 200]}),
                                 'GBDT': (GradientBoostingClassifier(),
                                          {'learning_rate': [0.01, 1, 100]}),
                                 'RF': (RandomForestClassifier(),
                                        {'n_estimators': [100, 150, 200]})}

        # 比较结果的DataFrame
        results_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                                  index=list(model_name_param_dict.keys()))
        results_df.index.name = 'Model'

        for model_name, (model, param_range) in model_name_param_dict.items():
            best_clf, best_acc, mean_duration = train_model(proc_feats_train, y_train, proc_feats_test, y_test,
                                                            model_name, model, param_range)
            results_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
            results_df.loc[model_name, 'Time (s)'] = mean_duration

        results_df.to_csv(os.path.join(config.output_path, 'model_comparison.csv'))

        # 模型及结果比较
        print('\n===================== 模型及结果比较 =====================')

        plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        results_df.plot(y=['Accuracy (%)'], kind='bar', ylim=[50, 100], ax=ax1, title='Accuracy(%)', legend=False)

        ax2 = plt.subplot(1, 2, 2)
        results_df.plot(y=['Time (s)'], kind='bar', ax=ax2, title='Time (s)', legend=False)
        plt.tight_layout()
        plt.savefig(os.path.join(config.output_path, 'pred_results.png'))
        plt.show()


if __name__ == '__main__':
    main()
