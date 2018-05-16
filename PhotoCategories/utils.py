# -*- coding: utf-8 -*-

"""
    
    文件名:    utils.py
    功能：     工具文件，包含
                - 数据加载
                - 图像数据显示
                - 特征工程等
"""
import pandas as pd
import config
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler


def load_fashion_mnist_dataset(data_file):
    """
        根据给定的fashion_mnist数据集文件读取数据

        参数：
            - data_file: 数据集路径
        返回：
            - X: 数据矩阵，[n_samples, img_rows * img_cols]
            - y: 标签
    """
    data_df = pd.read_csv(data_file)
    X = data_df.iloc[:, 1:].values.astype(np.uint8)
    y = data_df.iloc[:, 0].values.astype(np.uint8)

    print('共有{}个图像'.format(X.shape[0]))

    return X, y


def plot_random_samples(X):
    """
        随机选取9张图像数据进行可视化

        参数：
            - X: 数据矩阵，[n_samples, img_rows * img_cols]
    """

    random_X = X[np.random.choice(X.shape[0], 9, replace=False), :]

    for i in range(9):
        img_data = random_X[i, :].reshape(config.img_rows, config.img_cols)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_data, cmap='gray')
        plt.tight_layout()


def extract_feats(X):
    """
        特征提取

        参数：
            - X: 数据矩阵，[n_samples, img_rows * img_cols]

        返回：
            -feat_arr: 特征矩阵
    """
    n_samples = X.shape[0]

    feat_list = []

    for i in range(n_samples):
        img_data = X[i, :].reshape(config.img_rows, config.img_cols)
        # 中值滤波，去除噪声
        blur_img_data = cv2.medianBlur(img_data, 3)

        # 直方图均衡化
        equ_blur_img_data = cv2.equalizeHist(blur_img_data)

        # cv2.imshow('original', img_data)
        # cv2.imshow('blurred', blur_img_data)
        # cv2.imshow('equalized', equ_blur_img_data)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 将图像转换为特征向量
        feat = equ_blur_img_data.flatten()
        feat_list.append(feat)

        if (i + 1) % 5000 == 0:
            print('已完成{}个图像的特征提取。'.format(i + 1))

    feat_arr = np.array(feat_list)
    return feat_arr


def do_feature_engineering(feats_train, feats_test):
    """
        特征处理

        参数：
            - feats_train: 训练数据特征矩阵
            - feats_test:  测试数据特征矩阵

        返回：
            - scaled_feats_train: 处理后的训练数据特征
            - scaled_feats_test: 处理后的测试数据特征
    """
    # 标准化
    std_scaler = StandardScaler()
    scaled_feats_train = std_scaler.fit_transform(feats_train.astype(np.float64))
    scaled_feats_test = std_scaler.transform(feats_test.astype(np.float64))

    return scaled_feats_train, scaled_feats_test
