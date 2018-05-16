# -*- coding: utf-8 -*-

"""
   
    文件名:    config.py
    功能：     配置文件

"""
import os

# 指定数据集路径
dataset_path = './data'

# 训练集路径
train_data_file = os.path.join(dataset_path, 'fashion-mnist_train.csv')

# 测试集路径
test_data_file = os.path.join(dataset_path, 'fashion-mnist_test.csv')

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 图像大小
img_rows, img_cols = 28, 28


