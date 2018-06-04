# -*- coding: utf-8 -*-

"""
    文件名:    config.py
    功能：     配置文件

    
"""
import os

# 数据集路径
data_file = './data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv'

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 数据的统计值列
stats_cols = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']

# 原始数据的标签列
raw_label_col = 'Weighted_Price'

# 用于模型训练的标签列
label_col = 'Label_Price'

# 开始预测年份
year_start_pred = 2017

# 模型存放路径
model_file = os.path.join(output_path, 'trained_lstm_model.h5')

# LSTM模型参数
timestep = 1
nodes = 4
batch_size = 1
nb_epoch = 10
