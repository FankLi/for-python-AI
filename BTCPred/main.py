# -*- coding: utf-8 -*-

"""
    日期:     2018/03
    文件名:    main.py
    功能：     主程序

    比特币价格分析
    任务：搭建深度神经网络LSTM，使用比特币的历史价格数据对其未来价格进行预测。

    数据集来源： https://www.kaggle.com/mczielinski/bitcoin-historical-data

"""

from keras.models import load_model
import os
import pandas as pd
import matplotlib.pyplot as plt

import config
import utils

# 是否加载已经训练好的模型
IS_LOAD_MODEL = True


def main():
    """
        主函数
    """
    # 加载数据集
    all_daily_df = utils.load_data()

    # 对时序数据进行差分操作
    stationary_df = utils.make_stationary_seq(all_daily_df, vis=False)

    # 构建可用于模型的数据
    train_data, test_data = utils.make_data_for_model(stationary_df)

    # 归一化数据
    y_scaler, X_train, y_train, X_test, y_test = utils.scale_data(train_data, test_data)

    if not IS_LOAD_MODEL:
        # 训练LSTM模型
        lstm_model = utils.fit_lstm(X_train, y_train)
    else:
        # 加载LSTM模型
        if os.path.exists(config.model_file):
            lstm_model = load_model(config.model_file)
        else:
            print('{}模型文件不存在'.format(config.model_file))
            return

    # 验证模型
    test_dates = test_data.index.tolist()
    pred_daily_df = pd.DataFrame(columns=['True Value', 'Pred Value'], index=test_dates)
    pred_daily_df['True Value'] = all_daily_df[config.raw_label_col]

    for i, test_date in enumerate(test_dates):
        X = X_test[i].reshape(1, -1)    # 将一天的数据特征转成行向量
        y_pred = utils.forecast_lstm(lstm_model, X)
        # scale反向操作，恢复数据范围
        rescaled_y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))[0, 0]

        # 差分反向操作，恢复数据的值：加上前一天的真实标签
        previous_date = test_date - pd.DateOffset(days=1)
        recoverd_y_pred = rescaled_y_pred + all_daily_df.loc[previous_date][config.raw_label_col]

        # 保存数据
        pred_daily_df.loc[test_date, 'Pred Value'] = recoverd_y_pred
        print('Date={}, 真实值={}, 预测值={}'.format(test_date,
                                               all_daily_df.loc[test_date][config.raw_label_col],
                                               recoverd_y_pred))

    # 保存结果
    pred_daily_df.to_csv(os.path.join(config.output_path, 'pred_daily_df.csv'))
    pred_daily_df.plot()
    plt.savefig(os.path.join(config.output_path, 'pred_daily_df.png'))
    plt.show()


if __name__ == '__main__':
    main()
