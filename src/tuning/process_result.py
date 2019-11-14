import os

import pandas as pd
import numpy as np

from metrics import evaluate
from src.tuning.function import plot_distribution, plot_predict
import seaborn as sns
import matplotlib.pyplot as plt



df_result = pd.read_csv('/home/tienthien/Desktop/Mine/gan_timeseries/result/tuning/bayes/result_tuning.csv')
for data, group_data in df_result.groupby(by='data'):
    print(data)
    for sub_data, group_sub_data in group_data.groupby(by='sub_data'):
        print("\t\t", sub_data)
        for loss, group_loss in group_sub_data.groupby(by='loss_type'):
            print("\t\t\t\t", loss)
            group_loss = group_loss.sort_values('jsd')
            best = group_loss.iloc[0]
            path_predict = f"/home/tienthien/Desktop/Mine/gan_timeseries/result/tuning/{best['tuning_algo']}/{best['algo']}/{best['loss_type']}/{best['data']}/{best['sub_data']}/{best['filename']}"
            folder_stability = f"/home/tienthien/Desktop/Mine/gan_timeseries/result/stability/{best['tuning_algo']}/{best['algo']}/{best['loss_type']}/{best['data']}/{best['sub_data']}"
            if not os.path.exists(folder_stability):
                os.makedirs(folder_stability, exist_ok=True)

            df_predict = pd.read_csv(path_predict)

            actual = df_predict.iloc[:, 0].values
            predicts = df_predict.iloc[:, 1:].values
            predict_mean = np.mean(predicts, axis=1)
            predict_std = np.std(predicts, axis=1)
            mean_std = np.mean(predict_std)

            res_eval = evaluate(actual, predict_mean,
                                metrics=('mae', 'smape', "jsd"))


            df_predict.to_csv(f"{folder_stability}/{best['filename']}")
            plot_predict(actual, predict_mean, predict_std, title=res_eval, path=f"{folder_stability}/predict_{best['filename']}")
            plot_distribution(actual, predict_mean, title=res_eval, path=f"{folder_stability}/distribution_{best['filename']}")

exit(0)


df = pd.read_csv(
    '/home/tienthien/Desktop/Mine/gan_timeseries/logs/tuning/gru_gan/[2, 1]_tanh_0.0_tanh_gru_after_[2, 1]_tanh_0.5_sigmoid_gru_[2, 1]_[1, 1]_[2, 1]_adam_adam_0.0001_0.0001_3_False_.csv')

pred_min = df.iloc[:, 1:].min(axis=1)
pred_max = df.iloc[:, 1:].max(axis=1)
pred = df.iloc[:, 1:].mean(axis=1)
# pred = pred_max

index_pred = pred > 100

pred[index_pred] = pred_max[index_pred]

df_main = pd.concat([df.iloc[:, 0], pred], axis=1)

# print(np.mean(np.abs(df_main.iloc[:, 0].values - df_main.iloc[:, 1].values)))
start = 1600
end = start + 300

print(np.mean(np.abs(df_main.iloc[start: end, 0].values - df_main.iloc[start: end, 1].values)))
df_main.iloc[start: end].plot()
df_main.iloc[start: end].hist(bins=200)

plt.show()



