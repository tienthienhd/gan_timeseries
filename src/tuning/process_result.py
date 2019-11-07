import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/tienthien/Desktop/Mine/gan_timeseries/logs/tuning/gru_gan/[2, 1]_tanh_0.0_tanh_gru_after_[2, 1]_tanh_0.5_sigmoid_gru_[2, 1]_[1, 1]_[2, 1]_adam_adam_0.0001_0.0001_3_False_.csv')

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
