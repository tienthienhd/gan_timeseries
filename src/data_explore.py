import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/gg_trace/data_full.csv", usecols=[0, 1, 3, 4], header=None)
df.plot()

plt.show()
