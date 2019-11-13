import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/wc98/wc98_workload_5min.csv", usecols=[1], header=0)
df.plot()

plt.show()
