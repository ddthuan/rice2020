import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

headers = ['epoch', 'time']
df = pd.read_csv('time/time_test_3layer.csv',names=headers)

x = df['epoch']

#df['time'] = df['time'].apply(lambda x: np.round(int(x),2))
#df.round({"time":2})
y = df['time']

plt.plot(x,y)
plt.gcf().autofmt_xdate()
plt.show()