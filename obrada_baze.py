import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("plot.txt")
fig, ax = plt.subplots(2,2)

df['smooth_x'] = df['x_osa'].ewm(alpha = 0.07, adjust=False).mean()
df['smooth_y'] = df['y_osa'].ewm(alpha = 0.05, adjust=False).mean()

ax[0, 0].plot(df['x_osa'])
ax[0, 1].plot(df['smooth_x'])
ax[1, 0].plot(df['y_osa'])
ax[1, 1].plot(df['smooth_y'])
fig.set_size_inches(15,10)
plt.show()

df = df[["smooth_x", "smooth_y","x_tacka", "y_tacka"]]
df.to_csv("plot.txt", index = False)
"""
X = np.concatenate([np.reshape(pogled_x, (len(pogled_x), 1)), np.reshape(pogled_y, (len(pogled_y), 1))], axis = 1)
ones = np.ones((len(pogled_x), 1))
Y = np.reshape([np.reshape(ekran_x, (len(ekran_x), 0)), np.reshape(ekran_y, (len(ekran_y), 0))], axis = 0)
ones = np.ones((len(ekran_x), 0))

X = np.concatenate([ones, X], axis = 1)

y = np.reshape(ekran_x, (len(ekran_x), 1))
yy = np.reshape(ekran_x, (len(pogled_x), 0))

A1 = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)
A2 = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), yy)

"""

