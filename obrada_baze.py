import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("novi_podaci.txt")
df = df[20:]
fig, ax = plt.subplots(2,3)

df['smooth_x'] = df['x_osa'].ewm(alpha = 0.07, adjust=False).mean()
df['smooth_y'] = df['y_osa'].ewm(alpha = 0.05, adjust=False).mean()

ax[0, 0].plot(df['x_osa'])
ax[0, 1].plot(df['x_tacka'])
ax[0, 2].plot(df['smooth_x'])
ax[1, 0].plot(df['y_osa'])
ax[1, 1].plot(df['y_tacka'])
ax[1, 2].plot(df['smooth_y'])
fig.set_size_inches(15,10)
plt.show()

df = df[["smooth_x", "smooth_y","x_tacka", "y_tacka"]]
df.to_csv("obradjena_baza.txt", index = False)


