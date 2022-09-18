from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#čitanje podataka iz fajla novi_podaci
df = pd.read_csv("novi_podaci.txt")
df = df[30:] 
fig, ax = plt.subplots(3,3)

#eksponencijalnom funkcijom je ukljanjaju šumovi
df['smooth_x'] = df['x_osa'].ewm(alpha = 0.09, adjust=False).mean()
df['smooth_y'] = df['y_osa'].ewm(alpha = 0.05, adjust=False).mean()

#prikazivanje filtriranih podataka, podataka iz fajla
ax[0, 0].plot(df['x_osa'])
ax[0, 1].plot(df['x_tacka'])
ax[0, 2].plot(df['smooth_x'])
ax[0, 0].set_title('X - osa')

ax[1, 0].plot(df['y_osa'])
ax[1, 1].plot(df['y_tacka'])
ax[1, 2].plot(df['smooth_y'])
ax[1, 0].set_title('Y - osa')

ax[2, 0].plot(df['smooth_x'], df['smooth_y'], 'o', color = 'black')

#ubacivanje filtriranih podataka u novi fajl
df = df[["smooth_x", "smooth_y","x_tacka", "y_tacka"]]
df.to_csv("obrajena_baza.txt", index = False)

#izračunavanje polinominalne regresije i prikazivanje dobijenih rezultata
from matrice_parametri import getCalibration
x_model, y_model = getCalibration(3)
x_screen = x_model(df['smooth_x'])
y_screen = y_model(df['smooth_y'])
ax[2, 1].plot(x_screen, y_screen, 'o', color = 'red', markersize=1)

x_screen = 1400 - x_screen
ax[2, 2].plot(x_screen, y_screen, 'o', color = 'red', markersize=1)
ax[2, 2].plot(df['x_tacka'], df['y_tacka'], 'x')
ax[2, 0].set_title('Polinomialna regresija')


fig.set_size_inches(15,10)
plt.show()




