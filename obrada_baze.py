from optparse import Values
from statistics import mean
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
ax[0, 2].plot(df['x_tacka'])
ax[0, 1].plot(df['smooth_x'])
ax[0, 0].set_title('X - osa')
ax[0, 2].set_title('x-koordinata na ekranu')
ax[0, 1].set_title('x - osa filtrirana')

ax[1, 0].plot(df['y_osa'])
ax[1, 2].plot(df['y_tacka'])
ax[1, 1].plot(df['smooth_y'])
ax[1, 0].set_title('Y - osa')
ax[1, 2].set_title('y - koordinata na ekranu')
ax[1, 1].set_title('y - osa filtrirana')

ax[2, 0].plot(df['smooth_x'], df['smooth_y'], 'o', color = 'black')
ax[2, 0].set_title('Koordinate oka na ekranu')

#ubacivanje filtriranih podataka u novi fajl
df = df[["smooth_x", "smooth_y","x_tacka", "y_tacka"]]
df.to_csv("obrajena_baza.txt", index = False)

#izračunavanje polinominalne regresije i prikazivanje dobijenih rezultata
from matrice_parametri import getCalibration
x_model, y_model = getCalibration(3)
x_screen = x_model(df['smooth_x'])
y_screen = y_model(df['smooth_y'])
x_screen = 1400 - x_screen
ax[2, 1].plot(x_screen, y_screen, 'o', color = 'red', markersize=1)
ax[2, 1].plot(df['x_tacka'], df['y_tacka'], 'x')
ax[2, 1].set_title('Polinomialna regresija')

x_hist = [104, 539, 1157, 1180, 601, 139, 205, 720, 1224]
y_hist = [ 233, 123, 91, 327, 450, 567, 705, 701, 624]

x_ekran = [100, 700, 1300, 1300, 700, 100, 100, 700, 1300]
y_ekran = [100, 100, 100, 400, 400, 400, 700, 700, 700]

pomeraj = []

xy_kovarijanse = []

cov_matrix = []
for i in range(0, len(x_hist)):
    points = []
    for j in range(0, len(x_screen)):
        if np.sqrt((x_hist[i] - x_screen[j])**2 + (y_hist[i] - y_screen[j])**2) <= 200:
            points.append([x_screen[j], y_screen[j]])
    
    pomeraj.append(np.sqrt((x_hist[i] - x_ekran[i])**2 + (y_hist[i] - y_ekran[i])**2))
    points = np.array(points)
    cov_matrix.append(np.cov(points.T))
    #xy_kovarijanse.append([np.sqrt(cov_matrix[i][0, 0]), np.sqrt(points[i][1, 1])])

# print(pomeraj)
# print(xy_kovarijanse)


from numpy.linalg import det, inv
dx = 7
dy = 8
mat = np.zeros((1400//dx, 800//dy))
i = 0
for x in range(0, 1400, dx):
    for y in range(0, 800, dy):
        p = 0
        for k in range(0, 9):
            data = np.array([x, y])
            mi = np.array([x_hist[k], y_hist[k]])
            p += 1/(2*np.pi*np.sqrt(np.abs(det(cov_matrix[k]))))*np.exp(-0.5*(data-mi)@inv(cov_matrix[k])@(data-mi).T)  
        mat[(1399-x)//dx, y//dy] = p

x = np.linspace(0, 1400, 1400//dx)
y = np.linspace(0, 800, 800//dy)
X, Y = np.meshgrid(x, y)
ax[2,2].imshow(mat.T, cmap='jet')
ax[2,2].get_xaxis().set_visible(False)
ax[2,2].get_yaxis().set_visible(False)
ax[2, 2].set_title('Centar masa')

fig.set_size_inches(15,10)
plt.show()

#fig = plt.figure()
#ax = fig.gca(projection = '2d')
#jet = plt.get_cmap('jet')
#surf = ax.plot_surface(X, Y, mat, rstride = 1, cstride = 1, cmap = jet, linewidth = 0)

# plt.figure()
# plt.hist2d(x_screen, y_screen, 100)
# plt.show()



# plt.figure()
# plt.hist(x_screen, 100)
# plt.show()

# plt.figure()
# plt.hist(y_screen, 100)
# plt.show()

"""
x_mean = mean(x_screen)
y_mean = mean(y_screen)

std_x = std(x_screen)
std_y = std(y_screen)

distx = dist(x_mean, std_x)
distx = dist(y_mean, std_y)

valuesx = [value for value in ]
"""