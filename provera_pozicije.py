from ctypes.wintypes import DOUBLE
import math
from re import T
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv

#ucitavanje podataka iz .txt file-a
podaci = []
with open("provera.txt", 'r') as f:
     next(f)
     for myline in f:
        #rastavljanje pomocu zareza
         arr = myline.split(',')
         for i in range(len(arr) - 1):
             arr[i] = float(arr[i])

         arr[-1] = float(arr[-1][:-1])
         podaci.append(arr)
#velicina matrice
n = 3
m = len(podaci)
#inicijalizacija nizova
x_predvidjeno = np.zeros(m)
y_predvidjeno = np.zeros(m)
x_poznato = np.zeros(m)
y_poznato = np.zeros(m)
#radius veceg kruga (sa ekrana, radi poredjenja)
radiusV = 100
#brojac za odredjivaje preciznosti
br = 0
for i in range (m): 
    #izdvajenje kolona iz .txt fajla u nizove za podatke za predvidjanje i poznate podatke
    x_predvidjeno[i] = podaci[i][0]
    y_predvidjeno[i]= podaci[i][1]
    x_poznato[i]= podaci[i][2]
    y_poznato[i]= podaci[i][3]
    #formula za duzinu duzi
    r1 = (x_predvidjeno[i]-x_poznato[i])**2 + (y_predvidjeno[i]-y_poznato[i])**2
    #ispitivanje da li nas pogled dodiruje ili sece ciljani krug na ekranu
    if(radiusV**2>=r1):
        br += 1
#plotovanje
plt.figure()
plt.subplot(121)
plt.plot(y_predvidjeno, x_predvidjeno)
plt.xlabel('x_predvidjeno')
plt.ylabel('y_predvidjeno')
plt.subplot(122)
plt.plot(y_poznato, x_poznato)
plt.xlabel('x_poznato')
plt.ylabel('y_poznato')
plt.show()
#preciznost
print(br/m)

