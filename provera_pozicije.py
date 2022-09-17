from ctypes.wintypes import DOUBLE
import math
from re import T
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv


podaci = []
with open("provera.txt", 'r') as f:
     next(f)
     for myline in f:
         arr = myline.split(',')
         for i in range(len(arr) - 1):
             arr[i] = float(arr[i])

         arr[-1] = float(arr[-1][:-1])
         podaci.append(arr)

n = 3
m = len(podaci)

x_predvidjeno = np.zeros(m)
y_predvidjeno = np.zeros(m)
x_poznato = np.zeros(m)
y_poznato = np.zeros(m)

radiusV = 66
br = 0
for i in range (m): 
    x_predvidjeno[i] = podaci[i][0]
    y_predvidjeno[i]= podaci[i][1]
    x_poznato[i]= podaci[i][2]
    y_poznato[i]= podaci[i][3]

    r1 = (x_predvidjeno[i]-x_poznato[i])**2 + (y_predvidjeno[i]-y_poznato[i])**2
    if(radiusV**2>=r1):
        br += 1

plt.figure()
plt.subplot(121)
plt.plot( y_predvidjeno, x_predvidjeno, 'o',color = 'black')
plt.xlabel('x_predvidjeno')
plt.ylabel('y_predvidjeno')
plt.subplot(122)
plt.plot( y_poznato, x_poznato, 'o',color = 'black')
plt.xlabel('x_poznato')
plt.ylabel('y_poznato')
plt.show()

print(br/m)


