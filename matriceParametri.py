from ctypes.wintypes import DOUBLE
import math
from re import T
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv


podaci = []
with open("plot.txt", 'r') as f:
     next(f)
     for myline in f:
         arr = myline.split(',')
         for i in range(len(arr) - 1):
             arr[i] = float(arr[i])

         arr[-1] = float(arr[-1][:-1])
         podaci.append(arr)

n = 3
m = len(podaci)

x_predvidjeno = []
y_predvidjeno = []
x_poznato = []
y_poznato = []

radiusV = 66
br = 0
for i in range (m): 
    x_predvidjeno[i] = podaci[:, 0]
    y_predvidjeno[i]= podaci[:, 1]
    x_poznato[i]= podaci[:, 2]
    y_poznato[i]= podaci[:, 3]

    r1 = math.sqrt(math.pow(x_predvidjeno[i]-x_poznato[i])+math.pow(y_predvidjeno[i]-y_poznato[i]))
    if(radiusV>r1):
        br += br

procenat = m/br
print(procenat)


