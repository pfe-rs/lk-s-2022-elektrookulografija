from ctypes.wintypes import DOUBLE
import math
from re import T
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv


podaci = []
with open("C:\\Users\\EliteBook\\Documents\\lk-s-2022-elektrookulografija\\eog_podaci.txt", 'r') as f:
     next(f)
     for myline in f:
         arr = myline.split(',')
         for i in range(len(arr) - 1):
             arr[i] = float(arr[i])

         arr[-1] = float(arr[-1][:-1])
         podaci.append(arr)


#print(podaci)

n = 3
m = len(podaci)

zenica = np.zeros((m,n))

for i in range (m): 
    zenica[i][0] = 1
    zenica[i][1] = podaci[i][0]
    zenica[i][2] = podaci[i][1]

pogledx = np.zeros((m,1))
pogledy = np.zeros((m,1))

for i in range (m):
     pogledx[i][0] = podaci[i][2]
     pogledy[i][0] = podaci[i][3]

t_zenica = np.transpose(zenica)


mnozenje = np.dot(t_zenica, zenica)
invertovana = inv(mnozenje)
inv_mno = np.dot(invertovana, t_zenica)

ax = np.dot(inv_mno, pogledx)
ay = np.dot(inv_mno, pogledy)

print(ax)
print(ay)