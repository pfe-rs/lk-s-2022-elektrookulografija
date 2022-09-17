from ctypes.wintypes import DOUBLE
import math
from re import T
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv

#funkcija za izračunavanje polinominalne i linearne regresije
def getCalibration(N):

    #uzimanje podataka iz fajla
    podaci = []
    with open("obradjena_baza.txt", 'r') as f:
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
    pogledx = np.zeros((m,1))
    pogledy = np.zeros((m,1))

    #inicijalizacija matrica sa podacima
    for i in range (m): 
        zenica[i][0] = 1
        zenica[i][1] = podaci[i][0]
        zenica[i][2] = podaci[i][1]
        pogledx[i][0] = podaci[i][2]
        pogledy[i][0] = podaci[i][3]


    #transponovanje matrice zenice
    t_zenica = np.transpose(zenica)

    #mnozenje matrica
    mnozenje = np.matmul(t_zenica, zenica)
    invertovana = inv(mnozenje)
    inv_mno = np.matmul(invertovana, t_zenica)

    #iyracunavanje parametara linearne regresije
    ax = np.matmul(inv_mno, pogledx)
    ay = np.matmul(inv_mno, pogledy)

    #izračunavanje parametara polinominalne regresije
    Xmodel = np.poly1d(np.polyfit(zenica[:,1], pogledx[:,0], N))
    Ymodel = np.poly1d(np.polyfit(zenica[:,2], pogledy[:,0], N))

    return Xmodel, Ymodel