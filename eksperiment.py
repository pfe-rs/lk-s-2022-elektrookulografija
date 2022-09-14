from os import times_result
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame 
import serial
import time

import control
from scipy.optimize import differential_evolution

def remove_noise(img):

    kernel1 = np.ones((9, 9), np.uint8)
    kernel2 = np.ones((11, 11),np.uint8)

    iterations = 1
    img1 = img.copy()

    img1 = cv2.erode(img1, kernel1, iterations)
    img1 = cv2.dilate(img1, kernel2, iterations)
    
    return img1


def frame_work(frame):

    #crno-belo      
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #zamutiti
    blure = cv2.GaussianBlur(gray,(15,15),0)
    #binarizacija
    ret,bin= cv2.threshold(blure,70,255,cv2.THRESH_BINARY)
    #dilatacija i erozija
    mask = remove_noise(bin)   
    #keni
    edges = cv2.Canny(mask, 120, 160) # 75, 150
    #krugovi
    rows = frame.shape[0]
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows, param1=150, param2=12, minRadius=50, maxRadius=100)
    
    return circles

def koordinate(circles, frame):
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(frame, center, 1, (255, 0, 0), 10)

        # circle outline
        radius = i[2]
        cv2.circle(frame, center, radius, (255, 0, 0), 10)
    return center, radius


kamera = 1
cap = cv2.VideoCapture(kamera)
set_color = True
xosa = []
yosa = []
rskup = []

timet = 0 #vreme provedeno na jednoj tacki
timeu = 0 #ukupno t
radi = True

print("Pogledaj u prvu tacku")

while radi:
    timet = time.time()
    while(timet<=5):
        flag, frame = cap.read()
        circles, bin, edges = frame_work(frame)
        cv2.imshow('Siva', frame)
        cv2.imshow('Binarizovano', bin)
        cv2.imshow('Keni', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        xy, radius = koordinate(circles, frame) #pozicija zenice
        cv2.imshow('RGB+circle', frame)
        xosa.append(xy[0])
        yosa.append(xy[1])
        rskup.append(radius)
    timet = 0
    timeu = timeu + timet
    print("Pogledaj u sledecu tacku")
    if(timeu > 25): radi = False    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()