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
    center = (0, 0)
    radius = 1
    if circles is not None:
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


start_veca = time.time()
print("Pocinjemo") 
while True:
    start = time.time()
    while (1):
        flag, frame = cap.read()
        circles = frame_work(frame)
        eyePozit, r =  koordinate(circles, frame)
        xosa.append(eyePozit[0])
        yosa.append(eyePozit[1])
        rskup.append(r)
        end = time.time()
        timeDiff = end - start

        if(timeDiff > 5):
            print("Pogledaj u sledecu tacku")
            break
    end_veca = time.time()
    timeDiff_veca = end_veca - start_veca

    if(timeDiff_veca > 20):
        break

#print(xosa)
        
cap.release()
cv2.destroyAllWindows()

t = np.linspace(0, 20, len(xosa))
plt.figure()
plt.plot(t, xosa)
plt.xlabel('t - axis')
plt.ylabel('x - axis')
plt.show()

plt.figure()
plt.plot(t, yosa)
plt.xlabel('t - axis')
plt.ylabel('y - axis')
plt.show()

plt.figure()
plt.plot(t, rskup)
plt.xlabel('t - axis')
plt.ylabel('r - axis')
plt.show()