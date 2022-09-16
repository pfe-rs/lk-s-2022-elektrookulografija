from os import times_result
from turtle import width
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame 
import serial
import time
import math

import control
from scipy.optimize import differential_evolution


def remove_noise(img):

    kernel1 = np.ones((10, 10), np.uint8)
    kernel2 = np.ones((8, 8),np.uint8)

    iterations = 1
    img1 = img.copy()

    img1 = cv2.erode(img1, kernel1, iterations)
    img1 = cv2.dilate(img1, kernel2, iterations)
    
    return img1

def get_threshold(frame, p):
    image_hist = np.histogram(frame.flatten(), 256)[0]
    image_cum_hist = np.cumsum(image_hist)
    image_cum_hist = image_cum_hist / image_cum_hist[-1]
    image_cum_hist = (image_cum_hist > p).astype(int)
    thr = np.argmax(np.diff(image_cum_hist))
    return thr

def frame_work(frame):
    #crno-belo      
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #zamutiti
    blure = cv2.GaussianBlur(gray,(15,15),0)
    #binarizacija
    thr = get_threshold(blure, 0.15)
    ret,bin= cv2.threshold(blure,thr,255,cv2.THRESH_BINARY)
    #dilatacija i erozija
    mask = remove_noise(bin)   
    #keni
    edges = cv2.Canny(mask, 120, 160) # 75, 150
    #krugovi
    rows = frame.shape[0]
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows*3, param1=150, param2=12, minRadius=50, maxRadius=100)
    
    return circles, bin, edges

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


kamera = 0
cap = cv2.VideoCapture(kamera)
set_color = True
xosa = []
yosa = []
rskup = []

xosa_filter = []
yosa_filter = []
rskup_filter = []

n = 10
i = 0
k = 0

start_veca = time.time()
print("Pocinjemo") 
while True:
    start = time.time()
    while (1):
        flag, frame = cap.read()
        circles, bin, edges = frame_work(frame)
        eyePozit, r =  koordinate(circles, frame)

        width, height = frame.shape[:2]
        cv2.line(frame, (width//2, 0), (width//2, height), (0, 0, 255), 5) 
        cv2.line(frame, (0, height//2), (width, height//2), (0, 0, 255), 5)

        cv2.imshow('Siva', frame)
        cv2.imshow('Binarizovano', bin)
        cv2.imshow('Keni', edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        xosa.append(eyePozit[0])
        yosa.append(eyePozit[1])
        rskup.append(r)

        
        if(i<n):
            xosa_filter.append(np.mean(xosa[0:i]))
            yosa_filter.append(np.mean(yosa[0:i]))
            rskup_filter.append(np.mean(rskup[0:i]))
        else:
            xosa_filter.append(np.mean(xosa[i-n:i]))
            rskup_filter.append(np.mean(rskup[i-n:i]))
            c = n + 5
            yosa_filter.append(np.mean(yosa[i-c:i]))

        i = i + 1
        end = time.time()
        timeDiff = end - start

        if(timeDiff > 5):
            print("Pogledaj u tacku")
            print(k+1)
            k = k + 1
            break
    end_veca = time.time()
    timeDiff_veca = end_veca - start_veca

    if(timeDiff_veca > 45):
        break


#print(xosa)
        
cap.release()
cv2.destroyAllWindows()

t = np.linspace(0, 20, len(xosa))
plt.figure()
plt.plot(t, xosa)
plt.plot(t, xosa_filter)
plt.xlabel('t - axis')
plt.ylabel('x - axis')
plt.show()

plt.figure()
plt.plot(t, yosa)
plt.plot(t, yosa_filter)
plt.xlabel('t - axis')
plt.ylabel('y - axis')
plt.show()

plt.figure()
plt.plot(t, rskup)
plt.plot(t, rskup_filter)
plt.xlabel('t - axis')
plt.ylabel('r - axis')
plt.show()