from os import stat, times_result
from tkinter.messagebox import OKCANCEL
from turtle import width
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame 
import serial
import time
import math

#x' = a1 + a2x + a3y
# x - pozicija oka 
# y - pozicija oka 
# a1, a2, a3 - parametri 
# x' - kordinate oka

ax = [2399.41443373, -9.48694785, 3.89732282]
ay = [-4.16886724e+03, 2.40274057e+00, 1.79816618e+01]

def remove_noise(img):

    kernel1 = np.ones((25, 25), np.uint8)
    kernel2 = np.ones((9, 9),np.uint8)

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
    blure = cv2.GaussianBlur(gray,(25,25),150)
    #binarizacija
    thr = get_threshold(blure, 0.20)
    ret,bin= cv2.threshold(blure,thr,255,cv2.THRESH_BINARY)
    #dilatacija i erozija
    #mask = remove_noise(bin)   
    #keni
    edges = cv2.Canny(bin, 120, 160) # 75, 150
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


pygame.init()

white = (255, 255, 255)
red = (200, 0 , 0)

window_width = 1400
window_height = 800

ww = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Gledaj u beli prozor')

xosa = []
yosa = []
xosa_filter = []
yosa_filter = []

n = 10
i = 0

kamera = 1
cap = cv2.VideoCapture(kamera)
set_color = True

clock = pygame.time.Clock() 
state = True

while state:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state = False

    pygame.display.update()
    clock.tick(30)
    
    ww.fill(white)

    flag, frame = cap.read()
    circles, bin, edges = frame_work(frame)
    eyePozit, r =  koordinate(circles, frame)

    cv2.imshow('Siva', frame)
    cv2.imshow('Binarizovano', bin)
    cv2.imshow('Keni', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    xosa.append(eyePozit[0])
    yosa.append(eyePozit[1])

    #milomir's filter  
    if(i<n):
        xosa_filter.append(np.mean(xosa[0:i]))
        yosa_filter.append(np.mean(yosa[0:i]))
    else:
        xosa_filter.append(np.mean(xosa[i-n:i]))
        c = n + 5
        yosa_filter.append(np.mean(yosa[i-c:i]))
    
    x = ax[0] + ax[1] * xosa_filter[i] + ax[2] * yosa_filter[i]
    y = ay[0] + ay[1] * xosa_filter[i] + ay[2] * yosa_filter[i]

    if x<=20: x = 40
    elif x>=window_width: x = window_width-40

    if y<=20: y = 40
    elif y>=window_height: y = window_height-40

    pygame.draw.circle(ww, red, (x, y), radius = 20)
       
    i = i + 1

    pygame.display.update()
    clock.tick(30)

pygame.quit()
quit()
