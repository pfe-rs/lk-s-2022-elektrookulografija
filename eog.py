from turtle import shape, width
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

import serial
import time

import control
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import differential_evolution

def remove_noise(img):
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((4, 4),np.uint8)
    iterations = 1
    img1 = img.copy()
    img1 = cv2.erode(img1, kernel1, iterations)
    img1 = cv2.dilate(img1, kernel2, iterations)
    
    return img1



def frame_work(frame):
               
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = frame.copy()
    mask = cv2.GaussianBlur(mask,(15,15),0)
    blure = mask.copy()

    ret,mask= cv2.threshold(frame,70,255,cv2.THRESH_BINARY)#menjala
    bin = mask.copy()
    mask = remove_noise(mask)   
    
    return frame, mask, bin, blure


kamera = 1
cap = cv2.VideoCapture(kamera)
set_color = True

#width, height, t = frame.shape

while cap.isOpened():
    
    ret, frame = cap.read()
    rgb = frame.copy()

    
    frame, mask,  bin, blure = frame_work(frame)
    # erozija i dilatacije 

    edges = cv2.Canny(mask, 120, 160) # 75, 150
    
    rows = frame.shape[0]
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows, param1=150, param2=12, minRadius=50, maxRadius=100)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(rgb, center, 1, (255, 0, 0), 10)
            cv2.circle(edges, center, 1, (255, 0, 0), 10)

            # circle outline
            radius = i[2]
            cv2.circle(rgb, center, radius, (255, 0, 0), 10)
            cv2.circle(edges, center, radius, (255, 0, 0), 10)
    
    cv2.imshow("Detected circles", mask)
    cv2.imshow('Siva', frame)
    cv2.imshow('RGB+circle', rgb)
    cv2.imshow('Binarizovano', bin)
    cv2.imshow('Keni', edges)
    cv2.imshow('Mutna', blure)
    
   # cv2.imshow('Nadjena boja', indikator)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()