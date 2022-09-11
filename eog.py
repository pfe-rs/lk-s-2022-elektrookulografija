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
    kernel1 = np.ones((30, 30), np.uint8)
    kernel2 = np.ones((50,50),np.uint8)
    iterations = 1
    img1 = img.copy()
    img1 = cv2.erode(img1, kernel1, iterations)
    img1 = cv2.dilate(img1, kernel2, iterations)
    
    return img1

def find_the_centre(mask):

    x, y = np.where(mask == 255)
    n = len(x)
    if n == 0:
        n = 1
    x_final = int(sum(x) / n)
    y_final = int(sum(y) / n)
    
    return x_final, y_final


""""
def binarizacija(frame):
    treshold1 = 151
    treshold2 = 99
    width = frame.shape[0] 
    hight = frame.shape[1]
    for i in range (width): 
        for j in range(hight): 
            if frame[i,j] >= treshold1 and frame[i,j] <= treshold2: 
                bin[i, j] = 0 
            else: bin [i, j] = 0
    return bin
"""


def frame_work(frame):
               
    #rotiranje,
    #binaroizacija na osnovu narandzaste boje
    #i nalazenje centra 
    
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    #orange_hsv = [20, 200, 200]
    
    indikator = frame.copy()
    indikator[:, :] = [0, 0, 255]
    
    treshold = 50


    #mask = cv2.adaptiveThreshold(frame, treshold)
    #mask = cv2.inRange(frame, (30, 100, 58), (30, 255, 255))  

    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = frame
    #frame = cv2.GaussianBlur(frame,(5,5),0)
    #frame = cv2.medianBlur(frame,5)
    #ret, mask = cv2.threshold(frame,0,255,cv2.THRESH_BINARY)

    ret,frame = cv2.threshold(frame,40,255,cv2.THRESH_BINARY)

    frame = remove_noise(frame)
    
    x, y = find_the_centre(frame)
    
    #frame[x-5 : x+5, y-5 : y+5, :] = [255, 0, 255]
            
    position = [x, y]    
    
    return frame, mask, position

def fix_pos(pos, frame):
    x = pos[1]
    y = frame.shape[0] - pos[0]
    return x, y


kamera = 1
cap = cv2.VideoCapture(kamera)
set_color = True

while cap.isOpened():
    
    ret, frame = cap.read()
    
    frame, mask, position = frame_work(frame)
    
    cv2.imshow('Webcam', frame)
    cv2.imshow('Filtrirano', mask)
    
   # cv2.imshow('Nadjena boja', indikator)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()