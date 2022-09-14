import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math

import serial
import time

import control
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import differential_evolution

kamera = 0
cap = cv2.VideoCapture(kamera)
set_color = True
scaling_factor = 0.7


while cap.isOpened():

    ret, img = cap.read()
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh_gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        x, y, width, height = rect
        radius = 0.25 * (width + height)

        area_condition = (100 <= area <= 200)
        symmetry_condition = (abs(1 - float(width)/float(height)) <= 0.2)
        fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.3)

        if area_condition and symmetry_condition and fill_condition:
            cv2.circle(img, (int(x + radius), int(y + radius)), int(1.3*radius), (0,180,0), -1)

    cv2.imshow('Pupil Detector', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()