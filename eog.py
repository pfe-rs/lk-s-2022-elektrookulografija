import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import differential_evolution

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
    thr = get_threshold(blure, 0.13)
    ret,bin= cv2.threshold(blure,thr,255,cv2.THRESH_BINARY)
    #dilatacija i erozija
    #mask = remove_noise(bin)   
    #keni
    edges = cv2.Canny(bin, 120, 160) # 75, 150
    #krugovi
    rows = frame.shape[0]
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows*3, param1=150, param2=12, minRadius=50, maxRadius=100)
    
    return frame,  bin, blure, edges, circles

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

#width, height, t = frame.shape

while cap.isOpened():
    
    ret, frame = cap.read()
    rgb = frame.copy()

    
    frame,  bin, blure, edges, circles = frame_work(frame)
    # erozija i dilatacije 
    
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
    
    height, width = blure.shape[:2]
    cv2.line(blure, (width//2, 0), (width//2, height), 50)
    cv2.line(blure, (0, height//2), (width, height//2), 50)

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