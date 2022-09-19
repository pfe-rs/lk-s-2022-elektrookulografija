import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import differential_evolution
#f-ja za uklanjanje suma
def remove_noise(img):

    kernel1 = np.ones((17, 17), np.uint8)
    kernel2 = np.ones((9, 9),np.uint8)

    iterations = 1
    img1 = img.copy()
    #erozija i dilatacija
    img1 = cv2.erode(img1, kernel1, iterations)
    img1 = cv2.dilate(img1, kernel2, iterations)
    
    return img1
#dobijanje praga pomocu histograma
def get_threshold(frame, p):
    image_hist = np.histogram(frame.flatten(), 256)[0]
    image_cum_hist = np.cumsum(image_hist)
    image_cum_hist = image_cum_hist / image_cum_hist[-1]
    image_cum_hist = (image_cum_hist > p).astype(int)
    thr = np.argmax(np.diff(image_cum_hist))
    return thr
#f-ja u kojoj se dobija circle zenice, blurovana slika i canny
def frame_work(frame):
    #crno-belo      
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #zamutiti
    blure = cv2.GaussianBlur(gray,(29,29),150)
    #binarizacija
    thr = get_threshold(blure, 0.17)
    #thr = 80
    ret,bin= cv2.threshold(blure,thr,255,cv2.THRESH_BINARY)
    #dilatacija i erozija
    #mask = remove_noise(bin)   
    mask = bin
    #keni
    edges = cv2.Canny(mask, 120, 160) # 75, 150
    #krugovi
    rows = frame.shape[0]#rows*7, minr = 70, maxr = 120
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows*6, param1=150, param2=12, minRadius=45, maxRadius=95) #45 95
    if circles is None: 
        print('nan')
    else:
        print(circles[0][0][2])
    return circles, bin, edges  
#dobijanje koordinata centra i radius
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

#ukljucivanje kamere
kamera = 1
cap = cv2.VideoCapture(kamera)
set_color = True

#width, height, t = frame.shape

while cap.isOpened():
    #citanje videa
    ret, frame = cap.read()
    rgb = frame.copy()
    
    circles, bin,  edges  = frame_work(frame)
    # erozija i dilatacije 
    
    if circles is not None:
        #niz circles pomocu kojeg se zadaje centar
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(rgb, center, 1, (255, 0, 0), 10)

            # circle outline
            radius = i[2]
            cv2.circle(rgb, center, radius, (255, 0, 0), 10)
    
    cv2.imshow('Siva', frame)
    cv2.imshow('RGB+circle', rgb)
    cv2.imshow('Binarizovano', bin)
    cv2.imshow('Keni', edges)
    #closing on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
