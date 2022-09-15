import cv2
import matplotlib.pyplot as plt
import numpy as np


def remove_noise(img):

    kernel1 = np.ones((17, 17), np.uint8)
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
    
    #binarizacija
    #thr = get_threshold(frame, 0.30)
    ret,bin1= cv2.threshold(frame,get_threshold(frame, 0.30),255,cv2.THRESH_BINARY)
    ret,bin2= cv2.threshold(frame,get_threshold(frame, 0.20),255,cv2.THRESH_BINARY)
    ret,bin3= cv2.threshold(frame,get_threshold(frame, 0.09),255,cv2.THRESH_BINARY)
    #dilatacija i erozija
    mask = remove_noise(bin3)   
    #keni
    edges = cv2.Canny(mask, 120, 160) # 75, 150
    #krugovi
   
    return  bin1, bin2, bin3, edges, mask



img = cv2.imread('Screenshot 2022-09-15 193827.png',cv2.IMREAD_GRAYSCALE)
bin1, bin2, bin3, edges, mask = frame_work(img)


cv2.imshow('image',img)
cv2.imshow('mask',mask)
cv2.imshow('bin1',bin1)
cv2.imshow('bin2',bin2)
cv2.imshow('bin3',bin3)

cv2.imshow('edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()