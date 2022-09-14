from os import stat, times_result
from turtle import width
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame 
import serial
import time
import math

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
window_hieght = 800

ww = pygame.display.set_mode((window_width, window_hieght))
pygame.display.set_caption('Gledaj u tacku koja se cveni')

x1 = 40; y1 = 40; radius = 20
x2 = window_width//2; y2 = 40; radius = 20
x3 = window_width-40; y3 = 40; radius = 20
x4 = 40; y4 = window_hieght//2; radius = 20
x5 = window_width//2; y5 = window_hieght//2; radius = 20
x6 = window_width-40; y6 = window_hieght//2; radius = 20
x7 = 40; y7 = window_hieght-40; radius = 20
x8 = window_width//2; y8 = window_hieght-40; radius = 20
x9 = window_width-40; y9 = window_hieght-40; radius = 20

kamera = 1
cap = cv2.VideoCapture(kamera)
set_color = True

xosa = []
yosa = []
rskup = []

xosa_filter = []
yosa_filter = []
rskup_filter = []

niz_x = []
niz_y = []

n = 10
i = 0
k = 0

clock = pygame.time.Clock() 
state = True

start_time = pygame.time.get_ticks()
end_time = 0

while state:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state = False

    pygame.display.update()
    clock.tick(30)
    
    ww.fill(white)

    end_time =  pygame.time.get_ticks()

    flag, frame = cap.read()
    circles, bin, edges = frame_work(frame)
    eyePozit, r =  koordinate(circles, frame)
    '''cv2.imshow('Siva', frame)
    cv2.imshow('Binarizovano', bin)
    cv2.imshow('Keni', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break'''

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

    if(end_time - start_time<5000):
        pygame.draw.circle(ww, red, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)
        niz_x.append(x1)
        niz_y.append(y1)

    elif(end_time - start_time > 5000 and end_time - start_time < 10000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, red, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)
        niz_x.append(x2)
        niz_y.append(y2)

    elif(end_time - start_time > 10000 and end_time - start_time < 15000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, red, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)
        niz_x.append(x3)
        niz_y.append(y3)

    elif(end_time - start_time > 15000 and end_time - start_time < 20000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, red, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)
        niz_x.append(x4)
        niz_y.append(y4)

    elif(end_time - start_time > 20000 and end_time - start_time < 25000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, red, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)
        niz_x.append(x5)
        niz_y.append(y5)

    elif(end_time - start_time > 25000 and end_time - start_time < 30000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, red, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)
        niz_x.append(x6)
        niz_y.append(y6)

    elif(end_time - start_time > 30000 and end_time - start_time < 35000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, red, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)
        niz_x.append(x7)
        niz_y.append(y7)

    elif(end_time - start_time > 35000 and end_time - start_time < 40000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, red, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)
        niz_x.append(x8)
        niz_y.append(y8)

    elif(end_time - start_time > 40000 and end_time - start_time < 45000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, red, (x9,y9), radius)
        niz_x.append(x9)
        niz_y.append(y9)
    
    else: state = False

    i = i + 1


    pygame.display.update()
    clock.tick(30)


        
cap.release()
cv2.destroyAllWindows()

t = np.linspace(0, 27, len(xosa))
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

fp = open("C:\\Users\\EliteBook\\OneDrive\\Radna površina\\eogcode\\eog_podaci.txt", 'w')
fp.write('x_osa,y_osa,x_tacka,y_tacka\n')

minLen = min(len(xosa_filter), len(niz_x))
if len(xosa_filter) > minLen:
    print("Xosa je duzi za ", len(xosa_filter) - minLen)
    xosa_filter = xosa_filter[:minLen]
    yosa_filter = yosa_filter[:minLen]
if len(niz_x) > minLen:
    print("Niz x je duzi za", len(niz_x) - minLen)
    niz_x = niz_x[:minLen]

for i in range(len(xosa_filter)):
    if math.isnan(xosa_filter[i]) or math.isnan(yosa_filter[i]):
        continue 
    fp.write(f'{xosa_filter[i]},{yosa_filter[i]},{niz_x[i]},{niz_y[i]}\n')



pygame.quit()
quit()