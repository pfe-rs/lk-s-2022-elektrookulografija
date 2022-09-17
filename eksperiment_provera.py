from os import stat, times_result
from turtle import width
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame 
import serial
import time
import math
from eog import frame_work, koordinate

"""
ax = [836.75768617, -9.90113855, 13.87188183]
ay = [-3.44557824e+02, -9.46133751e-02, 3.29555479e+00]
"""

ax = [ 5.09363881e+03, -1.54116705e+01, 3.97037117e-01]
ay = [-3.43612717e+03, 2.82582983e+00, 1.48521313e+01]

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
    #crno-belo      
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #zamutiti
    blure = cv2.GaussianBlur(gray,(29,29),150)
    #binarizacija
    thr = get_threshold(blure, 0.15)
    ret,bin= cv2.threshold(blure,thr,255,cv2.THRESH_BINARY)
    #dilatacija i erozija
    #mask = remove_noise(bin)   
    #keni
    edges = cv2.Canny(bin, 120, 160) # 75, 150
    #krugovi
    rows = frame.shape[0]
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows*7, param1=150, param2=12, minRadius=80, maxRadius=120)
    
    return circles, bin, edges

def koordinate(circles, frame):
    center = (0, 0)
    radius = 1
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1]) #ovde mozda
            # circle center
            cv2.circle(frame, center, 1, (255, 0, 0), 10)

            # circle outline
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 0), 10)
    return center, radius


pygame.init()

white = (255, 255, 255)
red = (200, 0 , 0)
yellow = (90,90,90)
blue = (100, 50, 255)
black = (0, 0, 0)
orange = (255, 165, 0)

window_width = 1000
window_hieght = 500

ww = pygame.display.set_mode((window_width, window_hieght))
pygame.display.set_caption('Gledaj u tacku koja se cveni')

x = [0] * 9
y = [0] * 9
 
x[0] = x[5] = x[6] = 100
x[1] = x[4] = x[7] = window_width // 2
x[2] = x[3] = x[8] = window_width - 100

y[0] = y[1] = y[2] = 100
y[3] = y[4] = y[5] = window_hieght//2
y[6] = y[7] = y[8] = window_hieght - 100

radius1 = 22
radius2 = 66

kamera = 1
cap = cv2.VideoCapture(kamera)
set_color = True

xosa = []
yosa = []
rskup = []

xosa_filter = []
yosa_filter = []
rskup_filter = []

niz_x = [] # ekran
niz_y = [] # ekran

x_predvidjeno = []
y_predvidjeno = []

n = 10
i = 0
k = 0
t = 10000
tacaka = 9

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
    
    ww.fill(black)

    end_time =  pygame.time.get_ticks()

    flag, frame = cap.read()
    circles, bin, edges = frame_work(frame)
    eyePozit, r =  koordinate(circles, frame)

    width, height = bin.shape[:2]
    cv2.line(bin, (height//2, 0), (height//2, width), (0, 0, 255), 5) 
    cv2.line(bin , (0, width//2), (height, width//2), (0, 0, 255), 5)
    
    cv2.imshow('Siva', frame)
    cv2.imshow('Binarizovano', bin)
    cv2.imshow('Keni', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    xosa.append(eyePozit[0])
    yosa.append(eyePozit[1])
    rskup.append(r)

    #milomir's filter  
    if(i<n):
        xosa_filter.append(np.mean(xosa[0:i]))
        yosa_filter.append(np.mean(yosa[0:i]))
        rskup_filter.append(np.mean(rskup[0:i]))
    else:
        xosa_filter.append(np.mean(xosa[i-n:i]))
        rskup_filter.append(np.mean(rskup[i-n:i]))
        c = n + 5
        yosa_filter.append(np.mean(yosa[i-c:i]))

    xp = ax[0] + ax[1] * xosa_filter[i] + ax[2] * yosa_filter[i]
    yp = ay[0] + ay[1] * xosa_filter[i] + ay[2] * yosa_filter[i]

    if xp<=100: xp = 100
    elif xp>=window_width-100: xp = window_width-100

    if yp<=100: yp = 100
    elif yp>=window_hieght-100: yp = window_hieght-100

     
    if(end_time - start_time < t): pygame.draw.circle(ww, white, (xp, yp), 30, height)

    elif(end_time - start_time > t and end_time - start_time < t*2):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 0):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
            x_predvidjeno.append(xp)
            y_predvidjeno.append(yp) 
            pygame.draw.circle(ww, white, (xp, yp), 30, height)

        

    elif(end_time - start_time > t*2 and end_time - start_time < t*3):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 1):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
            x_predvidjeno.append(xp)
            y_predvidjeno.append(yp) 
            pygame.draw.circle(ww, white, (xp, yp), 30, height)


    elif(end_time - start_time > t*3 and end_time - start_time < t*4):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 2):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
            x_predvidjeno.append(xp)
            y_predvidjeno.append(yp) 
            pygame.draw.circle(ww, white, (xp, yp), 30, height)


    elif(end_time - start_time > t*4 and end_time - start_time < t*5):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 3):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
            x_predvidjeno.append(xp)
            y_predvidjeno.append(yp) 
            pygame.draw.circle(ww, white, (xp, yp), 30, height)


    elif(end_time - start_time > t*5 and end_time - start_time < t*6):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 4):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
            x_predvidjeno.append(xp)
            y_predvidjeno.append(yp) 
            pygame.draw.circle(ww, white, (xp, yp), 30, height)


    elif(end_time - start_time > t*6 and end_time - start_time < t*7):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 5):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
            x_predvidjeno.append(xp)
            y_predvidjeno.append(yp) 
            pygame.draw.circle(ww, white, (xp, yp), 30, height)


    elif(end_time - start_time > t*7 and end_time - start_time < t*8):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 6):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
            x_predvidjeno.append(xp)
            y_predvidjeno.append(yp) 
            pygame.draw.circle(ww, white, (xp, yp), 30, height)



    elif(end_time - start_time > t*8 and end_time - start_time < t*9):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 7):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
            x_predvidjeno.append(xp)
            y_predvidjeno.append(yp) 
            pygame.draw.circle(ww, white, (xp, yp), 30, height)



    elif(end_time - start_time > t*9 and end_time - start_time < t*10):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 8):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
            x_predvidjeno.append(xp)
            y_predvidjeno.append(yp) 
            pygame.draw.circle(ww, white, (xp, yp), 30, height)


    
    else: state = False

    i = i + 1


    pygame.display.update()
    clock.tick(30)


        
cap.release()
cv2.destroyAllWindows()

t = np.linspace(0, 90, len(xosa))
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



fp = open("C:\\Users\\EliteBook\\Documents\\lk-s-2022-elektrookulografija\\provera.txt", 'w')
fp.write('x_predvidjeno,y_predvidjeno,x_ekran,y_ekran\n')

minLen = min(len(x_predvidjeno), len(niz_x))
if len(x_predvidjeno) > minLen:
      print("Xosa je duzi za ", len(xosa_filter) - minLen)
      x_predvidjeno = x_predvidjeno[:minLen]
      y_predvidjeno = y_predvidjeno[:minLen]
if len(niz_x) > minLen:
      print("Niz x je duzi za", len(niz_x) - minLen)
      niz_x = niz_x[:minLen]

for i in range(len(x_predvidjeno)):
      if math.isnan(xosa_filter[i]) or math.isnan(yosa_filter[i]):
          continue 
      fp.write(f'{x_predvidjeno[i]},{y_predvidjeno[i]},{niz_x[i]},{niz_y[i]}\n')



pygame.quit()
quit()