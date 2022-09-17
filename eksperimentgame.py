from os import stat, times_result
from turtle import width
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame 
import serial
import time

import math

from eog import *

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
#pravljenje 9 tacaka
#mozda odavde
x = [0] * 9
y = [0] * 9
 #inicijalizacija po redosledu
x[0] = x[5] = x[6] = 100
x[1] = x[4] = x[7] = window_width // 2
x[2] = x[3] = x[8] = window_width - 100

y[0] = y[1] = y[2] = 100
y[3] = y[4] = y[5] = window_hieght//2
y[6] = y[7] = y[8] = window_hieght - 100
#radius manje i vece tacke sa ekrana
radius1 = 22
radius2 = 66
#ukljucivanje kamere
kamera = 1
cap = cv2.VideoCapture(kamera)
set_color = True
#niz za tacke koje mi gledamo
xosa = []
yosa = []
rskup = []
#niz za filtrirane tacke
xosa_filter = []
yosa_filter = []
rskup_filter = []
#niz za
niz_x = []
niz_y = []

n = 10
i = 0
k = 0
#vreme za timer
t = 10000#u ms
tacaka = 9
#timer
clock = pygame.time.Clock() 
state = True #is on
#pocetno i krajnje vreme preko kojih se racuna proteklo vreme
start_time = pygame.time.get_ticks()
end_time = 0

while state:
#if it works->>
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state = False
#refresh-ovanje ekrana
    pygame.display.update()
    clock.tick(30)
    #background
    ww.fill(black)

    end_time =  pygame.time.get_ticks()
    #pozivanje obradjenih slika
    flag, frame = cap.read()
    circles, bin, edges = frame_work(frame)
    #pozicija oka i radius
    eyePozit, r =  koordinate(circles, frame)
    #ekran
    width, height = bin.shape[:2]
    #koord sistem za centriranje zenice
    cv2.line(bin, (height//2, 0), (height//2, width), (0, 0, 255), 5) 
    cv2.line(bin , (0, width//2), (height, width//2), (0, 0, 255), 5)
    
    # cv2.imshow('Siva', frame)
    # cv2.imshow('Binarizovano', bin)
    # cv2.imshow('Keni', edges)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #      break
    #dodavanje u koordinate
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
    #ako je vreme manje od pocetnog
    if(end_time - start_time<t):
        #za svaku tacku
        for j in range (tacaka):
            #crtanje krugova
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 0):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)
        

    elif(end_time - start_time > t and end_time - start_time < t*2):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 1):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)

    elif(end_time - start_time > t*2 and end_time - start_time < t*3):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 2):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)

    elif(end_time - start_time > t*3 and end_time - start_time < t*4):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 3):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)

    elif(end_time - start_time > t*4 and end_time - start_time < t*5):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 4):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)

    elif(end_time - start_time > t*5 and end_time - start_time < t*6):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 5):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)

    elif(end_time - start_time > t*6 and end_time - start_time < t*7):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 6):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)


    elif(end_time - start_time > t*7 and end_time - start_time < t*8):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 7):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)


    elif(end_time - start_time > t*8 and end_time - start_time < t*9):
        for j in range (tacaka):
            pygame.draw.circle(ww, yellow, (x[j],y[j]), radius2)
            if(j == 8):
                pygame.draw.circle(ww, red, (x[j],y[j]), radius1)
                niz_x.append(x[j])
                niz_y.append(y[j])
            else: pygame.draw.circle(ww, blue, (x[j],y[j]), radius1)

    
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


#upisivanje u datoteku
fp = open("C:\\Users\\EliteBook\\Documents\\lk-s-2022-elektrookulografija\\novi_podaci.txt", 'w')
fp.write('x_osa,y_osa,x_tacka,y_tacka\n')

#provera i uporedjivanje duzine niza, i skracivanje po potrebi
minLen = min(len(xosa_filter), len(niz_x))
if len(xosa_filter) > minLen:
     print("Xosa je duzi za ", len(xosa_filter) - minLen)
     xosa_filter = xosa_filter[:minLen]
     yosa_filter = yosa_filter[:minLen]
if len(niz_x) > minLen:
     print("Niz x je duzi za", len(niz_x) - minLen)
     niz_x = niz_x[:minLen]

#upisivanje u niz
for i in range(len(xosa_filter)):
     if math.isnan(xosa_filter[i]) or math.isnan(yosa_filter[i]):
         continue 
     fp.write(f'{xosa_filter[i]},{yosa_filter[i]},{niz_x[i]},{niz_y[i]}\n')



pygame.quit()
quit()