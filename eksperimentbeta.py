from os import times_result
from turtle import width
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame 
import serial
import time
import math
from eog import frame_work, koordinate, get_threshold, remove_noise



kamera = 1
cap = cv2.VideoCapture(kamera)
set_color = True
xosa = []
yosa = []
rskup = []

xosa_filter = []
yosa_filter = []
rskup_filter = []

n = 10
i = 0
k = 0

start_veca = time.time()
print("Pocinjemo") 
while True:
    start = time.time()
    while (1):
        flag, frame = cap.read()
        circles, bin, edges = frame_work(frame)
        eyePozit, r =  koordinate(circles, frame)

        width, height = frame.shape[:2]
        cv2.line(frame, (width//2, 0), (width//2, height), (0, 0, 255), 5) 
        cv2.line(frame, (0, height//2), (width, height//2), (0, 0, 255), 5)

        # cv2.imshow('Siva', frame)
        # cv2.imshow('Binarizovano', bin)
        # cv2.imshow('Keni', edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

        i = i + 1
        end = time.time()
        timeDiff = end - start

        if(timeDiff > 5):
            print("Pogledaj u tacku")
            print(k+1)
            k = k + 1
            break
    end_veca = time.time()
    timeDiff_veca = end_veca - start_veca

    if(timeDiff_veca > 45):
        break


#print(xosa)
        
cap.release()
cv2.destroyAllWindows()

t = np.linspace(0, 45, len(xosa))
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

fp = open("C:\\Users\\EliteBook\\Documents\\lk-s-2022-elektrookulografija\\game_baza.txt", 'w')
fp.write('x_osa,y_osa\n')


for i in range(len(xosa_filter)):
     if math.isnan(xosa_filter[i]) or math.isnan(yosa_filter[i]):
         continue 
     fp.write(f'{xosa_filter[i]},{yosa_filter[i]}\n')

