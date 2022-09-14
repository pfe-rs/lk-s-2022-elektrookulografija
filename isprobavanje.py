# import pygame module in this program
import pygame
import time
import math
 
# activate the pygame library
# initiate pygame and give permission
# to use pygame's functionality.
pygame.init()
 
# define the RGB value for white,
#  green, blue colour .
white = (255, 255, 255)
black = (0, 0, 0)
red = (200, 0 , 0)
green = (0, 255, 0)
blue = (0, 0, 128)
 
# assigning values to X and Y variable
window_width = 1400
window_hieght = 800
 
# create the display surface object
# of specific dimension..e(X, Y).
ww = pygame.display.set_mode((window_width, window_hieght))
 
# set the pygame window name
pygame.display.set_caption('Gledaj u tacku koja se cveni')

#krugovi, tacke
x1 = 40; y1 = 40; radius = 20
x2 = window_width//2; y2 = 40; radius = 20
x3 = window_width-40; y3 = 40; radius = 20
x4 = 40; y4 = window_hieght//2; radius = 20
x5 = window_width//2; y5 = window_hieght//2; radius = 20
x6 = window_width-40; y6 = window_hieght//2; radius = 20
x7 = 40; y7 = window_hieght-40; radius = 20
x8 = window_width//2; y8 = window_hieght-40; radius = 20
x9 = window_width-40; y9 = window_hieght-40; radius = 20



# infinite loop
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

    if(end_time - start_time<3000):
        pygame.draw.circle(ww, red, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)

    elif(end_time - start_time > 3000 and end_time - start_time < 6000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, red, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)

    elif(end_time - start_time > 6000 and end_time - start_time < 9000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, red, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)

    elif(end_time - start_time > 9000 and end_time - start_time < 12000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, red, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)

    elif(end_time - start_time > 12000 and end_time - start_time < 15000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, red, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)

    elif(end_time - start_time > 15000 and end_time - start_time < 18000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, red, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)

    elif(end_time - start_time > 18000 and end_time - start_time < 21000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, red, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)

    elif(end_time - start_time > 21000 and end_time - start_time < 24000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, red, (x8,y8), radius)
        pygame.draw.circle(ww, white, (x9,y9), radius)

    elif(end_time - start_time > 24000 and end_time - start_time < 27000):
        pygame.draw.circle(ww, white, (x1,y1), radius)
        pygame.draw.circle(ww, white, (x2,y2), radius)
        pygame.draw.circle(ww, white, (x3,y3), radius)
        pygame.draw.circle(ww, white, (x4,y4), radius)
        pygame.draw.circle(ww, white, (x5,y5), radius)
        pygame.draw.circle(ww, white, (x6,y6), radius)
        pygame.draw.circle(ww, white, (x7,y7), radius)
        pygame.draw.circle(ww, white, (x8,y8), radius)
        pygame.draw.circle(ww, red, (x9,y9), radius)
    
    else: state = False

    pygame.display.update()
    clock.tick(30)


pygame.quit()
quit()