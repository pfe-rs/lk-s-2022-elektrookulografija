# import pygame module in this program
import pygame
import time
 
# activate the pygame library
# initiate pygame and give permission
# to use pygame's functionality.
pygame.init()
 
# define the RGB value for white,
#  green, blue colour .
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
 
# assigning values to X and Y variable
X = 1400
Y = 800
 
# create the display surface object
# of specific dimension..e(X, Y).
display_surface = pygame.display.set_mode((X, Y))
 
# set the pygame window name
pygame.display.set_caption('Show Text')
 
# create a font object.
# 1st parameter is the font file
# which is present in pygame.
# 2nd parameter is size of the font
font = pygame.font.Font('freesansbold.ttf', 32)
 
# create a text surface object,
# on which text is drawn on it.
text2 = font.render("Pogledaj u tacku 1", True, green, blue) 
text3 = font.render('Pogledaj u tacku 2', True, green, blue)
text4 = font.render("Pogledaj u tacku 3", True, green, blue) 
text5 = font.render('Pogledaj u tacku 4', True, green, blue)

tacka1 = font.render('1', True, green,blue)
tacka2 = font.render('2', True, green,blue)
tacka3 = font.render('3', True, green,blue)
tacka4 = font.render('4', True, green,blue)
# create a rectangular object for the
# text surface object
textRect2 = text2.get_rect()
textRect3 = text3.get_rect()
textRect4 = text4.get_rect()
textRect5 = text5.get_rect()

textTacka1 = tacka1.get_rect()
textTacka2 = tacka2.get_rect()
textTacka3 = tacka3.get_rect()
textTacka4 = tacka4.get_rect()
# set the center of the rectangular object.
textRect2.center = (X // 2, Y // 2)
textRect3.center = (X // 2, Y // 2)
textRect4.center = (X // 2, Y // 2)
textRect5.center = (X // 2, Y // 2)

textTacka1.center = (35, Y // 2)
textTacka2.center = (X // 2, 35)
textTacka3.center = (X // 2, Y - 35)
textTacka4.center = (X - 35, Y // 2)

t = 5000
# infinite loop
clock = pygame.time.Clock 
while True:
 
    display_surface.fill((255, 255, 255))
    display_surface.blit(text1, textRect1)

    start = time.time()
    while (1):
        end = time.time()
        display_surface.fill((255, 255, 255))
        display_surface.blit(text1, textRect1)
        timeDiff = end - start

        if(timeDiff > 5):
            
            break
    display_surface.blit(text2, textRect2)
    display_surface.blit(text, textCircle)

    for event in pygame.event.get():
 
        # if event object type is QUIT
        # then quitting the pygame
        # and program both.
        if event.type == pygame.QUIT:
 
            # deactivates the pygame library
            pygame.quit()
 
            # quit the program.
            quit()
 
        # Draws the surface object to the screen.
        pygame.display.update()