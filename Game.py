import pygame
from Graphics import *
from Environment import *
from State import *

pygame.init()

clock = pygame.time.Clock()
graphics = Graphics()
state = State()
env = Environment(state=state)

FPS = 60

def main ():
    run = True

    while (run):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               run = False
               
        graphics.draw(state=state)
        pygame.display.update()
    

if __name__ == '__main__':
    main()