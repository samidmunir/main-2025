import pygame
import os
######################################################################
WIDTH, HEIGHT = 900, 500
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Invaders")
FPS = 60
VELOCITY = 5
SPACESHIP_WIDTH, SPACESHIP_HEIGHT = 55, 40

BORDER = pygame.Rect((WIDTH / 2 - 5, 0, 10, HEIGHT))

YELLOW_SPACESHIP_IMAGE = pygame.image.load(os.path.join('assets', 'spaceship_yellow.png'))
YELLOW_SPACESHIP = pygame.transform.rotate(pygame.transform.scale(YELLOW_SPACESHIP_IMAGE, (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)), 90)

RED_SPACESHIP_IMAGE = pygame.image.load(os.path.join('assets', 'spaceship_red.png'))
RED_SPACESHIP = pygame.transform.rotate(pygame.transform.scale(RED_SPACESHIP_IMAGE, (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)), 270)
######################################################################
def draw_window(red, yellow):
    WINDOW.fill('#262626')
    pygame.draw.rect(WINDOW, '#000000', BORDER)
    
    # Use .blit() when to draw a surface onto the WINDOW.
    WINDOW.blit(YELLOW_SPACESHIP, (yellow.x, yellow.y))
    WINDOW.blit(RED_SPACESHIP, (red.x, red.y))
    
    # Use .display.update() to update the display after drawing.
    pygame.display.update()

def handle_yellow_movement(keys_pressed, yellow):
    if (keys_pressed[pygame.K_LEFT]) and yellow.x - VELOCITY - 10 > BORDER.x: # left
        yellow.x -= VELOCITY
    if (keys_pressed[pygame.K_RIGHT]) and yellow.x + VELOCITY + yellow.height < WIDTH: # right
        yellow.x += VELOCITY
    if (keys_pressed[pygame.K_UP]) and yellow.y  - VELOCITY > 0: # up
        yellow.y -= VELOCITY
    if (keys_pressed[pygame.K_DOWN]) and yellow.y + VELOCITY + yellow.width < HEIGHT: # down
        yellow.y += VELOCITY

def handle_red_movement(keys_pressed, red):
    if keys_pressed[pygame.K_a] and red.x - VELOCITY > 0: # left
        red.x -= VELOCITY
    if keys_pressed[pygame.K_d] and red.x + VELOCITY + red.height < BORDER.x: # right
        red.x += VELOCITY
    if keys_pressed[pygame.K_w] and red.y - VELOCITY > 0: # up
        red.y -= VELOCITY
    if keys_pressed[pygame.K_s] and red.y + VELOCITY + red.width < HEIGHT: # down
        red.y += VELOCITY

def main():
    red = pygame.Rect(100, 300, SPACESHIP_WIDTH, SPACESHIP_HEIGHT)
    yellow = pygame.Rect(700, 300, SPACESHIP_WIDTH, SPACESHIP_HEIGHT)
    
    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Handle keyboard events.
        keys_pressed = pygame.key.get_pressed()
        handle_yellow_movement(keys_pressed, yellow)
        handle_red_movement(keys_pressed, red)
        
        # Call the draw_window function to handle drawing.
        draw_window(red, yellow)
    #     
    pygame.quit()
    
if __name__ == '__main__':
    main()