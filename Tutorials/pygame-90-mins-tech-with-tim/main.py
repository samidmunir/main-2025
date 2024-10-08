import pygame
import os
pygame.font.init()
pygame.mixer.init()
######################################################################
WIDTH, HEIGHT = 1200, 800
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
HEALTH_FONT = pygame.font.SysFont('comicsans', 40)
WINNER_FONT = pygame.font.SysFont('comicsans', 100)
BULLET_HIT_SOUND = pygame.mixer.Sound(os.path.join('assets', 'Grenade+1.mp3'))
BULLET_FIRE_SOUND = pygame.mixer.Sound(os.path.join('assets', 'Gun+Silencer.mp3'))
pygame.display.set_caption("Space Invaders")
FPS = 60
VELOCITY = 5
BULLET_VELOCITY = 10
MAX_BULLETS = 10

# New USEREVENTS for bullet collision detection.
YELLOW_HIT = pygame.USEREVENT + 1
RED_HIT = pygame.USEREVENT + 2

SPACESHIP_WIDTH, SPACESHIP_HEIGHT = 55, 40
BORDER = pygame.Rect((WIDTH // 2 - 5, 0, 10, HEIGHT))

YELLOW_SPACESHIP_IMAGE = pygame.image.load(os.path.join('assets', 'spaceship_yellow.png'))
YELLOW_SPACESHIP = pygame.transform.rotate(pygame.transform.scale(YELLOW_SPACESHIP_IMAGE, (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)), 90)

RED_SPACESHIP_IMAGE = pygame.image.load(os.path.join('assets', 'spaceship_red.png'))
RED_SPACESHIP = pygame.transform.rotate(pygame.transform.scale(RED_SPACESHIP_IMAGE, (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)), 270)
######################################################################
def draw_window(red, yellow, red_bullets, yellow_bullets, red_health, yellow_health):
    WINDOW.fill('#262626')
    pygame.draw.rect(WINDOW, '#000000', BORDER)
    
    # Draw health/score text.
    red_health_text = HEALTH_FONT.render('Health: ' + str(red_health), 1, '#ffffff')
    yellow_health_text = HEALTH_FONT.render('Health: ' + str(yellow_health), 1, '#ffffff')
    WINDOW.blit(red_health_text, (10, 10))
    WINDOW.blit(yellow_health_text, (WIDTH - yellow_health_text.get_width() - 10, 10))
    
    # Use .blit() when to draw a surface onto the WINDOW.
    WINDOW.blit(YELLOW_SPACESHIP, (yellow.x, yellow.y))
    WINDOW.blit(RED_SPACESHIP, (red.x, red.y))
    
    # Draw bullets.
    for bullet in red_bullets:
        pygame.draw.rect(WINDOW, '#ff0000', bullet)
    for bullet in yellow_bullets:
        pygame.draw.rect(WINDOW, '#00ffff', bullet)
    
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
        
def handle_bullets(yellow_bullets, red_bullets, yellow, red):
    for bullet in yellow_bullets:
        bullet.x -= BULLET_VELOCITY
        if red.colliderect(bullet):
            pygame.event.post(pygame.event.Event(RED_HIT))
            yellow_bullets.remove(bullet)
        elif bullet.x < 0:
            yellow_bullets.remove(bullet)
    
    for bullet in red_bullets:
        bullet.x += BULLET_VELOCITY
        if yellow.colliderect(bullet):
            pygame.event.post(pygame.event.Event(YELLOW_HIT))
            red_bullets.remove(bullet)
        elif bullet.x > WIDTH:
            red_bullets.remove(bullet)

def draw_winner(text):
    draw_text = WINNER_FONT.render(text, 1, '#ffffff')
    WINDOW.blit(draw_text, (WIDTH // 2 - draw_text.get_width() // 2, HEIGHT // 2 - draw_text.get_height() // 2))
    pygame.display.update()
    pygame.time.delay(5000)

def main():
    red = pygame.Rect(100, 300, SPACESHIP_WIDTH, SPACESHIP_HEIGHT)
    yellow = pygame.Rect(700, 300, SPACESHIP_WIDTH, SPACESHIP_HEIGHT)
    
    red_health = 10
    yellow_health = 10
    
    yellow_bullets = []
    red_bullets = [] 
    
    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f and len(red_bullets) < MAX_BULLETS:
                    bullet = pygame.Rect(red.x + red.width, red.y + red.height // 2 - 2, 10, 5)
                    red_bullets.append(bullet)
                    BULLET_FIRE_SOUND.play()
                if event.key == pygame.K_PERIOD and len(yellow_bullets) < MAX_BULLETS:
                    bullet = pygame.Rect(yellow.x, yellow.y + yellow.height // 2 - 2, 10, 5)
                    yellow_bullets.append(bullet)
                    BULLET_FIRE_SOUND.play()
        
            if event.type == RED_HIT:
                red_health -= 1
                BULLET_HIT_SOUND.play()
        
            if event.type == YELLOW_HIT:
                yellow_health -= 1
                BULLET_HIT_SOUND.play()
        
        winner_text = ''
        if red_health <= 0:
            winner_text = 'Yellow wins!'
        
        if yellow_health <= 0:
            winner_text = 'Red wins!'
            
        if winner_text != '':
            draw_winner(winner_text)
            break
        
        # Handle keyboard events.
        keys_pressed = pygame.key.get_pressed()
        handle_yellow_movement(keys_pressed, yellow)
        handle_red_movement(keys_pressed, red)
        
        # Handle bullet actions/collisions.
        handle_bullets(yellow_bullets, red_bullets, yellow, red)
        
        # Call the draw_window function to handle drawing.
        draw_window(red, yellow, red_bullets, yellow_bullets, red_health, yellow_health)
    main()
    
if __name__ == '__main__':
    main()