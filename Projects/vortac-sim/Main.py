# Python imports
import pygame as PG
import os

# Custom imports
import Draw_Utils as DU
import Aircraft as AC

# CONSTANTS definitions
#####################################################################
# pygame initialization CONSTANTS
WIDTH, HEIGHT = 1200, 800
CANVAS = PG.display.set_mode((WIDTH, HEIGHT))
PG.font.init()
GENERIC_FONT = PG.font.SysFont('Helvetica', 18)
PG.display.set_caption('VORTAC Sim: Air Traffic Control Radar') # CANVAS title.
FPS = 60 # frames per second.

# Aircraft constants.
#   Aircraft icon marker.
AIRCRAFT_IMAGE = PG.image.load(os.path.join('assets', 'aircraft_blue.png'))
#   Aircraft speed scalar.
AIRCRAFT_VELOCITY_SCALAR = 0.001
#####################################################################

def main():
    # Variables to control main game loop.
    clock = PG.time.Clock()
    running = True
    
    # Create a new aircraft object.
    aircraft = AC.Aircraft('SDM147', 'B77W', 600, 400, 250, 15000, 360, [])
    AIRCRAFTS = [aircraft]
    
    turning_right = False
    turn_progress = 0
    turn_increment = 90 / FPS
    
    def handle_default_movement(aircraft: AC.Aircraft):
        if aircraft.heading == 0 or aircraft.heading == 360:
            aircraft.y_loc -= aircraft.speed * AIRCRAFT_VELOCITY_SCALAR
        elif aircraft.heading == 90:
            aircraft.x_loc += aircraft.speed * AIRCRAFT_VELOCITY_SCALAR
        elif aircraft.heading == 180:
            aircraft.y_loc += aircraft.speed * AIRCRAFT_VELOCITY_SCALAR
        elif aircraft.heading == 270:
            aircraft.x_loc -= aircraft.speed * AIRCRAFT_VELOCITY_SCALAR
    
    def handle_user_movement(keys_pressed, aircraft: AC.Aircraft):
        if keys_pressed == PG.K_UP:
            aircraft.y_loc -= aircraft.speed * AIRCRAFT_VELOCITY_SCALAR
            aircraft.heading = 0
        elif keys_pressed == PG.K_RIGHT:
            turning_right = True
            turn_progress = 0
            print('test')
            handle_right_turn(aircraft)
        elif keys_pressed == PG.K_DOWN:
            aircraft.y_loc += aircraft.speed * AIRCRAFT_VELOCITY_SCALAR
            aircraft.heading = 180
        elif keys_pressed == PG.K_LEFT:
            aircraft.x_loc -= aircraft.speed * AIRCRAFT_VELOCITY_SCALAR
            aircraft.heading = 270
    
    def handle_right_turn(aircraft: AC.Aircraft):
        nonlocal turning_right, turn_progress
        if turning_right:
            aircraft.heading += turn_increment
            turn_progress += turn_increment
            
            # if aircraft.heading >= 360:
            #     aircraft.heading -= 360
            # if turn_progress >= 90:
            #     turning_right = False
        handle_default_movement(aircraft)

            
        
    def handle_left_turn(aircraft: AC.Aircraft):
        pass
        
    
    # Main game loop.
    while running:
        clock.tick(FPS) # Limit the frame rate to FPS.
        
        # Event handling
        for event in PG.event.get():
            if event.type == PG.QUIT:
                running = False
                PG.quit()
            if event.type == PG.KEYDOWN:
                handle_user_movement(event.key, aircraft)
                
        handle_default_movement(aircraft)
        # handle_right_turn(aircraft)
        
        DU.Draw_Utils(CANVAS).draw_canvas(AIRCRAFTS)
        
        PG.display.flip()
        # main()

if __name__ == '__main__':
    main()