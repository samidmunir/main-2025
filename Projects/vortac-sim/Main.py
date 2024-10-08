# Python imports
import pygame as PG
import os

# Custom imports
import Draw_Utils as DU
import Airport as AP
import Aircraft as AC

PG.font.init()

# CONSTANTS definitions
#####################################################################
# pygame initializatio CONSTANTS
WIDTH, HEIGHT = 1200, 800
CANVAS = PG.display.set_mode((WIDTH, HEIGHT))
GENERIC_FONT = PG.font.SysFont('Helvetica', 18)
PG.display.set_caption('VORTAC Sim: Air Traffic Control Radar') # CANVAS title.
FPS = 60 # frames per second.

# Aircraft icon marker.
AIRCRAFT_IMAGE = PG.image.load(os.path.join('assets', 'aircraft_blue.png'))

# Aircraft movement constants.
AIRCRAFT_VELOCITY_SCALAR = 0.05
#####################################################################

def create_new_airport(name: str, city: str, country: str, loc_x: float, loc_y: float, number_of_runways: int, runways: list, active: bool):
    new_airport = AP.Airport(name, city, country, loc_x, loc_y, number_of_runways, runways, active)
    return new_airport

def create_new_aircraft(flight_number: str, ac_type: str, x_loc: float, y_loc: float, speed: float, altitude: float, heading: float):
    new_aircraft = AC.Aircraft(flight_number, ac_type, x_loc, y_loc, speed, altitude, heading)
    return new_aircraft

def main():
    clock = PG.time.Clock()
    running = True
    aircraft_1 = create_new_aircraft('SDM147', 'B77W', 800, 200, 250, 12000, 220)
    aircraft_2 = create_new_aircraft('UAL549', 'B739', 200, 200, 265, 15000, 162)
    aircraft_3 = create_new_aircraft('AAL1245', 'A20N', 125, 700, 300, 39000, 42)
    AIRCRAFTS = [aircraft_1, aircraft_2, aircraft_3]
    
    # Create the permanent airport.
    airport_1 = create_new_airport('N77X', 'Somerset', 'United States', 600, 400, 1, ['36'], True)
    airport_2 = create_new_airport('KLGA', 'New York', 'United States', 1000, 100, 2, ['36'], True)
    AIRPORTS = [airport_1, airport_2]
        
    # Create permanent airway vector lines.
    airway_1 = ((1010, 100), (1010, 50))
    airway_2 = ((1010, 50), (910, 50))
    airway_3 = ((910, 50), (910, 100))
    airway_4 = ((910, 100), (810, 200))
    airway_5 = ((910, 100), (910, 200))
        
    airway_6 = ((610, 400), (610, 300))
    airway_7 = ((610, 300), (510, 300))
    airway_8 = ((510, 300), (410, 400))
    airway_9 = ((510, 300), (510, 400))
    AIRWAYS = [airway_1, airway_2, airway_3, airway_4, airway_5, airway_6, airway_7, airway_8, airway_9]
        
    while running:
        clock.tick(FPS) # Limit the frame rate to FPS.
        
        # Event handling
        for event in PG.event.get():
            if event.type == PG.QUIT:
                running = False
                PG.quit()
            if event.type == PG.KEYDOWN:
                if event.key == PG.K_UP:
                    aircraft_1.y_loc -= 5
                    aircraft_1.heading = 360
                if event.key == PG.K_DOWN:
                    aircraft_1.y_loc += 5
                    aircraft_1.heading = 180
                if event.key == PG.K_LEFT:
                    aircraft_1.x_loc -= 5
                    aircraft_1.heading = 270
                if event.key == PG.K_RIGHT:
                    aircraft_1.x_loc += 5
                    aircraft_1.heading = 90
        if aircraft_1.heading == 360:
            aircraft_1.y_loc -= 1
        elif aircraft_1.heading == 180:
            aircraft_1.y_loc += 1
        elif aircraft_1.heading == 270:
            aircraft_1.x_loc -= 1
        else:
            aircraft_1.x_loc += 1
                
        DU.Draw_Utils(CANVAS).draw_canvas(AIRPORTS, AIRWAYS, AIRCRAFTS)
        # main()

if __name__ == '__main__':
    main()