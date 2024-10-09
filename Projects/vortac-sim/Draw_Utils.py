import pygame as PG

from Main import GENERIC_FONT
from Main import AIRCRAFT_IMAGE

import Aircraft as AC

class Draw_Utils:
    def __init__(self, canvas):
        self.CANVAS = canvas

    """
        function draw_aircraft()
        - In charge of drawing an aircraft on the canvas.
    """
    def draw_aircraft(self, aircraft: AC.Aircraft):
        # AIRCRAFT_IMAGE_CORRECT_HEADING = PG.transform.rotate(AIRCRAFT_IMAGE, aircraft.heading)
        AIRCRAFT_IMAGE_CORRECT_HEADING = PG.transform.rotate(AIRCRAFT_IMAGE, 360 - aircraft.heading)
        self.CANVAS.blit(AIRCRAFT_IMAGE_CORRECT_HEADING, (aircraft.x_loc, aircraft.y_loc))
        # Text for aircraft tag.
        aircraft_tag_flight_number = GENERIC_FONT.render(aircraft.flight_number, True, '#ffffff')
        aircraft_tag_ac_type = GENERIC_FONT.render(aircraft.ac_type, True, '#ffffff')
        aircraft_tag_speed = GENERIC_FONT.render(f'{aircraft.speed}kts', True, '#ffffff')
        aircraft_tag_altitude = GENERIC_FONT.render(f'{aircraft.altitude}ft', True, '#ffffff')
        self.CANVAS.blit(aircraft_tag_flight_number, (aircraft.x_loc, aircraft.y_loc - 40))
        self.CANVAS.blit(aircraft_tag_ac_type, (aircraft.x_loc, aircraft.y_loc - 60))
        self.CANVAS.blit(aircraft_tag_speed, (aircraft.x_loc + 80, aircraft.y_loc - 40))
        self.CANVAS.blit(aircraft_tag_altitude, (aircraft.x_loc + 80, aircraft.y_loc - 60))
        
    """
        function draw_canvas()
        - In charge of drawing the entire canvas.
        - Handles drawing of aicrafts.
    """
    def draw_canvas(self, AIRCRAFTS: list):
        # Filling CANVAS with background color.
        self.CANVAS.fill('#262626')
        
        # Drawing aircrafts from AIRCRAFTS list.
        for aircraft in AIRCRAFTS:
            self.draw_aircraft(aircraft)
        
        PG.display.update()