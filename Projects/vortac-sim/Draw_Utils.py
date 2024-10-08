import pygame as PG

from Main import GENERIC_FONT
from Main import AIRCRAFT_IMAGE

import Airport as AP
import Aircraft as AC

class Draw_Utils:
    def __init__(self, canvas):
        self.CANVAS = canvas
        
    # def draw_airway_line(self, start_x_loc, start_y_loc, end_x_loc, end_y_loc):
    #     PG.draw.line(self.CANVAS, '#0000ff', (start_x_loc, start_y_loc), (end_x_loc, end_y_loc), 3)
    
    def draw_aircraft(self, aircraft: AC.Aircraft):
        # AIRCRAFT_IMAGE_CORRECT_HEADING = PG.transform.rotate(AIRCRAFT_IMAGE, aircraft.heading)
        AIRCRAFT_IMAGE_CORRECT_HEADING = PG.transform.rotate(AIRCRAFT_IMAGE, 360 - aircraft.heading)
        self.CANVAS.blit(AIRCRAFT_IMAGE_CORRECT_HEADING, (aircraft.x_loc, aircraft.y_loc))
        # Text for aircraft tag.
        aircraft_tag_flight_number = GENERIC_FONT.render(aircraft.flight_number, True, '#ffffff')
        aircraft_tag_ac_type = GENERIC_FONT.render(aircraft.ac_type, True, '#ffffff')
        aircraft_tag_speed = GENERIC_FONT.render(f'{aircraft.speed}kts', True, '#ffffff')
        aircraft_tag_altitude = GENERIC_FONT.render(f'{aircraft.altitude}ft', True, '#ffffff')
        self.CANVAS.blit(aircraft_tag_flight_number, (aircraft.x_loc, aircraft.y_loc - 60))
        self.CANVAS.blit(aircraft_tag_ac_type, (aircraft.x_loc, aircraft.y_loc - 80))
        self.CANVAS.blit(aircraft_tag_speed, (aircraft.x_loc + 80, aircraft.y_loc - 60))
        self.CANVAS.blit(aircraft_tag_altitude, (aircraft.x_loc + 80, aircraft.y_loc - 80))
        
    
    def draw_airport(self, airport: AP.Airport):
        # Creating rect for airport marker.
        airport_rect = PG.Rect(airport.loc_x, airport.loc_y, 25, 25)
        PG.draw.rect(self.CANVAS, '#ffffff', airport_rect)
        # Creating label for airport name.
        airport_label = GENERIC_FONT.render(airport.name, True, '#ffffff')
        self.CANVAS.blit(airport_label, (airport.loc_x + 35, airport.loc_y + 5))
        # Drawing a line for runway marker.
        PG.draw.line(self.CANVAS, '#00ff00', (airport.loc_x + 12, airport.loc_y + 25), (airport.loc_x + 12, airport.loc_y + 100), 3)
        # Creating label for runway number/name.
        runway_label = GENERIC_FONT.render('36', True, '#ffffff')
        self.CANVAS.blit(runway_label, (airport.loc_x + 3, airport.loc_y + 110))
    
    def draw_canvas(self, airports, airways, aircrafts):
        # Filling CANVAS with background color.
        self.CANVAS.fill('#262626')
        # Drawing airport and runway markers and labels.
        for airport in airports:
            self.draw_airport(airport)
        
        # Drawing airway lines.    
        for airway in airways:
            PG.draw.line(self.CANVAS, '#0000ff', airway[0], airway[1], 1)
        
        # Drawing aircrafts.
        for aircraft in aircrafts:
            self.draw_aircraft(aircraft)
        
        PG.display.update()