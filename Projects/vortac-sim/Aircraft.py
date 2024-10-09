import time
#####################################################################
# Aircraft CONSTANTS.
AIRCRAFT_VELOCITY_SCALAR = 0.001
#####################################################################
class Aircraft:
    def __init__(self, flight_number: str, ac_type: str, x_loc: float, y_loc: float, speed: float, altitude: float, heading: float, flight_plan: list):
        self.flight_number = flight_number
        self.ac_type = ac_type
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.speed = speed
        self.altitude = altitude
        self.heading = heading
        self.flight_plan = flight_plan
        
    def handle_movement_v1(self, target_x_loc: float, target_y_loc: float, target_speed: float, target_altitude: float, target_heading: float, duration: float):
        print('\nAircraft.handle_movement_v1() called...')
    
    def update_flight_number(self, new_flight_number: str):
        self.flight_number = new_flight_number
    
    def update_ac_type(self, new_ac_type: str):
        self.ac_type = new_ac_type
    
    def update_x_loc(self, new_x_loc: float):
        self.x_loc = new_x_loc
    
    def update_y_loc(self, new_y_loc: float):
        self.y_loc = new_y_loc
    
    def update_speed(self, new_speed: float):
        self.speed = new_speed
        
    def update_altitude(self, new_altitude: float):
        self.altitude = new_altitude

    def update_heading(self, new_heading: float):
        self.heading = new_heading