class Aircraft:
    def __init__(self, flight_number: str, ac_type: str, x_loc: float, y_loc: float, speed: float, altitude: float, heading: float):
        print('\nAircraft.__init__() called -->')
        if flight_number is None or len(flight_number) < 4:
            print('***ERROR***: Aircraft initialization failed')
            print('\tInvalid flight number. Flight number must be at least 4 characters long.')
            return
        self.flight_number = flight_number
        self.ac_type = ac_type
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.speed = speed
        self.altitude = altitude
        self.heading = heading
        print('\tAircraft initialized successfully.')
    
    def update_flight_number(self, new_flight_number: str):
        print('\nAircraft.update_flight_number() called -->')
        if new_flight_number is None or len(new_flight_number) < 4:
            print('***ERROR***: Failed to update aircraft flight number')
            print('\tInvalid flight number. Flight number must be at least 4 characters long.')
            return
        self.flight_number = new_flight_number
        print('\tAircraft flight number updated successfully.')
    
    def update_ac_type(self, new_ac_type: str):
        print('\nAircraft.update_ac_type() called -->')
        if new_ac_type is None or len(new_ac_type) < 4:
            print('***ERROR***: Failed to update aircraft type')
            print('\tInvalid aircraft type. Type must be at least 4 characters long.')
            return
        self.ac_type = new_ac_type
        print('\tAircraft type updated successfully.')
    
    def update_x_loc(self, new_x_loc: float):
        print('\nAircraft.update_x_loc() called -->')
        if new_x_loc is None or new_x_loc < 0:
            print('***ERROR***: Failed to update aircraft location')
            print('\tInvalid x-coordinate. X-coordinate must be a non-negative numeric value.')
            return
        self.x_loc = new_x_loc
        print('\tAircraft x-coordinate updated successfully.')
    
    def update_y_loc(self, new_y_loc: float):
        print('\nAircraft.update_y_loc() called -->')
        if new_y_loc is None or new_y_loc < 0:
            print('***ERROR***: Failed to update aircraft location')
            print('\tInvalid y-coordinate. Y-coordinate must be a non-negative numeric value.')
            return
        self.y_loc = new_y_loc
        print('\tAircraft y-coordinate updated successfully.')
    
    def update_speed(self, new_speed: float):
        print('\nAircraft.update_speed() called -->')
        if new_speed is None or new_speed < 0:
            print('***ERROR***: Failed to update aircraft speed')
            print('\tInvalid speed. Speed must be a non-negative numeric value.')
            return
        self.speed = new_speed
        print('\tAircraft speed updated successfully.')
    
    def update_altitude(self, new_altitude: float):
        print('\nAircraft.update_altitude() called -->')
        if new_altitude is None or new_altitude < 0:
            print('***ERROR***: Failed to update aircraft altitude')
            print('\tInvalid altitude. Altitude must be a non-negative numeric value.')
            return
        self.altitude = new_altitude
        print('\tAircraft altitude updated successfully.')
        
    def update_heading(self, new_heading: float):
        print('\nAircraft.update_heading() called -->')
        if new_heading is None or (new_heading < 0 or new_heading > 360):
            print('***ERROR***: Failed to update aircraft heading')
            print('\tInvalid heading. Heading must be between 0 and 360 degrees.')
            return
        self.heading = new_heading
        print('\tAircraft heading updated successfully.')