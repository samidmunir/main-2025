class Aircraft:
    def __init__(self, flight_number: str, type: str, x_loc: float, y_loc: float, speed: float, altitude: float, heading: float):
        print(f'\nAircraft.__init__() called -->')
        if flight_number is None or len(flight_number) < 3:
            print('***ERROR***: Aircraft initialization failed')
            print('\tInvalid flight number. Flight number must be at least 3 characters long.')
            return
        self.flight_number = flight_number
        if type is None or len(type) < 3:
            print('***ERROR***: Aircraft initialization failed')
            print('\tInvalid aircraft type. Type must be at least 3 characters long.')
            return
        self.type = type
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.speed = speed
        self.altitude = altitude
        if heading < 0 or heading > 360:
            print('***ERROR***: Aircraft initialization failed')
            print('\tInvalid heading. Heading must be between 0 and 360 degrees.')
            return
        self.heading = heading
        print(f'\tAircraft initialized successfully [{self.flight_number}]')