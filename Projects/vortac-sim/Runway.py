class Runway:
    def __init__(self, name: str, lenght: float, heading: float, start_x_loc: float, start_y_loc: float,        end_x_loc: float, end_y_loc: float, active: bool = True):
        print('\nRunway.__init__() called -->')
        if name is None or len(name) < 2:
            print('***ERROR***: Runway initialization failed')
            print('\tInvalid name. Name must be at least 2 characters long.')
        self.name = name
        self.lenght = lenght
        self.heading = heading
        self.start_x_loc = start_x_loc
        self.start_y_loc = start_y_loc
        self.end_x_loc = end_x_loc
        self.end_y_loc = end_y_loc
        self.active = active
        print('\tRunway initialized successfully.')